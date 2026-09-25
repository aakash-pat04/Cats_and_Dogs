"""Preprocessing/feature "levels" and cached, parallel feature extraction.

Defines the named preprocessing levels (the original cumulative P0 minimal
-> P1 +equalize -> P2 +denoise, plus the later subject-cropping and
flip-augmentation levels) and feature levels (named combinations of
feature *blocks* such as HOG, LBP, spatial LBP), plus the extraction +
on-disk caching logic that makes running the sweep repeatedly cheap:
each block is computed once per (preprocessing level, split part) and
cached to disk; every downstream consumer (each dataset-size step, each
classifier) just reads/slices the cached matrix instead of recomputing
pixels or gradients.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from joblib import Parallel, delayed
from PIL import Image
from skimage.feature import hog, local_binary_pattern
from skimage.measure import moments_central, moments_hu, moments_normalized

from preprocess import PreprocessConfig, load_image_and_mask

logger = logging.getLogger(__name__)

Record = Tuple[Path, int]
Blocks = Dict[str, np.ndarray]


@dataclass(frozen=True)
class FeatureConfig:
    """Feature-extraction parameters. HOG/LBP defaults match the homework's recommended values.

    Fields added after the original HOG/LBP ones are `repr=False` so that
    `repr(fc)` — part of the legacy hog/lbp cache signature — is unchanged
    and those caches stay valid. Each newer block's own cache signature
    includes its parameters explicitly (see `_BLOCK_PARAMS`).
    """

    hog_orientations: int = 9
    hog_pixels_per_cell: Tuple[int, int] = (8, 8)
    hog_cells_per_block: Tuple[int, int] = (2, 2)
    hog_block_norm: str = "L2-Hys"
    lbp_P: int = 8
    lbp_R: float = 1.0
    lbp_method: str = "uniform"
    hog_coarse_pixels_per_cell: Tuple[int, int] = field(default=(16, 16), repr=False)
    slbp_grid: Tuple[int, int] = field(default=(4, 4), repr=False)
    slbp_radii: Tuple[float, ...] = field(default=(1.0, 2.0), repr=False)
    color_hsv_bins: Tuple[int, int, int] = field(default=(8, 4, 4), repr=False)

    @property
    def lbp_bins(self) -> int:
        """Number of LBP histogram bins. P+2 for method='uniform'."""
        return self.lbp_P + 2


@dataclass(frozen=True)
class PreprocessLevel:
    """A named preprocessing recipe.

    Attributes:
        train_flip: If True, horizontally mirrored copies of the *training*
            images are added as extra training rows (val/test are never
            augmented).
        cache_name: Share feature caches with another level whose `config`
            is identical (e.g. "P0_minimal_flip" reuses "P0_minimal"'s
            unflipped features instead of recomputing them).
        fc: Feature-extraction parameters for this level — they must scale
            with `config.resize` (e.g. 16-px HOG cells at 256x256 describe
            the same regions as 8-px cells at 128x128).
    """

    name: str
    config: PreprocessConfig
    train_flip: bool = False
    cache_name: Optional[str] = None
    fc: FeatureConfig = FeatureConfig()

    @property
    def cache_stem(self) -> str:
        return self.cache_name or self.name


_P0_CONFIG = PreprocessConfig(resize=(128, 128), grayscale=True, normalize="uint8", denoise=False, equalize=False)
# Built on P0 (not P1/P2): equalize/denoise made no difference or slightly
# hurt in the original sweep — see experiment_log.md.
_SQUARE_CONFIG = replace(_P0_CONFIG, crop_mode="center_square")
_SALIENCY_CONFIG = replace(_P0_CONFIG, crop_mode="saliency")

# Round 3 (see RUNBOOK_round3.md): resolution, sharpening, segmentation masks.
# All build on the round-2 winner (center square crop, P0 otherwise).
_R192_CONFIG = replace(_SQUARE_CONFIG, resize=(192, 192))
_R256_CONFIG = replace(_SQUARE_CONFIG, resize=(256, 256))
# Same cell layout as 128x128 with 8/16-px cells, from more source pixels.
_FC_R192 = FeatureConfig(hog_pixels_per_cell=(12, 12), hog_coarse_pixels_per_cell=(24, 24), slbp_radii=(1.0, 2.0, 3.0))
_FC_R256C16 = FeatureConfig(hog_pixels_per_cell=(16, 16), hog_coarse_pixels_per_cell=(32, 32), slbp_radii=(1.0, 2.0, 4.0))
# Finer cell layout (4x the HOG cells, ~44.6k dims) — memory-heavy.
_FC_R256C8 = FeatureConfig(hog_pixels_per_cell=(8, 8), hog_coarse_pixels_per_cell=(16, 16), slbp_radii=(1.0, 2.0, 4.0))

_UNSHARP_CONFIG = replace(_SQUARE_CONFIG, sharpen="unsharp")
_RL_CONFIG = replace(_SQUARE_CONFIG, sharpen="rl")

# Which segmentation model's masks the mask levels use. Must be the same on
# every machine (it's part of the cache signature, so mixing is caught, but
# results from different mask models aren't comparable under one name).
MASK_MODEL = os.environ.get("CATDOG_MASK_MODEL", "isnet-general-use")
_MASK_CONFIG = replace(_P0_CONFIG, crop_mode="mask", mask_model=MASK_MODEL)
_MASK_GRAY_CONFIG = replace(_MASK_CONFIG, background="gray")
_MASK_BLUR_CONFIG = replace(_MASK_CONFIG, background="blur")
_MASK_RAW_CONFIG = replace(_MASK_CONFIG, mask_post="raw")

PREPROCESS_LEVELS: List[PreprocessLevel] = [
    # Original cumulative levels: each builds on the previous one. All share
    # the homework's recommended minimum (resize 128x128, grayscale).
    PreprocessLevel("P0_minimal", _P0_CONFIG),
    PreprocessLevel(
        "P1_equalize",
        PreprocessConfig(resize=(128, 128), grayscale=True, normalize="uint8", denoise=False, equalize=True),
    ),
    PreprocessLevel(
        "P2_denoise",
        PreprocessConfig(
            resize=(128, 128), grayscale=True, normalize="uint8", denoise=True, denoise_radius=1.0, equalize=True
        ),
    ),
    # Subject-focused levels: square crops so the resize stops distorting
    # the animal, optionally zoomed onto the salient region (subject.py).
    PreprocessLevel("P3_square", _SQUARE_CONFIG),
    PreprocessLevel("P4_saliency", _SALIENCY_CONFIG),
    # Train-time horizontal-flip augmentation on top of the above.
    PreprocessLevel("P0_minimal_flip", _P0_CONFIG, train_flip=True, cache_name="P0_minimal"),
    PreprocessLevel("P3_square_flip", _SQUARE_CONFIG, train_flip=True, cache_name="P3_square"),
    PreprocessLevel("P4_saliency_flip", _SALIENCY_CONFIG, train_flip=True, cache_name="P4_saliency"),
    # --- Round 3: resolution ---
    PreprocessLevel("P3_square_r192", _R192_CONFIG, fc=_FC_R192),
    PreprocessLevel("P3_square_flip_r192", _R192_CONFIG, train_flip=True, cache_name="P3_square_r192", fc=_FC_R192),
    PreprocessLevel("P3_square_r256c16", _R256_CONFIG, fc=_FC_R256C16),
    PreprocessLevel(
        "P3_square_flip_r256c16", _R256_CONFIG, train_flip=True, cache_name="P3_square_r256c16", fc=_FC_R256C16
    ),
    # No flip variant on purpose: with flips the pool alone needs ~8 GB of
    # RAM. Compare against P3_square (no flip) instead.
    PreprocessLevel("P3_square_r256c8", _R256_CONFIG, fc=_FC_R256C8),
    # --- Round 3: sharpening / deblurring ---
    PreprocessLevel("P3_square_unsharp", _UNSHARP_CONFIG),
    PreprocessLevel("P3_square_flip_unsharp", _UNSHARP_CONFIG, train_flip=True, cache_name="P3_square_unsharp"),
    PreprocessLevel("P3_square_rl", _RL_CONFIG),
    PreprocessLevel("P3_square_flip_rl", _RL_CONFIG, train_flip=True, cache_name="P3_square_rl"),
    # --- Round 3: segmentation masks (crop to the mask, optionally replace background) ---
    PreprocessLevel("P5_maskcrop", _MASK_CONFIG),
    PreprocessLevel("P5_maskcrop_flip", _MASK_CONFIG, train_flip=True, cache_name="P5_maskcrop"),
    PreprocessLevel("P6_maskcrop_bggray", _MASK_GRAY_CONFIG),
    PreprocessLevel("P6_maskcrop_bggray_flip", _MASK_GRAY_CONFIG, train_flip=True, cache_name="P6_maskcrop_bggray"),
    PreprocessLevel("P7_maskcrop_bgblur", _MASK_BLUR_CONFIG),
    PreprocessLevel("P7_maskcrop_bgblur_flip", _MASK_BLUR_CONFIG, train_flip=True, cache_name="P7_maskcrop_bgblur"),
    # Mask post-processing ablation: same as P5 but without clean_mask.
    PreprocessLevel("P5_maskcrop_raw", _MASK_RAW_CONFIG),
    PreprocessLevel("P5_maskcrop_raw_flip", _MASK_RAW_CONFIG, train_flip=True, cache_name="P5_maskcrop_raw"),
]
PREPROCESS_LEVELS_BY_NAME = {level.name: level for level in PREPROCESS_LEVELS}
# What "--preprocess-level all" means. Deliberately still just the original
# three, so sweep commands already running on other machines keep meaning
# exactly what they meant (and keep writing to the same chunk filenames).
DEFAULT_PREPROCESS_LEVELS: Tuple[str, ...] = ("P0_minimal", "P1_equalize", "P2_denoise")


def extract_hog(image: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    """HOG feature vector for one 2D grayscale image."""
    if image.ndim != 2:
        raise ValueError(f"extract_hog expects a 2D grayscale image, got shape {image.shape}")
    return hog(
        image,
        orientations=fc.hog_orientations,
        pixels_per_cell=fc.hog_pixels_per_cell,
        cells_per_block=fc.hog_cells_per_block,
        block_norm=fc.hog_block_norm,
        feature_vector=True,
    ).astype(np.float32)


def extract_hog_coarse(image: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    """Second, coarser-scale HOG (larger cells) capturing overall body/head shape rather than fine edges."""
    return hog(
        image,
        orientations=fc.hog_orientations,
        pixels_per_cell=fc.hog_coarse_pixels_per_cell,
        cells_per_block=fc.hog_cells_per_block,
        block_norm=fc.hog_block_norm,
        feature_vector=True,
    ).astype(np.float32)


def extract_lbp_histogram(image: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    """Normalized LBP-code histogram for one 2D grayscale image.

    `local_binary_pattern` returns a per-pixel code image, not a
    fixed-length vector — this converts it into a normalized histogram of
    codes (bins = P+2 for method="uniform"), the standard way to turn LBP
    into a compact per-image feature for a classifier.
    """
    if image.ndim != 2:
        raise ValueError(f"extract_lbp_histogram expects a 2D grayscale image, got shape {image.shape}")
    codes = local_binary_pattern(image, P=fc.lbp_P, R=fc.lbp_R, method=fc.lbp_method)
    hist, _ = np.histogram(codes.ravel(), bins=fc.lbp_bins, range=(0, fc.lbp_bins), density=True)
    return hist.astype(np.float32)


def extract_spatial_lbp(image: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    """Per-cell LBP histograms on a grid, at several radii, concatenated.

    Unlike the single global histogram in `extract_lbp_histogram` (10 numbers
    for the whole image, which throws away *where* each texture occurs),
    this keeps a separate 59-bin "nri_uniform" histogram per grid cell and
    radius (the LBPH layout used in face recognition), so fur texture on the
    head vs. body vs. background stay distinguishable. Each cell histogram
    is L1-normalized.
    """
    n_bins = fc.lbp_P * (fc.lbp_P - 1) + 3  # 59 for P=8 with nri_uniform
    rows, cols = fc.slbp_grid
    h, w = image.shape
    parts = []
    for radius in fc.slbp_radii:
        codes = local_binary_pattern(image, P=fc.lbp_P, R=radius, method="nri_uniform").astype(np.int64)
        for r in range(rows):
            for c in range(cols):
                cell = codes[r * h // rows : (r + 1) * h // rows, c * w // cols : (c + 1) * w // cols]
                hist = np.bincount(cell.ravel(), minlength=n_bins).astype(np.float32)
                parts.append(hist / max(hist.sum(), 1.0))
    return np.concatenate(parts)


def extract_color_histogram(rgb: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    """Joint HSV color histogram (L1-normalized) of one RGB uint8 image."""
    hsv = np.asarray(Image.fromarray(rgb).convert("HSV"))
    bins = fc.color_hsv_bins
    idx = [(hsv[..., ch].astype(np.int64) * bins[ch]) // 256 for ch in range(3)]
    flat = (idx[0] * bins[1] + idx[1]) * bins[2] + idx[2]
    hist = np.bincount(flat.ravel(), minlength=bins[0] * bins[1] * bins[2]).astype(np.float32)
    return hist / hist.sum()


def extract_shape(mask: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    """Silhouette descriptor from the (processed, pixel-aligned) foreground mask.

    HOG of the binary mask at the coarse cell size (outline shape: ear
    tips, snout length, body proportions) + log-scaled Hu moments
    (rotation/scale-invariant blob shape) + foreground area and bounding-box
    aspect ratio. Only the mask's geometry is used — nothing from the
    segmentation model beyond "which pixels are foreground".
    """
    fg = (mask > 0.5).astype(np.float64)
    silhouette_hog = hog(
        fg,
        orientations=fc.hog_orientations,
        pixels_per_cell=fc.hog_coarse_pixels_per_cell,
        cells_per_block=fc.hog_cells_per_block,
        block_norm=fc.hog_block_norm,
        feature_vector=True,
    )
    if fg.any():
        hu = moments_hu(moments_normalized(moments_central(fg), 3))
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-30)
        ys, xs = np.nonzero(fg)
        aspect = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
    else:
        hu, aspect = np.zeros(7), 1.0
    extras = np.array([fg.mean(), aspect])
    return np.concatenate([silhouette_hog, hu, extras]).astype(np.float32)


def extract_spatial_lbp_foreground(image: np.ndarray, mask: np.ndarray, fc: FeatureConfig) -> np.ndarray:
    """Like `extract_spatial_lbp`, but each cell histogram counts only foreground (animal) pixels.

    Keeps background texture (carpet, grass, cage bars) out of the fur
    texture statistics. A cell with no foreground gets an all-zero histogram.
    """
    n_bins = fc.lbp_P * (fc.lbp_P - 1) + 3
    rows, cols = fc.slbp_grid
    h, w = image.shape
    fg = mask > 0.5
    parts = []
    for radius in fc.slbp_radii:
        codes = local_binary_pattern(image, P=fc.lbp_P, R=radius, method="nri_uniform").astype(np.int64)
        for r in range(rows):
            for c in range(cols):
                sl = (slice(r * h // rows, (r + 1) * h // rows), slice(c * w // cols, (c + 1) * w // cols))
                hist = np.bincount(codes[sl][fg[sl]], minlength=n_bins).astype(np.float32)
                parts.append(hist / max(hist.sum(), 1.0))
    return np.concatenate(parts)


# Blocks computed from the preprocessed grayscale image.
_GRAY_EXTRACTORS: Dict[str, Callable[[np.ndarray, FeatureConfig], np.ndarray]] = {
    "hog": extract_hog,
    "lbp": extract_lbp_histogram,
    "hog16": extract_hog_coarse,
    "slbp": extract_spatial_lbp,
}
# Blocks computed from an RGB version of the same crop (no equalize/denoise/sharpen).
_COLOR_BLOCKS = ("color",)
# Blocks that need the segmentation mask (levels with a mask_model only).
_MASK_EXTRACTORS: Dict[str, Callable[[np.ndarray, np.ndarray, FeatureConfig], np.ndarray]] = {
    "shape": lambda gray, mask, fc: extract_shape(mask, fc),
    "slbp_fg": extract_spatial_lbp_foreground,
}
ALL_BLOCKS: Tuple[str, ...] = (*_GRAY_EXTRACTORS, *_COLOR_BLOCKS, *_MASK_EXTRACTORS)

# The original two blocks share one cache file per (level, part), in the
# format used since the start of the project. Newer blocks get one file each.
_LEGACY_BLOCKS = ("hog", "lbp")
# Block-specific parameters folded into each newer block's cache signature.
_BLOCK_PARAMS: Dict[str, Callable[[FeatureConfig], str]] = {
    "hog16": lambda fc: f"{fc.hog_orientations}|{fc.hog_coarse_pixels_per_cell}|{fc.hog_cells_per_block}|{fc.hog_block_norm}",
    "slbp": lambda fc: f"{fc.lbp_P}|{fc.slbp_grid}|{fc.slbp_radii}|nri_uniform",
    "color": lambda fc: f"hsv|{fc.color_hsv_bins}",
    "shape": lambda fc: f"{fc.hog_orientations}|{fc.hog_coarse_pixels_per_cell}|{fc.hog_cells_per_block}|hu7|area|aspect",
    "slbp_fg": lambda fc: f"{fc.lbp_P}|{fc.slbp_grid}|{fc.slbp_radii}|nri_uniform|fg",
}

# Named, cumulative combinations of blocks. Order within a tuple is the
# column order of the concatenated feature vector.
FEATURE_LEVELS: Dict[str, Tuple[str, ...]] = {
    "hog": ("hog",),
    "hog_lbp": ("hog", "lbp"),
    "hog_slbp": ("hog", "slbp"),
    "hog2_slbp": ("hog", "hog16", "slbp"),
    "hog2_slbp_color": ("hog", "hog16", "slbp", "color"),
    # Mask levels only:
    "hog2_slbp_shape": ("hog", "hog16", "slbp", "shape"),
    "hog2_slbpfg_shape": ("hog", "hog16", "slbp_fg", "shape"),
}
# What "--feature-level all" means — the original two, for the same reason
# as DEFAULT_PREPROCESS_LEVELS.
DEFAULT_FEATURE_LEVELS: Tuple[str, ...] = ("hog", "hog_lbp")


def blocks_for(feature_levels: Sequence[str]) -> List[str]:
    """Union of blocks needed by `feature_levels`, in first-seen order."""
    needed: List[str] = []
    for level in feature_levels:
        if level not in FEATURE_LEVELS:
            raise ValueError(f"Unknown feature_level {level!r}; expected one of {list(FEATURE_LEVELS)}")
        for block in FEATURE_LEVELS[level]:
            if block not in needed:
                needed.append(block)
    return needed


def build_feature_vector(blocks: Blocks, feature_level: str) -> np.ndarray:
    """Concatenate cached block matrices per the requested feature level."""
    if feature_level not in FEATURE_LEVELS:
        raise ValueError(f"Unknown feature_level {feature_level!r}; expected one of {list(FEATURE_LEVELS)}")
    names = FEATURE_LEVELS[feature_level]
    if len(names) == 1:
        return blocks[names[0]]
    return np.concatenate([blocks[name] for name in names], axis=1)


def _extract_batch(
    records: Sequence[Record], config: PreprocessConfig, fc: FeatureConfig, block_names: Sequence[str]
) -> Blocks:
    """Preprocess + extract the requested blocks for a batch of records (runs in a worker process)."""
    gray_blocks = [b for b in block_names if b in _GRAY_EXTRACTORS]
    color_blocks = [b for b in block_names if b in _COLOR_BLOCKS]
    mask_blocks = [b for b in block_names if b in _MASK_EXTRACTORS]
    if mask_blocks and not config.mask_model:
        raise ValueError(f"Blocks {mask_blocks} need a preprocessing level with a segmentation mask")
    rgb_config = replace(config, grayscale=False, equalize=False, denoise=False, sharpen="none", normalize="uint8")

    rows: Dict[str, List[np.ndarray]] = {b: [] for b in block_names}
    for path, _ in records:
        if gray_blocks or mask_blocks:
            gray, mask = load_image_and_mask(Path(path), config)
            if gray is None:
                # dataset_split.validate_and_filter should have already
                # excluded any undecodable file. A silent drop here would
                # desync row counts from the per-class boundaries the
                # experiment runner relies on for prefix slicing.
                raise RuntimeError(f"{path} failed to preprocess despite passing validate_and_filter")
            for b in gray_blocks:
                rows[b].append(_GRAY_EXTRACTORS[b](gray, fc))
            for b in mask_blocks:
                rows[b].append(_MASK_EXTRACTORS[b](gray, mask, fc))
        if color_blocks:
            rgb, _ = load_image_and_mask(Path(path), rgb_config)
            if rgb is None:
                raise RuntimeError(f"{path} failed to preprocess despite passing validate_and_filter")
            rows["color"].append(extract_color_histogram(rgb, fc))
    return {b: np.stack(v).astype(np.float32) for b, v in rows.items()}


def extract_blocks(
    records: Sequence[Record],
    config: PreprocessConfig,
    fc: FeatureConfig,
    block_names: Sequence[str],
    n_jobs: Optional[int] = None,
    batch_size: int = 256,
) -> Blocks:
    """Feature matrices for `records` (row order preserved), extracted in parallel across CPU cores.

    Every image is independent, so batches are farmed out to worker
    processes and concatenated back in their original order.
    `n_jobs` defaults to the FEATURE_N_JOBS environment variable, else all cores.
    """
    if not records:
        raise ValueError("extract_blocks got an empty list of records")
    if n_jobs is None:
        n_jobs = int(os.environ.get("FEATURE_N_JOBS", "-1"))
    batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]

    t0 = time.perf_counter()
    results = Parallel(n_jobs=n_jobs)(delayed(_extract_batch)(b, config, fc, block_names) for b in batches)
    elapsed = time.perf_counter() - t0
    logger.info(
        "Extracted %s for %d image(s) in %.1fs (%.2f ms/image wall-clock)",
        "+".join(block_names), len(records), elapsed, 1000 * elapsed / len(records),
    )
    # Concatenate one block at a time, dropping each batch's copy as we go,
    # so peak memory stays near 1x (not 2x) the final matrices — matters for
    # the ~4 GB 256x256 HOG pool.
    out: Blocks = {}
    for b in block_names:
        out[b] = np.concatenate([r[b] for r in results], axis=0)
        for r in results:
            del r[b]
    return out


def _records_hash(hasher, records: Sequence[Record]) -> None:
    for path, label in records:
        hasher.update(f"{path}|{label}\n".encode())


def _legacy_signature(records: Sequence[Record], config: PreprocessConfig, fc: FeatureConfig) -> str:
    """Signature of the combined hog+lbp cache file.

    Identical to the original project's signature whenever the newer
    PreprocessConfig options are at their defaults, so pre-existing caches
    keep loading.
    """
    hasher = hashlib.sha256()
    hasher.update(repr(config).encode())
    hasher.update(repr(fc).encode())
    _records_hash(hasher, records)
    extras = config.cache_extras()
    if extras:
        hasher.update(f"extras:{extras}".encode())
    return hasher.hexdigest()


def _block_signature(records: Sequence[Record], config: PreprocessConfig, fc: FeatureConfig, block: str) -> str:
    hasher = hashlib.sha256()
    hasher.update(repr(config).encode())
    hasher.update(f"extras:{config.cache_extras()}|block:{block}|{_BLOCK_PARAMS[block](fc)}".encode())
    _records_hash(hasher, records)
    return hasher.hexdigest()


def _load_if_valid(path: Path, signature: str) -> Optional[np.lib.npyio.NpzFile]:
    if not path.exists():
        return None
    cached = np.load(path, allow_pickle=False)
    if str(cached["signature"]) == signature:
        return cached
    logger.info("Cache at %s is stale (signature mismatch); recomputing", path)
    return None


def load_feature_blocks(
    records: Sequence[Record],
    level: PreprocessLevel,
    fc: FeatureConfig,
    split_part: str,
    cache_dir: Path,
    block_names: Sequence[str],
    force_recompute: bool = False,
    flip: bool = False,
) -> Tuple[Blocks, np.ndarray, List[str]]:
    """Compute (or load cached copies of) the requested feature blocks for `records`.

    Returns `(blocks, labels, paths)`, with `blocks` mapping block name ->
    (N, dim) matrix in `records` order. Only missing/stale blocks are
    computed, in a single parallel pass over the images. Caches:
    - hog + lbp: `{cache_stem}__{part}.npz` (the original format)
    - others:    `{cache_stem}__{part}__{block}.npz`
    each keyed by a signature over the preprocessing/feature config and the
    exact ordered (path, label) list, so a stale cache is recomputed rather
    than silently reused. `flip=True` extracts from mirrored images (for
    train-time augmentation) under split part `{part}_flip`.
    """
    unknown = [b for b in block_names if b not in ALL_BLOCKS]
    if unknown:
        raise ValueError(f"Unknown feature block(s) {unknown}; expected some of {ALL_BLOCKS}")

    config = replace(level.config, flip=True) if flip else level.config
    part = f"{split_part}_flip" if flip else split_part
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{level.cache_stem}__{part}"

    blocks: Blocks = {}
    legacy_path = cache_dir / f"{stem}.npz"
    legacy_sig = _legacy_signature(records, config, fc)
    if any(b in _LEGACY_BLOCKS for b in block_names) and not force_recompute:
        cached = _load_if_valid(legacy_path, legacy_sig)
        if cached is not None:
            blocks.update({"hog": cached["hog"], "lbp": cached["lbp"]})
            logger.info("Loaded cached hog+lbp from %s", legacy_path)
    for b in block_names:
        if b in _LEGACY_BLOCKS or force_recompute:
            continue
        cached = _load_if_valid(cache_dir / f"{stem}__{b}.npz", _block_signature(records, config, fc, b))
        if cached is not None:
            blocks[b] = cached["X"]
            logger.info("Loaded cached %s from %s", b, cache_dir / f"{stem}__{b}.npz")

    missing = [b for b in block_names if b not in blocks]
    if any(b in _LEGACY_BLOCKS for b in missing):
        # The legacy file always holds both, so compute both together.
        missing = list(dict.fromkeys([*_LEGACY_BLOCKS, *missing]))
    if missing:
        computed = extract_blocks(records, config, fc, missing)
        if "hog" in computed:
            np.savez_compressed(
                legacy_path,
                hog=computed["hog"],
                lbp=computed["lbp"],
                labels=np.array([label for _, label in records], dtype=np.int64),
                paths=np.array([str(p) for p, _ in records]),
                signature=np.array(legacy_sig),
            )
            logger.info("Cached hog+lbp to %s", legacy_path)
        for b in missing:
            if b in _LEGACY_BLOCKS:
                continue
            path = cache_dir / f"{stem}__{b}.npz"
            np.savez_compressed(path, X=computed[b], signature=np.array(_block_signature(records, config, fc, b)))
            logger.info("Cached %s to %s", b, path)
        blocks.update(computed)

    labels = np.array([label for _, label in records], dtype=np.int64)
    paths = [str(p) for p, _ in records]
    return {b: blocks[b] for b in block_names}, labels, paths


def training_matrix(
    blocks: Blocks, flip_blocks: Optional[Blocks], feature_level: str, pool_max: int, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """The size-`n` training matrix for `feature_level`, built block by block.

    Slices each block's rows first and only then concatenates the columns,
    so the full-pool concatenation (several GB at 256x256) is never built.
    """
    parts = []
    y = None
    for name in FEATURE_LEVELS[feature_level]:
        X_b, y = training_rows(blocks[name], flip_blocks[name] if flip_blocks is not None else None, pool_max, n)
        parts.append(X_b)
    return (parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)), y


def training_rows(
    X_pool: np.ndarray, X_pool_flip: Optional[np.ndarray], pool_max: int, n: int
) -> Tuple[np.ndarray, np.ndarray]:
    """The size-`n`-per-class training matrix and labels, sliced from the full cached pool.

    The pool is laid out as `pool_max` cats followed by `pool_max` dogs
    (see `dataset_split.training_subset`); the first `n` of each is taken so
    smaller sizes stay nested prefixes of larger ones. With `X_pool_flip`
    (same layout, mirrored images), each class's mirrored copies of those
    same `n` images are added too, doubling the rows.
    """
    parts = [X_pool[:n]]
    if X_pool_flip is not None:
        parts.append(X_pool_flip[:n])
    n_cat = sum(len(p) for p in parts)
    parts.append(X_pool[pool_max : pool_max + n])
    if X_pool_flip is not None:
        parts.append(X_pool_flip[pool_max : pool_max + n])
    X = np.concatenate(parts, axis=0)
    y = np.concatenate([np.zeros(n_cat, dtype=int), np.ones(len(X) - n_cat, dtype=int)])
    return X, y
