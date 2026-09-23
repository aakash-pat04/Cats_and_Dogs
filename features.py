"""Preprocessing/feature "levels" and cached HOG+LBP feature extraction.

Defines the cumulative preprocessing levels (P0 minimal -> P1 +equalize ->
P2 +denoise) and feature levels (F0 HOG-only -> F1 HOG+LBP) used by the
experiment sweep, plus the extraction + on-disk caching logic that makes
running that sweep repeatedly cheap: HOG/LBP are computed once per
(preprocessing level, split part) and cached to disk; every downstream
consumer (each dataset-size step, each classifier) just reads/slices the
cached matrix instead of recomputing pixels or gradients.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from skimage.feature import hog, local_binary_pattern

from preprocess import PreprocessConfig, preprocess_paths

logger = logging.getLogger(__name__)

Record = Tuple[Path, int]


@dataclass(frozen=True)
class PreprocessLevel:
    """A named, cumulative preprocessing recipe."""

    name: str
    config: PreprocessConfig


# Cumulative: each level builds on the previous one. All share the
# homework's recommended minimum (resize 128x128, grayscale) as the floor.
PREPROCESS_LEVELS: List[PreprocessLevel] = [
    PreprocessLevel(
        "P0_minimal",
        PreprocessConfig(resize=(128, 128), grayscale=True, normalize="uint8", denoise=False, equalize=False),
    ),
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
]
PREPROCESS_LEVELS_BY_NAME = {level.name: level for level in PREPROCESS_LEVELS}

FEATURE_LEVELS: Tuple[str, ...] = ("hog", "hog_lbp")


@dataclass(frozen=True)
class FeatureConfig:
    """HOG/LBP parameters. Defaults match the homework's recommended values."""

    hog_orientations: int = 9
    hog_pixels_per_cell: Tuple[int, int] = (8, 8)
    hog_cells_per_block: Tuple[int, int] = (2, 2)
    hog_block_norm: str = "L2-Hys"
    lbp_P: int = 8
    lbp_R: float = 1.0
    lbp_method: str = "uniform"

    @property
    def lbp_bins(self) -> int:
        """Number of LBP histogram bins. P+2 for method='uniform'."""
        return self.lbp_P + 2


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


def extract_raw_features(images: np.ndarray, fc: FeatureConfig) -> Tuple[np.ndarray, np.ndarray]:
    """HOG and LBP-histogram matrices for a batch of grayscale images.

    Returns `(hog_matrix, lbp_matrix)` of shape (N, hog_dim) and
    (N, lbp_bins). HOG/LBP are independent per image (no cross-image
    state), so row order is preserved and rows may be freely sliced by the
    caller afterward.
    """
    if images.shape[0] == 0:
        raise ValueError("extract_raw_features got an empty batch of images")

    n = images.shape[0]
    hog_dim = extract_hog(images[0], fc).shape[0]
    lbp_dim = fc.lbp_bins

    hog_matrix = np.empty((n, hog_dim), dtype=np.float32)
    lbp_matrix = np.empty((n, lbp_dim), dtype=np.float32)

    t0 = time.perf_counter()
    for i in range(n):
        hog_matrix[i] = extract_hog(images[i], fc)
        lbp_matrix[i] = extract_lbp_histogram(images[i], fc)
    elapsed = time.perf_counter() - t0
    logger.info("Extracted HOG+LBP for %d image(s) in %.1fs (%.2f ms/image)", n, elapsed, 1000 * elapsed / n)

    return hog_matrix, lbp_matrix


def build_feature_vector(hog_matrix: np.ndarray, lbp_matrix: np.ndarray, feature_level: str) -> np.ndarray:
    """Concatenate raw HOG/LBP matrices per the requested cumulative feature level."""
    if feature_level == "hog":
        return hog_matrix
    if feature_level == "hog_lbp":
        return np.concatenate([hog_matrix, lbp_matrix], axis=1)
    raise ValueError(f"Unknown feature_level {feature_level!r}; expected one of {FEATURE_LEVELS}")


def _cache_signature(records: Sequence[Record], preprocess_level: PreprocessLevel, fc: FeatureConfig) -> str:
    hasher = hashlib.sha256()
    hasher.update(repr(preprocess_level.config).encode())
    hasher.update(repr(fc).encode())
    for path, label in records:
        hasher.update(f"{path}|{label}\n".encode())
    return hasher.hexdigest()


def compute_or_load_raw_features(
    records: Sequence[Record],
    preprocess_level: PreprocessLevel,
    fc: FeatureConfig,
    split_part: str,
    cache_dir: Path,
    force_recompute: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Compute (or load a cached copy of) HOG+LBP features for `records`.

    Returns `(hog_matrix, lbp_matrix, labels, paths)`. Cached to
    `cache_dir/{preprocess_level.name}__{split_part}.npz`, keyed by a
    signature (hash of the preprocessing/feature config and the exact
    ordered list of (path, label) records) so a stale cache from a
    different split/config is automatically recomputed rather than
    silently reused.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{preprocess_level.name}__{split_part}.npz"
    signature = _cache_signature(records, preprocess_level, fc)

    if not force_recompute and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=False)
        if str(cached["signature"]) == signature:
            logger.info("Loaded cached features from %s", cache_path)
            return cached["hog"], cached["lbp"], cached["labels"], list(cached["paths"])
        logger.info("Cache at %s is stale (signature mismatch); recomputing", cache_path)

    images, labels, paths = preprocess_paths(list(records), preprocess_level.config)
    if len(paths) != len(records):
        # dataset_split.validate_and_filter should have already excluded any
        # undecodable file, so every record here is expected to succeed. A
        # silent drop here would desync row counts from the per-class
        # boundaries the experiment runner relies on for prefix slicing.
        raise RuntimeError(
            f"{len(records) - len(paths)} record(s) failed to preprocess for "
            f"{preprocess_level.name}/{split_part}, despite passing validate_and_filter. "
            "Re-run dataset_split.build_split (or investigate the affected files) "
            "before continuing."
        )
    hog_matrix, lbp_matrix = extract_raw_features(images, fc)

    np.savez_compressed(
        cache_path,
        hog=hog_matrix,
        lbp=lbp_matrix,
        labels=labels,
        paths=np.array(paths),
        signature=np.array(signature),
    )
    logger.info("Cached features to %s", cache_path)
    return hog_matrix, lbp_matrix, labels, paths
