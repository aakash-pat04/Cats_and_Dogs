"""Configurable image preprocessing pipeline for the Cats vs Dogs assignment.

Loads raw JPEGs from a "cats" directory and a "dogs" directory, and applies a
deterministic pipeline (load -> resize -> color conversion -> optional
denoise/equalize -> normalize) driven by a `PreprocessConfig`. Corrupt or
unreadable files are skipped and logged rather than raising.

Feature extraction (HOG/LBP) and model training are separate, later steps and
are intentionally not part of this module.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError

from subject import apply_background, crop_box, crop_signature, load_mask, mask_signature, prepare_mask

logger = logging.getLogger(__name__)

PathLike = Union[str, Path]

# Maps a friendly interpolation name to the corresponding Pillow resampling
# filter, so PreprocessConfig can stay JSON/repr-friendly (plain strings)
# instead of holding a PIL enum member.
_INTERPOLATION_METHODS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
    "box": Image.Resampling.BOX,
    "hamming": Image.Resampling.HAMMING,
}

_DEFAULT_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class PreprocessConfig:
    """Knobs for the preprocessing pipeline.

    Attributes:
        resize: Target (width, height) in pixels.
        grayscale: If True, convert to single-channel grayscale (needed for
            HOG/LBP). If False, keep 3-channel RGB (e.g. for future
            color-histogram features).
        interpolation: Resize filter name, one of "nearest", "bilinear",
            "bicubic", "lanczos", "box", "hamming".
        normalize: "float01" scales pixels to float32 in [0, 1]. "uint8"
            leaves pixels as uint8 in [0, 255].
        denoise: If True, apply a Gaussian blur before normalization.
        denoise_radius: Gaussian blur radius (Pillow's analog of kernel
            size) used when `denoise` is True. Larger = more smoothing.
        equalize: If True, apply histogram equalization (per channel for
            RGB) before normalization.
        valid_extensions: File suffixes (case-insensitive) treated as
            images; anything else in the directory is ignored.
        crop_mode: Square crop applied before resizing (see subject.py):
            "stretch" (no crop — the original behavior, distorts non-square
            images), "center_square", or "saliency" (zoom on the subject).
            "mask" (tight square around the segmentation mask; needs
            `mask_model`).
        flip: If True, mirror the image horizontally as the final step
            (used only to build flip-augmented *training* features).
        sharpen: Deblurring applied after resize: "none", "unsharp"
            (unsharp masking) or "rl" (Richardson-Lucy deconvolution with a
            small Gaussian point-spread function).
        mask_model: Name of the segmentation model whose precomputed masks
            (see segment.py / subject.mask_path_for) to use, or None.
        background: "keep", "gray" (flat gray outside the mask) or "blur"
            (heavily blurred background). Anything but "keep" needs
            `mask_model`.
        mask_post: "clean" (morphological cleanup + largest component +
            hole filling — the post-processing experiment) or "raw".

    Every field from `crop_mode` on is excluded from `repr` on purpose:
    feature caches are keyed on `repr(config)`, and leaving them out keeps
    every cache built before these options existed valid. `cache_extras()`
    carries them into cache signatures instead, only when non-default.
    """

    resize: Tuple[int, int] = (128, 128)
    grayscale: bool = True
    interpolation: str = "bilinear"
    normalize: Literal["float01", "uint8"] = "float01"
    denoise: bool = False
    denoise_radius: float = 2.0
    equalize: bool = False
    valid_extensions: Tuple[str, ...] = field(default=_DEFAULT_EXTENSIONS)
    crop_mode: Literal["stretch", "center_square", "saliency", "mask"] = field(default="stretch", repr=False)
    flip: bool = field(default=False, repr=False)
    sharpen: Literal["none", "unsharp", "rl"] = field(default="none", repr=False)
    mask_model: Optional[str] = field(default=None, repr=False)
    background: Literal["keep", "gray", "blur"] = field(default="keep", repr=False)
    mask_post: Literal["clean", "raw"] = field(default="clean", repr=False)

    def __post_init__(self) -> None:
        width, height = self.resize
        if width <= 0 or height <= 0:
            raise ValueError(f"resize must be positive (width, height), got {self.resize}")

        if self.interpolation not in _INTERPOLATION_METHODS:
            valid = ", ".join(sorted(_INTERPOLATION_METHODS))
            raise ValueError(f"interpolation must be one of {{{valid}}}, got {self.interpolation!r}")

        if self.normalize not in ("float01", "uint8"):
            raise ValueError(f"normalize must be 'float01' or 'uint8', got {self.normalize!r}")

        if self.denoise and self.denoise_radius <= 0:
            raise ValueError(f"denoise_radius must be positive, got {self.denoise_radius}")

        if self.crop_mode not in ("stretch", "center_square", "saliency", "mask"):
            raise ValueError(f"crop_mode must be 'stretch', 'center_square', 'saliency' or 'mask', got {self.crop_mode!r}")
        if self.sharpen not in ("none", "unsharp", "rl"):
            raise ValueError(f"sharpen must be 'none', 'unsharp' or 'rl', got {self.sharpen!r}")
        if self.background not in ("keep", "gray", "blur"):
            raise ValueError(f"background must be 'keep', 'gray' or 'blur', got {self.background!r}")
        if self.mask_post not in ("clean", "raw"):
            raise ValueError(f"mask_post must be 'clean' or 'raw', got {self.mask_post!r}")
        if (self.crop_mode == "mask" or self.background != "keep") and not self.mask_model:
            raise ValueError("crop_mode='mask' and background replacement need mask_model")

        self.valid_extensions = tuple(ext.lower() for ext in self.valid_extensions)

    def cache_extras(self) -> str:
        """Signature suffix for options that `repr` omits; empty when they're at their defaults."""
        extras = []
        if self.crop_mode != "stretch":
            extras.append(f"crop_mode={crop_signature(self.crop_mode)}")
        if self.flip:
            extras.append("flip=True")
        if self.sharpen != "none":
            extras.append(f"sharpen={self.sharpen}{_SHARPEN_PARAMS[self.sharpen]}")
        if self.mask_model:
            extras.append(
                f"mask_model={self.mask_model},background={self.background},mask_post={self.mask_post},{mask_signature()}"
            )
        return ";".join(extras)

    @property
    def resample_filter(self) -> Image.Resampling:
        """The Pillow resampling filter corresponding to `interpolation`."""
        return _INTERPOLATION_METHODS[self.interpolation]


def _list_image_paths(directory: PathLike, extensions: Tuple[str, ...]) -> List[Path]:
    """Return image file paths in `directory`, sorted for determinism."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    paths = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    return sorted(paths)


# Parameters of each sharpening method (folded into cache signatures).
_UNSHARP = dict(radius=2, percent=150, threshold=3)
_RL_SIGMA = 1.0
_RL_ITERATIONS = 10
_SHARPEN_PARAMS = {
    "unsharp": f"({_UNSHARP})",
    "rl": f"(gaussian_psf_sigma={_RL_SIGMA},iter={_RL_ITERATIONS})",
}


def _gaussian_psf(sigma: float) -> np.ndarray:
    radius = int(np.ceil(3 * sigma))
    ax = np.arange(-radius, radius + 1)
    kernel = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma**2))
    return kernel / kernel.sum()


def _sharpen(img: Image.Image, method: str) -> Image.Image:
    if method == "none":
        return img
    if method == "unsharp":
        return img.filter(ImageFilter.UnsharpMask(**_UNSHARP))
    if method == "rl":
        from skimage.restoration import richardson_lucy

        if img.mode != "L":
            raise ValueError("Richardson-Lucy sharpening is only implemented for grayscale images")
        arr = np.asarray(img, dtype=np.float64) / 255.0
        out = richardson_lucy(arr, _gaussian_psf(_RL_SIGMA), num_iter=_RL_ITERATIONS, clip=True)
        return Image.fromarray(np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8), mode="L")
    raise ValueError(f"Unknown sharpen method {method!r}")


def load_image_and_mask(path: Path, config: PreprocessConfig) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Load one image, run the preprocessing pipeline, and return `(image, mask)`.

    `mask` is the foreground alpha in [0, 1] carried through the same crop,
    resize and flip as the image (so it's pixel-aligned with it), or None
    when `config.mask_model` is unset. Returns `(None, None)` if the image
    file itself can't be read (logged and skipped, as before); a *missing
    mask* raises instead (see subject.load_mask).
    """
    try:
        with Image.open(path) as img:
            img.load()  # force decode now, so truncated/corrupt files fail here
            alpha = None
            if config.mask_model:
                img = img.convert("RGB")
                alpha = prepare_mask(load_mask(path, config.mask_model, img.size), config.mask_post)
                img = apply_background(img, alpha, config.background)
            box = crop_box(img, config.crop_mode, alpha)
            if box is not None:
                img = img.crop(box)
            img = img.resize(config.resize, resample=config.resample_filter)
            img = img.convert("L") if config.grayscale else img.convert("RGB")

            img = _sharpen(img, config.sharpen)
            if config.denoise:
                img = img.filter(ImageFilter.GaussianBlur(radius=config.denoise_radius))
            if config.equalize:
                img = ImageOps.equalize(img)
            if config.flip:
                img = ImageOps.mirror(img)

            arr = np.array(img)
    except FileNotFoundError:
        if config.mask_model and Path(path).exists():
            raise  # the mask is missing, not the image
        logger.warning("Skipping missing image %s", path)
        return None, None
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        logger.warning("Skipping unreadable image %s: %s", path, exc)
        return None, None

    mask_arr = None
    if alpha is not None:
        m = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
        if box is not None:
            m = m.crop(box)
        m = m.resize(config.resize, resample=Image.Resampling.BILINEAR)
        if config.flip:
            m = ImageOps.mirror(m)
        mask_arr = np.asarray(m, dtype=np.float32) / 255.0

    if config.normalize == "float01":
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.uint8)
    return arr, mask_arr


def _load_and_process_image(path: Path, config: PreprocessConfig) -> Optional[np.ndarray]:
    """Load a single image and run the preprocessing pipeline on it.

    Returns the processed array, or None if the file could not be read
    (corrupt/truncated/unsupported), in which case it is logged and skipped.
    """
    return load_image_and_mask(path, config)[0]


def list_image_paths(
    directory: PathLike,
    extensions: Tuple[str, ...] = _DEFAULT_EXTENSIONS,
) -> List[Path]:
    """Public wrapper around `_list_image_paths`: sorted image paths in `directory`."""
    return _list_image_paths(directory, tuple(ext.lower() for ext in extensions))


def preprocess_paths(
    records: Sequence[Tuple[PathLike, int]],
    config: Optional[PreprocessConfig] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Preprocess an explicit, caller-ordered list of (path, label) records.

    Like `load_dataset`, but takes an explicit list instead of scanning two
    directories, and preserves `records` order exactly (no sorting or
    grouping by label) — useful when the caller has already selected and
    ordered a specific subset of files (e.g. a seeded train/val/test split).

    Returns `(images, labels, paths)` with the same shape/dtype contract as
    `load_dataset`. Corrupt/unreadable files are skipped and logged, same
    as `load_dataset`. Raises RuntimeError if no images were kept.
    """
    config = config or PreprocessConfig()

    images: List[np.ndarray] = []
    labels: List[int] = []
    kept_paths: List[str] = []
    skipped = 0

    for path, label in records:
        arr = _load_and_process_image(Path(path), config)
        if arr is None:
            skipped += 1
            continue
        images.append(arr)
        labels.append(label)
        kept_paths.append(str(path))

    if skipped:
        logger.warning("Skipped %d unreadable/corrupt file(s) out of %d", skipped, len(records))
    if not images:
        raise RuntimeError("No valid images were loaded from the given records.")

    images_array = np.stack(images, axis=0)
    labels_array = np.array(labels, dtype=np.int64)
    return images_array, labels_array, kept_paths


def load_dataset(
    cat_dir: PathLike,
    dog_dir: PathLike,
    config: Optional[PreprocessConfig] = None,
    limit: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Load and preprocess a cats/dogs image dataset.

    Args:
        cat_dir: Directory containing only cat images.
        dog_dir: Directory containing only dog images.
        config: Preprocessing options. Defaults to `PreprocessConfig()`.
        limit: If set, only the first `limit` images (by sorted filename)
            are loaded per class. Useful for a fast sanity-check run;
            leave as None for a full dataset load.

    Returns:
        A tuple `(images, labels, paths)`:
        - images: np.ndarray of shape (N, H, W) if grayscale, or
          (N, H, W, 3) if RGB, dtype float32 or uint8 per `config.normalize`.
        - labels: np.ndarray of shape (N,), int64, 0 = cat, 1 = dog.
        - paths: list of N original file paths (as strings), in the same
          order as `images`/`labels`, for tracing predictions back to files.

        Order is deterministic: all cats (sorted by filename) followed by
        all dogs (sorted by filename). No randomness is introduced here.
    """
    config = config or PreprocessConfig()

    cat_paths = _list_image_paths(cat_dir, config.valid_extensions)
    dog_paths = _list_image_paths(dog_dir, config.valid_extensions)
    if limit is not None:
        cat_paths = cat_paths[:limit]
        dog_paths = dog_paths[:limit]
    logger.info("Found %d cat image(s), %d dog image(s)", len(cat_paths), len(dog_paths))

    images: List[np.ndarray] = []
    labels: List[int] = []
    kept_paths: List[str] = []
    skipped = 0

    for label, paths in ((0, cat_paths), (1, dog_paths)):
        for path in paths:
            arr = _load_and_process_image(path, config)
            if arr is None:
                skipped += 1
                continue
            images.append(arr)
            labels.append(label)
            kept_paths.append(str(path))

    if skipped:
        logger.warning("Skipped %d unreadable/corrupt file(s) out of %d", skipped, len(cat_paths) + len(dog_paths))
    if not images:
        raise RuntimeError("No valid images were loaded from the given directories.")

    images_array = np.stack(images, axis=0)
    labels_array = np.array(labels, dtype=np.int64)
    return images_array, labels_array, kept_paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Adjust these to wherever your Cat/ and Dog/ folders live.
    BASE_DIR = Path(__file__).resolve().parent.parent
    CAT_DIR = BASE_DIR / "data" / "PetImages" / "Cat"
    DOG_DIR = BASE_DIR / "data" / "PetImages" / "Dog"

    # `limit` keeps this sanity check fast; drop it for a real run.
    default_images, default_labels, default_paths = load_dataset(
        CAT_DIR, DOG_DIR, PreprocessConfig(), limit=20
    )
    print(
        f"[default config]  images={default_images.shape} "
        f"dtype={default_images.dtype} labels={default_labels.shape} "
        f"n_paths={len(default_paths)}"
    )

    alt_config = PreprocessConfig(
        resize=(64, 64), grayscale=True, equalize=True, denoise=True, denoise_radius=1.5
    )
    alt_images, alt_labels, alt_paths = load_dataset(CAT_DIR, DOG_DIR, alt_config, limit=20)
    print(
        f"[64x64+denoise+eq] images={alt_images.shape} "
        f"dtype={alt_images.dtype} labels={alt_labels.shape} "
        f"n_paths={len(alt_paths)}"
    )

    rgb_config = PreprocessConfig(resize=(96, 96), grayscale=False, normalize="uint8")
    rgb_images, rgb_labels, rgb_paths = load_dataset(CAT_DIR, DOG_DIR, rgb_config, limit=20)
    print(
        f"[96x96 RGB uint8]  images={rgb_images.shape} "
        f"dtype={rgb_images.dtype} labels={rgb_labels.shape} "
        f"n_paths={len(rgb_paths)}"
    )
