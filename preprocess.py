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
    """

    resize: Tuple[int, int] = (128, 128)
    grayscale: bool = True
    interpolation: str = "bilinear"
    normalize: Literal["float01", "uint8"] = "float01"
    denoise: bool = False
    denoise_radius: float = 2.0
    equalize: bool = False
    valid_extensions: Tuple[str, ...] = field(default=_DEFAULT_EXTENSIONS)

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

        self.valid_extensions = tuple(ext.lower() for ext in self.valid_extensions)

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


def _load_and_process_image(path: Path, config: PreprocessConfig) -> Optional[np.ndarray]:
    """Load a single image and run the preprocessing pipeline on it.

    Returns the processed array, or None if the file could not be read
    (corrupt/truncated/unsupported), in which case it is logged and skipped.
    """
    try:
        with Image.open(path) as img:
            img.load()  # force decode now, so truncated/corrupt files fail here
            img = img.resize(config.resize, resample=config.resample_filter)
            img = img.convert("L") if config.grayscale else img.convert("RGB")

            if config.denoise:
                img = img.filter(ImageFilter.GaussianBlur(radius=config.denoise_radius))
            if config.equalize:
                img = ImageOps.equalize(img)

            arr = np.array(img)
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        logger.warning("Skipping unreadable image %s: %s", path, exc)
        return None

    if config.normalize == "float01":
        arr = arr.astype(np.float32) / 255.0
    else:
        arr = arr.astype(np.uint8)
    return arr


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
