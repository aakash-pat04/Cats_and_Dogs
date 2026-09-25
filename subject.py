"""Subject-focused cropping ("zoom into the cat/dog") for preprocessing.

All crops here are square, so the later resize to a square target (e.g.
128x128) no longer distorts the animal's aspect ratio — the default
"stretch" resize squashes ~70% of this dataset's images by more than 20%.

- `center_square_box`: the largest centered square. Cheap, assumes the
  subject is roughly centered (true for most of this dataset).
- `saliency_square_box`: spectral-residual saliency (Hou & Zhang, CVPR
  2007) — a classical, training-free "what stands out" map computed from
  the image's log-amplitude spectrum — used to find where the subject is
  and zoom in on it. No learned model.
- `mask_square_box`: tight square around a foreground segmentation mask
  (produced ahead of time by segment.py with a class-agnostic
  salient-object model — deep learning is allowed in preprocessing per the
  professor, but only the mask is ever used, never any class output).

Masks are also used for background replacement (`apply_background`) and
are optionally cleaned up first (`clean_mask`: the "post-processing"
experiment).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

Box = Tuple[int, int, int, int]  # (left, upper, right, lower), PIL crop convention

# Saliency is computed on a small fixed-size thumbnail, as in the original
# paper — enough resolution to find the subject, and very cheap.
_SALIENCY_SIZE = 64
# Fraction of total saliency "mass" the bounding box must contain along each
# axis (trimming the outer (1 - q)/2 on each side), which is more robust to
# small bright specks than thresholding + largest connected component.
# Spectral-residual saliency is spread fairly widely over these photos: at
# 0.9 mass + 10% margin, 99% of boxes (on 200 val images) grew to the full
# short side — i.e. identical to center_square, no zoom at all. 0.6 + 5%
# gives a median box of ~0.8x the short side, which visibly zooms onto the
# animal's head on many images (and misfires on cage bars/busy fabric on
# some — see results/debug/saliency_grid.png).
_SALIENCY_MASS = 0.60
_SALIENCY_MARGIN = 0.05
# Never zoom in tighter than this fraction of the image's short side: a
# saliency box that small is far more likely to be an eye, a toy or a
# specular highlight than the whole animal.
_MIN_SIDE_FRAC = 0.5


def center_square_box(width: int, height: int) -> Box:
    side = min(width, height)
    left = (width - side) // 2
    upper = (height - side) // 2
    return left, upper, left + side, upper + side


def spectral_residual_saliency(gray: np.ndarray) -> np.ndarray:
    """Spectral-residual saliency map for a small 2D float image (same shape as input)."""
    spectrum = np.fft.fft2(gray)
    log_amplitude = np.log(np.abs(spectrum) + 1e-8)
    phase = np.angle(spectrum)
    residual = log_amplitude - ndimage.uniform_filter(log_amplitude, size=3, mode="nearest")
    saliency = np.abs(np.fft.ifft2(np.exp(residual + 1j * phase))) ** 2
    return ndimage.gaussian_filter(saliency, sigma=2.5)


def _mass_interval(weights: np.ndarray, mass: float) -> Tuple[float, float]:
    """[lo, hi] fractional positions (0..1) containing the central `mass` of a 1D weight profile."""
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    tail = (1.0 - mass) / 2.0
    lo = np.searchsorted(cdf, tail)
    hi = np.searchsorted(cdf, 1.0 - tail)
    n = len(weights)
    return lo / n, (hi + 1) / n


def _square_around(cx: float, cy: float, side: float, width: int, height: int) -> Box:
    """Square of `side` centered as close to (cx, cy) as possible while staying inside the image."""
    side = int(round(min(side, width, height)))
    left = int(round(cx - side / 2))
    upper = int(round(cy - side / 2))
    left = min(max(left, 0), width - side)
    upper = min(max(upper, 0), height - side)
    return left, upper, left + side, upper + side


def saliency_square_box(img: Image.Image) -> Box:
    """Square crop box around the most salient region of `img`, with a small margin."""
    width, height = img.size
    thumb = img.convert("L").resize((_SALIENCY_SIZE, _SALIENCY_SIZE), Image.Resampling.BILINEAR)
    saliency = spectral_residual_saliency(np.asarray(thumb, dtype=np.float64) / 255.0)

    x_lo, x_hi = _mass_interval(saliency.sum(axis=0), _SALIENCY_MASS)
    y_lo, y_hi = _mass_interval(saliency.sum(axis=1), _SALIENCY_MASS)

    box_w = (x_hi - x_lo) * width
    box_h = (y_hi - y_lo) * height
    side = max(box_w, box_h) * (1.0 + 2 * _SALIENCY_MARGIN)
    side = max(side, _MIN_SIDE_FRAC * min(width, height))
    cx = (x_lo + x_hi) / 2 * width
    cy = (y_lo + y_hi) / 2 * height
    return _square_around(cx, cy, side, width, height)


# --- Segmentation masks -----------------------------------------------------

# Margin added around the mask's bounding box before squaring it, and the
# smallest mask (fraction of image pixels) trusted for cropping — below
# that, the segmenter most likely missed the animal, so fall back to the
# center square rather than zooming onto a speck.
_MASK_MARGIN = 0.10
_MASK_MIN_AREA = 0.05
_MASK_THRESHOLD = 0.5
# Background replacement settings.
_BG_GRAY = 128
_BG_BLUR_RADIUS = 8


def mask_root(mask_model: str, image_path: Path) -> Path:
    """Directory holding `mask_model`'s masks: $CATDOG_MASK_ROOT/<model>, else data/masks/<model>.

    `image_path` is `.../data/PetImages/<Cat|Dog>/<name>.jpg`, so the default
    resolves to `.../data/masks/<model>` — a sibling of PetImages.
    """
    override = os.environ.get("CATDOG_MASK_ROOT")
    base = Path(override) if override else Path(image_path).parent.parent.parent / "masks"
    return base / mask_model


def mask_path_for(image_path: Path, mask_model: str) -> Path:
    image_path = Path(image_path)
    return mask_root(mask_model, image_path) / image_path.parent.name / f"{image_path.stem}.png"


def load_mask(image_path: Path, mask_model: str, size: Tuple[int, int]) -> np.ndarray:
    """Soft foreground mask in [0, 1] for `image_path`, at `size` (width, height).

    Raises FileNotFoundError with a pointer to segment.py if the mask hasn't
    been generated/copied onto this machine — silently falling back to "no
    mask" would quietly turn a mask experiment into a different experiment.
    """
    path = mask_path_for(image_path, mask_model)
    if not path.exists():
        raise FileNotFoundError(
            f"No {mask_model} mask for {image_path} (expected {path}). Generate masks with "
            f"`python segment.py run --model {mask_model} --shard i/N` or copy the shared mask folder "
            "onto this machine (see RUNBOOK_round3.md)."
        )
    with Image.open(path) as m:
        m = m.convert("L")
        if m.size != tuple(size):
            m = m.resize(size, Image.Resampling.BILINEAR)
        return np.asarray(m, dtype=np.float32) / 255.0


def clean_mask(binary: np.ndarray) -> np.ndarray:
    """Mask post-processing: morphological open/close, keep the largest component, fill holes.

    Removes the typical segmentation debris (specks of background marked
    as foreground, pinholes in dark fur) that would otherwise inflate the
    bounding box or leave holes in a background-replaced animal.
    """
    if not binary.any():
        return binary
    # Structuring element scaled to image size (~1% of the short side).
    k = max(3, int(round(0.01 * min(binary.shape))) | 1)
    structure = np.ones((k, k), dtype=bool)
    cleaned = ndimage.binary_opening(binary, structure=structure)
    cleaned = ndimage.binary_closing(cleaned, structure=structure)
    labels, n = ndimage.label(cleaned)
    if n == 0:
        return cleaned
    sizes = ndimage.sum(cleaned, labels, index=np.arange(1, n + 1))
    cleaned = labels == (int(np.argmax(sizes)) + 1)
    return ndimage.binary_fill_holes(cleaned)


def prepare_mask(soft: np.ndarray, mask_post: str) -> np.ndarray:
    """Soft mask -> final alpha in [0, 1]: "raw" keeps it as-is; "clean" zeroes everything outside `clean_mask`.

    If the result covers less than `_MASK_MIN_AREA` of the image, the
    segmentation is treated as failed (typically an animal behind a fence
    or cage bars, which salient-object models tend to miss) and the whole
    image counts as foreground: no zoom, and crucially no background
    replacement that would erase the animal itself.
    """
    if mask_post == "raw":
        alpha = soft
    elif mask_post == "clean":
        alpha = soft * clean_mask(soft > _MASK_THRESHOLD)
    else:
        raise ValueError(f"Unknown mask_post {mask_post!r}")
    if (alpha > _MASK_THRESHOLD).mean() < _MASK_MIN_AREA:
        return np.ones_like(alpha)
    return alpha


def mask_square_box(alpha: np.ndarray) -> Optional[Box]:
    """Square box around the foreground (alpha > 0.5) plus a margin; None if the mask is too small to trust."""
    height, width = alpha.shape
    fg = alpha > _MASK_THRESHOLD
    if fg.mean() < _MASK_MIN_AREA:
        return None
    ys, xs = np.nonzero(fg)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    side = max(x1 - x0, y1 - y0) * (1.0 + 2 * _MASK_MARGIN)
    return _square_around((x0 + x1) / 2, (y0 + y1) / 2, side, width, height)


def apply_background(img: Image.Image, alpha: np.ndarray, background: str) -> Image.Image:
    """Replace everything outside the (soft) mask with flat gray or a heavy blur of itself."""
    if background == "keep":
        return img
    if background == "gray":
        bg = Image.new(img.mode, img.size, color=(_BG_GRAY,) * len(img.getbands()))
    elif background == "blur":
        bg = img.filter(ImageFilter.GaussianBlur(radius=_BG_BLUR_RADIUS))
    else:
        raise ValueError(f"Unknown background {background!r}")
    alpha_img = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
    return Image.composite(img, bg, alpha_img)


def mask_signature() -> str:
    return (
        f"mask(margin={_MASK_MARGIN},min_area={_MASK_MIN_AREA},failed=whole_image,thr={_MASK_THRESHOLD},"
        f"gray={_BG_GRAY},blur={_BG_BLUR_RADIUS})"
    )


# --- Dispatch ----------------------------------------------------------------


def crop_signature(crop_mode: str) -> str:
    """The parameters behind `crop_mode`, for feature-cache signatures (so tuning them invalidates caches)."""
    if crop_mode == "saliency":
        return f"{crop_mode}(size={_SALIENCY_SIZE},mass={_SALIENCY_MASS},margin={_SALIENCY_MARGIN},min={_MIN_SIDE_FRAC})"
    return crop_mode


def crop_box(img: Image.Image, crop_mode: str, alpha: Optional[np.ndarray] = None) -> Optional[Box]:
    """The crop box for `crop_mode` (None = no crop). "mask" needs `alpha` at the image's size."""
    if crop_mode == "stretch":
        return None
    if crop_mode == "center_square":
        return center_square_box(*img.size)
    if crop_mode == "saliency":
        return saliency_square_box(img)
    if crop_mode == "mask":
        if alpha is None:
            raise ValueError("crop_mode='mask' needs a mask")
        return mask_square_box(alpha) or center_square_box(*img.size)  # (prepare_mask already handles failed masks)
    raise ValueError(f"Unknown crop_mode {crop_mode!r}")


def crop_subject(img: Image.Image, crop_mode: str) -> Image.Image:
    """Apply the named (mask-free) crop to a PIL image. "stretch" is a no-op (resize handles it)."""
    box = crop_box(img, crop_mode)
    return img if box is None else img.crop(box)
