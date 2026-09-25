"""Precompute foreground (animal) segmentation masks for every image, once.

Deep learning is allowed in *preprocessing* for this assignment (professor,
2026-09-24), so masks come from a pretrained, class-agnostic salient-object
segmentation network run through `rembg` (U²-Net / IS-Net / BiRefNet, ONNX).
These models output only "which pixels belong to the main subject" — they
have no notion of cat vs. dog — and only that mask is ever used downstream
(cropping, background replacement, silhouette features in subject.py /
features.py). The classifier itself stays classical.

Masks are written as 8-bit soft alpha PNGs, at each image's original size,
to data/masks/<model>/<Cat|Dog>/<name>.png (override the root with
$CATDOG_MASK_ROOT), next to data/PetImages. Every machine must use the same
mask files: generate them once (sharded across machines), then copy the
folder around.

Subcommands:
  pilot   Time a few candidate models on val images and save a visual grid
          per model to results/debug/ — pick the model from these.
  run     Generate masks for one shard of all images (resumable).
  check   Count present/missing masks, flag suspiciously small ones, and
          print a fingerprint to compare across machines.
  preview Show what each mask-based preprocessing level actually feeds the
          feature extractors, for a few val images (inspect before sweeping).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from dataset_split import SplitConfig, build_or_load_split
from subject import mask_path_for, mask_root

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAT_DIR = _PROJECT_ROOT / "data" / "PetImages" / "Cat"
DEFAULT_DOG_DIR = _PROJECT_ROOT / "data" / "PetImages" / "Dog"
_TINY_MASK_FRACTION = 0.05


def _split(results_dir: Path):
    config = SplitConfig(cat_dir=DEFAULT_CAT_DIR, dog_dir=DEFAULT_DOG_DIR)
    return build_or_load_split(config, cache_path=results_dir / "split_cache.json")


def all_image_paths(results_dir: Path) -> List[Path]:
    """Every valid image in the split (train pool + val + test), in a fixed, machine-independent order.

    Uses the same validated split as the experiments, so the known-corrupt
    file is excluded, and sorts by class folder + file name (not absolute
    path) so every machine computes identical shards.
    """
    split = _split(results_dir)
    paths = [*split.cat_train_pool, *split.dog_train_pool]
    paths += [p for p, _ in split.val_records] + [p for p, _ in split.test_records]
    return sorted({Path(p) for p in paths}, key=lambda p: (p.parent.name, p.name))


def _new_session(model: str):
    from rembg import new_session  # imported lazily: only needed for pilot/run

    return new_session(model)


def _predict_mask(session, image_path: Path) -> Image.Image:
    from rembg import remove

    with Image.open(image_path) as img:
        img = img.convert("RGB")
        mask = remove(img, session=session, only_mask=True, post_process_mask=False)
    mask = mask.convert("L")
    if mask.size != img.size:
        mask = mask.resize(img.size, Image.Resampling.BILINEAR)
    return mask


def _parse_shard(spec: str) -> Tuple[int, int]:
    i, n = (int(x) for x in spec.split("/"))
    if not 0 <= i < n:
        raise ValueError(f"--shard must be i/N with 0 <= i < N, got {spec!r}")
    return i, n


def cmd_run(args: argparse.Namespace) -> None:
    paths = all_image_paths(args.results_dir)
    i, n = _parse_shard(args.shard)
    shard = paths[i::n]
    if args.limit:
        shard = shard[: args.limit]
    todo = [p for p in shard if not mask_path_for(p, args.model).exists()]
    logger.info(
        "Model %s, shard %d/%d: %d images (%d already done, %d to do) -> %s",
        args.model, i, n, len(shard), len(shard) - len(todo), len(todo), mask_root(args.model, shard[0]),
    )
    if not todo:
        return

    session = _new_session(args.model)
    t0 = time.perf_counter()
    for done, path in enumerate(todo, start=1):
        out = mask_path_for(path, args.model)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.stem + ".tmp.png")
        _predict_mask(session, path).save(tmp)
        tmp.replace(out)  # atomic: an interrupted run never leaves a half-written mask
        if done % 200 == 0 or done == len(todo):
            elapsed = time.perf_counter() - t0
            rate = done / elapsed
            logger.info(
                "%d/%d masks (%.2f img/s, %.0f min left)", done, len(todo), rate, (len(todo) - done) / rate / 60
            )


def cmd_pilot(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    split = _split(args.results_dir)
    cats = [p for p, y in split.val_records if y == 0][: args.n // 2]
    dogs = [p for p, y in split.val_records if y == 1][: args.n - args.n // 2]
    sample = [Path(p) for p in cats + dogs]
    out_dir = args.results_dir / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_total = len(all_image_paths(args.results_dir))

    for model in [m.strip() for m in args.models.split(",") if m.strip()]:
        session = _new_session(model)
        _predict_mask(session, sample[0])  # warm-up (first call includes graph setup)
        t0 = time.perf_counter()
        masks = [_predict_mask(session, p) for p in sample]
        per_image = (time.perf_counter() - t0) / len(sample)
        fg = [np.asarray(m, dtype=np.float32).mean() / 255.0 for m in masks]
        tiny = np.mean([f < _TINY_MASK_FRACTION for f in fg])
        print(
            f"{model:>24}: {per_image:.3f} s/img -> all {n_total} images: {per_image * n_total / 3600:.1f} h on one "
            f"machine, {per_image * n_total / 7200:.1f} h each on two; tiny/empty masks: {tiny:.0%}"
        )

        # Grid: 4 cats + 4 dogs (from both ends of the sample), each as image | mask | gray-background cutout.
        picks = list(range(4)) + list(range(len(sample) - 4, len(sample)))
        fig, axes = plt.subplots(len(picks), 3, figsize=(7.5, 2.4 * len(picks)))
        for row, idx in enumerate(picks):
            with Image.open(sample[idx]) as img:
                img = img.convert("RGB")
            mask = masks[idx]
            cut = Image.composite(img, Image.new("RGB", img.size, (128, 128, 128)), mask)
            for col, (im, title) in enumerate(((img, sample[idx].parent.name), (mask, "mask"), (cut, "cutout"))):
                axes[row, col].imshow(im, cmap="gray" if col == 1 else None)
                axes[row, col].set_title(title, fontsize=8)
                axes[row, col].axis("off")
        fig.suptitle(f"{model}  ({per_image:.2f} s/img)")
        fig.tight_layout()
        fig.savefig(out_dir / f"seg_pilot_{model}.png", dpi=70)
        plt.close(fig)
    print(f"Grids saved to {out_dir}/seg_pilot_<model>.png")


def cmd_check(args: argparse.Namespace) -> None:
    paths = all_image_paths(args.results_dir)
    missing, tiny = [], []
    hasher = hashlib.sha256()
    for p in paths:
        mp = mask_path_for(p, args.model)
        if not mp.exists():
            missing.append(p)
            continue
        data = mp.read_bytes()
        hasher.update(f"{p.parent.name}/{p.name}|".encode())
        hasher.update(hashlib.sha256(data).digest())
        with Image.open(mp) as m:
            if (np.asarray(m.convert("L")) > 127).mean() < _TINY_MASK_FRACTION:
                tiny.append(p)
    print(f"{args.model}: {len(paths) - len(missing)}/{len(paths)} masks present, {len(missing)} missing")
    for p in missing[:10]:
        print(f"  missing: {p.parent.name}/{p.name}")
    print(
        f"  tiny/empty masks (<{_TINY_MASK_FRACTION:.0%} foreground; these fall back to a center crop): "
        f"{len(tiny)} ({len(tiny) / max(len(paths) - len(missing), 1):.1%})"
    )
    print(f"  fingerprint (compare across machines): {hasher.hexdigest()[:16]}")


def cmd_preview(args: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from features import PREPROCESS_LEVELS_BY_NAME
    from preprocess import load_image_and_mask

    split = _split(args.results_dir)
    cats = [p for p, y in split.val_records if y == 0][: args.n // 2]
    dogs = [p for p, y in split.val_records if y == 1][: args.n - args.n // 2]
    levels = ["P3_square", "P5_maskcrop", "P5_maskcrop_raw", "P6_maskcrop_bggray", "P7_maskcrop_bgblur"]
    fig, axes = plt.subplots(len(cats) + len(dogs), len(levels) + 1, figsize=(2 * (len(levels) + 1), 2.1 * (len(cats) + len(dogs))))
    for row, path in enumerate(cats + dogs):
        for col, name in enumerate(levels):
            img, mask = load_image_and_mask(Path(path), PREPROCESS_LEVELS_BY_NAME[name].config)
            axes[row, col].imshow(img, cmap="gray", vmin=0, vmax=255)
            axes[row, col].set_title(name, fontsize=7)
            if name == "P5_maskcrop":
                axes[row, -1].imshow(mask, cmap="gray", vmin=0, vmax=1)
                axes[row, -1].set_title("mask (P5 crop)", fontsize=7)
        for ax in axes[row]:
            ax.axis("off")
    fig.tight_layout()
    out = args.results_dir / "debug" / "mask_levels_preview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=70)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pilot", help="Time candidate models and save visual grids.")
    p.add_argument("--models", default="u2net,isnet-general-use,birefnet-general-lite")
    p.add_argument("--n", type=int, default=200, help="Number of val images to time on (half cats, half dogs).")

    p = sub.add_parser("run", help="Generate masks for one shard of all images (resumable).")
    p.add_argument("--model", required=True)
    p.add_argument("--shard", default="0/1", help="i/N: this machine processes every N-th image starting at i.")
    p.add_argument("--limit", type=int, default=0, help="Only the first N images of the shard (for testing).")

    p = sub.add_parser("check", help="Count/validate masks and print a cross-machine fingerprint.")
    p.add_argument("--model", required=True)

    p = sub.add_parser("preview", help="Visual check of the mask-based preprocessing levels.")
    p.add_argument("--n", type=int, default=12, help="Number of val images (half cats, half dogs).")

    args = parser.parse_args()
    {"pilot": cmd_pilot, "run": cmd_run, "check": cmd_check, "preview": cmd_preview}[args.command](args)


if __name__ == "__main__":
    main()
