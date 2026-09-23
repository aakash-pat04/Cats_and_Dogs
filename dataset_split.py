"""Deterministic, fixed train/val/test split for the Cats vs Dogs dataset.

Builds ONE fixed, seeded validation set and ONE fixed, seeded test set per
class, held out once and reused across every training-size experiment.
Everything else forms a per-class "training pool" that is itself seeded and
shuffled once, so that a training subset of size N is always a strict prefix
of the training subset of size M for any N < M ("nested" subsets). This lets
an experiment runner extract features once over the full pool and slice rows
per size step, instead of recomputing anything per step.

Corrupt or truncated files are filtered out up front via `validate_and_filter`
so they can never end up in a val/test set (which must stay reliable across
every experiment) or silently pollute a training subset with garbage pixels.
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from collections import OrderedDict

from preprocess import PathLike, PreprocessConfig, list_image_paths, preprocess_paths, _DEFAULT_EXTENSIONS

logger = logging.getLogger(__name__)

TrainSizeSpec = Union[int, str]
Record = Tuple[Path, int]


@dataclass(frozen=True)
class SplitConfig:
    """Knobs for building the fixed train-pool/val/test split.

    Attributes:
        cat_dir: Directory of cat images.
        dog_dir: Directory of dog images.
        seed: Random seed controlling the per-class shuffle. Fixed for
            reproducibility; the same seed always yields the same split.
        val_per_class: Number of images per class held out for validation.
        test_per_class: Number of images per class held out for testing.
        train_sizes: Training-set sizes per class to support, e.g.
            (80, 200, 500, 1000, 5000, 10000, "max"). "max" resolves to
            every remaining image in the smaller class's training pool.
        valid_extensions: Image file suffixes to consider.
    """

    cat_dir: PathLike
    dog_dir: PathLike
    seed: int = 42
    val_per_class: int = 500
    test_per_class: int = 500
    train_sizes: Tuple[TrainSizeSpec, ...] = (80, 200, 500, 1000, 5000, 10000, "max")
    valid_extensions: Tuple[str, ...] = field(default=_DEFAULT_EXTENSIONS)


@dataclass(frozen=True)
class DatasetSplit:
    """The resolved, fixed split: held-out val/test records plus nested train pools."""

    val_records: List[Record]
    test_records: List[Record]
    cat_train_pool: List[Path]
    dog_train_pool: List[Path]
    train_sizes_resolved: Dict[str, int]
    seed: int


def validate_and_filter(paths: List[Path]) -> List[Path]:
    """Filter out files that fail to decode cleanly.

    Runs a cheap decode (`resize=(1,1)`) through the normal preprocessing
    path with Pillow warnings promoted to errors, so a file that Pillow
    would otherwise "load" with a silent `UserWarning` (e.g. a truncated
    JPEG that decodes to garbage instead of raising) is excluded rather than
    silently kept. Decode success does not depend on resize/grayscale/
    denoise/equalize choices, so this only needs to run once regardless of
    how many PreprocessConfig variants are swept downstream.
    """
    probe_config = PreprocessConfig(resize=(1, 1))
    good: List[Path] = []
    for path in paths:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                images, _, _ = preprocess_paths([(path, 0)], probe_config)
            except Exception as exc:  # noqa: BLE001 - any decode failure/warning excludes the file
                logger.warning("Excluding undecodable file %s: %s", path, exc)
                continue
        if images.shape[0] == 1:
            good.append(path)
    logger.info("Validated %d/%d file(s)", len(good), len(paths))
    return good


def _resolve_train_sizes(
    train_sizes: Tuple[TrainSizeSpec, ...], max_size: int
) -> Dict[str, int]:
    resolved: Dict[str, int] = OrderedDict()
    seen_max = False
    for spec in train_sizes:
        if isinstance(spec, str):
            if spec != "max":
                raise ValueError(f"Unsupported train_sizes string {spec!r}; only 'max' is allowed")
            if seen_max:
                raise ValueError("'max' may only appear once in train_sizes")
            seen_max = True
            key, n = "max", max_size
        else:
            n = int(spec)
            if n <= 0:
                raise ValueError(f"train_sizes entries must be positive, got {spec}")
            key = str(n)
        if n > max_size:
            raise ValueError(
                f"Requested train size {key} ({n}/class) exceeds the available "
                f"per-class training pool ({max_size}/class after val/test are held out)"
            )
        if key in resolved:
            raise ValueError(f"Duplicate train_sizes entry: {key}")
        resolved[key] = n
    return resolved


def build_split(config: SplitConfig) -> DatasetSplit:
    """Build the fixed val/test sets and the nested, seeded training pools.

    Deterministic given `config.seed`: draws one permutation for the cat
    pool then one for the dog pool from a single `np.random.default_rng`
    stream, in that fixed order, so re-running with the same seed always
    reproduces the same split.
    """
    cat_paths = validate_and_filter(list_image_paths(config.cat_dir, config.valid_extensions))
    dog_paths = validate_and_filter(list_image_paths(config.dog_dir, config.valid_extensions))

    rng = np.random.default_rng(config.seed)
    cat_shuffled = [cat_paths[i] for i in rng.permutation(len(cat_paths))]
    dog_shuffled = [dog_paths[i] for i in rng.permutation(len(dog_paths))]

    held_out = config.val_per_class + config.test_per_class
    for name, shuffled in (("cat", cat_shuffled), ("dog", dog_shuffled)):
        if held_out >= len(shuffled):
            raise ValueError(
                f"val_per_class + test_per_class ({held_out}) leaves no training "
                f"images for the {name} class (only {len(shuffled)} valid files available)"
            )

    val_end = config.val_per_class
    test_end = val_end + config.test_per_class

    val_records: List[Record] = [(p, 0) for p in cat_shuffled[:val_end]] + [
        (p, 1) for p in dog_shuffled[:val_end]
    ]
    test_records: List[Record] = [(p, 0) for p in cat_shuffled[val_end:test_end]] + [
        (p, 1) for p in dog_shuffled[val_end:test_end]
    ]
    cat_train_pool = cat_shuffled[test_end:]
    dog_train_pool = dog_shuffled[test_end:]

    max_size = min(len(cat_train_pool), len(dog_train_pool))
    resolved_sizes = _resolve_train_sizes(config.train_sizes, max_size)
    logger.info(
        "Split built: val=%d/class, test=%d/class, train pool=%d cat / %d dog "
        "(max usable=%d/class); sizes=%s",
        config.val_per_class,
        config.test_per_class,
        len(cat_train_pool),
        len(dog_train_pool),
        max_size,
        dict(resolved_sizes),
    )

    return DatasetSplit(
        val_records=val_records,
        test_records=test_records,
        cat_train_pool=cat_train_pool,
        dog_train_pool=dog_train_pool,
        train_sizes_resolved=resolved_sizes,
        seed=config.seed,
    )


def _pool_signature(config: SplitConfig) -> Dict[str, object]:
    """The subset of SplitConfig that determines val/test/pool membership.

    Deliberately excludes `train_sizes`: which sizes you plan to sweep
    doesn't change who's in val/test/the training pool, only how far into
    the pool a given step reads. This lets a cached split stay valid across
    chunk invocations that request different --train-sizes.
    """
    return {
        "cat_dir": str(config.cat_dir),
        "dog_dir": str(config.dog_dir),
        "seed": config.seed,
        "val_per_class": config.val_per_class,
        "test_per_class": config.test_per_class,
        "valid_extensions": list(config.valid_extensions),
    }


def save_split(split: DatasetSplit, path: Path) -> None:
    """Persist the resolved val/test/pool path lists (not resolved sizes) to JSON."""
    data = {
        "seed": split.seed,
        "val_records": [[str(p), label] for p, label in split.val_records],
        "test_records": [[str(p), label] for p, label in split.test_records],
        "cat_train_pool": [str(p) for p in split.cat_train_pool],
        "dog_train_pool": [str(p) for p in split.dog_train_pool],
    }
    path.write_text(json.dumps(data))


def _load_split_pools(path: Path) -> Tuple[List[Record], List[Record], List[Path], List[Path], int]:
    data = json.loads(path.read_text())
    val_records = [(Path(p), label) for p, label in data["val_records"]]
    test_records = [(Path(p), label) for p, label in data["test_records"]]
    cat_train_pool = [Path(p) for p in data["cat_train_pool"]]
    dog_train_pool = [Path(p) for p in data["dog_train_pool"]]
    return val_records, test_records, cat_train_pool, dog_train_pool, data["seed"]


def build_or_load_split(config: SplitConfig, cache_path: Optional[Path] = None) -> DatasetSplit:
    """Like `build_split`, but reuses a cached val/test/pool assignment when available.

    The expensive part of `build_split` is the one-time decode validation
    over the full dataset (tens of seconds across ~25k files); this lets
    every chunk invocation in a multi-session sweep skip repeating it, as
    long as the underlying directories/seed/val/test sizes haven't changed
    (see `_pool_signature`). `train_sizes` is always re-resolved fresh
    against the cached pool, so different chunks may safely request
    different `--train-sizes` without invalidating the cache.
    """
    if cache_path is not None and cache_path.exists():
        sig_path = cache_path.with_suffix(".config.json")
        if sig_path.exists() and json.loads(sig_path.read_text()) == _pool_signature(config):
            logger.info("Loaded cached split from %s", cache_path)
            val_records, test_records, cat_train_pool, dog_train_pool, seed = _load_split_pools(cache_path)
            max_size = min(len(cat_train_pool), len(dog_train_pool))
            return DatasetSplit(
                val_records=val_records,
                test_records=test_records,
                cat_train_pool=cat_train_pool,
                dog_train_pool=dog_train_pool,
                train_sizes_resolved=_resolve_train_sizes(config.train_sizes, max_size),
                seed=seed,
            )
        logger.info("Split cache at %s is stale or config changed; rebuilding", cache_path)

    split = build_split(config)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        save_split(split, cache_path)
        cache_path.with_suffix(".config.json").write_text(json.dumps(_pool_signature(config)))
    return split


def training_subset(split: DatasetSplit, size: int) -> List[Record]:
    """First `size` cats + first `size` dogs from the fixed, shuffled training pools.

    Guaranteed to be a strict prefix of `training_subset(split, size2)` for
    any `size2 > size`, since `cat_train_pool`/`dog_train_pool` never change.
    """
    if size > len(split.cat_train_pool) or size > len(split.dog_train_pool):
        raise ValueError(
            f"Requested size {size} exceeds available pool "
            f"({len(split.cat_train_pool)} cat / {len(split.dog_train_pool)} dog)"
        )
    return [(p, 0) for p in split.cat_train_pool[:size]] + [(p, 1) for p in split.dog_train_pool[:size]]
