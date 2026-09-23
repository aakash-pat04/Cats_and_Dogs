"""CLI experiment runner: chunked, resumable preprocessing x feature x
classifier x dataset-size sweep for the Cats vs Dogs homework.

Subcommands (each independently invocable — separate terminal, separate
background process, separate overnight run):

  extract-features   Compute/cache HOG+LBP features for one or more
                      preprocessing levels and split parts.
  sweep               Train+validate classifiers across a slice of the
                      (preprocessing x feature x size x classifier) grid,
                      appending results to its own chunk CSV. Safe to
                      re-run the same command to resume after an
                      interruption (already-completed combos are skipped).
  merge               Combine all chunk CSVs into results/sweep_results.csv.
  final-eval          Pick the best validated combo and evaluate it once
                      on the held-out test set (accuracy, confusion
                      matrix, classification report).

See the recommended chunk breakdown in the project plan for example
invocations, including deferring RBF-SVM on the largest sizes to a later,
independent chunk.
"""

from __future__ import annotations

import argparse
import csv
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from dataset_split import SplitConfig, TrainSizeSpec, build_or_load_split, training_subset
from features import (
    FEATURE_LEVELS,
    FeatureConfig,
    PREPROCESS_LEVELS,
    PREPROCESS_LEVELS_BY_NAME,
    build_feature_vector,
    compute_or_load_raw_features,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAT_DIR = _PROJECT_ROOT / "data" / "PetImages" / "Cat"
DEFAULT_DOG_DIR = _PROJECT_ROOT / "data" / "PetImages" / "Dog"

# The unconstrained (default-hyperparameter) originals, plus capacity-limited
# variants added after the P0_minimal sweep showed near-100% train accuracy
# at every size for all three — see experiment_log.md for the full story.
# Adding these as NEW classifier names (not replacing the originals) is
# purely additive to sweep_results.csv/chunk CSVs: no existing row's schema
# changes, so no rerun of already-completed combos is triggered.
CLASSIFIER_BUILDERS = {
    "decision_tree": lambda: DecisionTreeClassifier(random_state=42),
    "random_forest": lambda: RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    "svm_rbf": lambda: SVC(kernel="rbf", C=10, random_state=42),
    "decision_tree_capped": lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42),
    "random_forest_capped": lambda: RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    "svm_rbf_low_c": lambda: SVC(kernel="rbf", C=0.1, random_state=42),
}
# Any classifier whose training cost scales the way RBF-SVM's does (roughly
# quadratic-to-cubic in sample count) should respect --svm-max-size, not
# just the literal "svm_rbf" name.
_SVM_LIKE_PREFIX = "svm_rbf"

RESULT_FIELDS = [
    "preprocess_level",
    "feature_level",
    "classifier",
    "train_size_key",
    "n_train_per_class",
    "n_train_total",
    "feature_dim",
    "train_accuracy",
    "train_precision_macro",
    "train_recall_macro",
    "train_f1_macro",
    "val_accuracy",
    "val_precision_macro",
    "val_recall_macro",
    "val_f1_macro",
    "fit_time_sec",
    "train_predict_time_sec",
    "val_predict_time_sec",
    "seed",
    "timestamp",
]
_KEY_FIELDS = ("preprocess_level", "feature_level", "classifier", "train_size_key", "seed")


def parse_train_sizes(spec: str) -> Tuple[TrainSizeSpec, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    return tuple(p if p == "max" else int(p) for p in parts)


def _split_config_from_args(args: argparse.Namespace) -> SplitConfig:
    return SplitConfig(
        cat_dir=args.cat_dir,
        dog_dir=args.dog_dir,
        seed=args.seed,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        train_sizes=parse_train_sizes(args.train_sizes),
    )


def _load_split(args: argparse.Namespace):
    cache_path = Path(args.results_dir) / "split_cache.json"
    return build_or_load_split(_split_config_from_args(args), cache_path=cache_path)


def _resolve_preprocess_levels(name: str):
    return PREPROCESS_LEVELS if name == "all" else [PREPROCESS_LEVELS_BY_NAME[name]]


def _resolve_feature_levels(name: str) -> List[str]:
    return list(FEATURE_LEVELS) if name == "all" else [name]


def run_single_combo(
    preprocess_level: str,
    feature_level: str,
    clf_name: str,
    train_size_key: str,
    n_per_class: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> Dict[str, object]:
    """Fit one classifier on one (preprocessing, feature, size) slice and score it on train + validation.

    Scoring on the training slice itself (not just validation) is what lets
    the train/val gap be plotted directly as an overfitting signal across
    the size sweep — see `reporting.plot_train_val_gap`.
    """
    import time

    clf = CLASSIFIER_BUILDERS[clf_name]()

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_train_pred = clf.predict(X_train)
    train_predict_time = time.perf_counter() - t0
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        y_train, y_train_pred, average="macro", zero_division=0
    )

    t0 = time.perf_counter()
    y_val_pred = clf.predict(X_val)
    val_predict_time = time.perf_counter() - t0
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="macro", zero_division=0
    )

    return {
        "preprocess_level": preprocess_level,
        "feature_level": feature_level,
        "classifier": clf_name,
        "train_size_key": train_size_key,
        "n_train_per_class": n_per_class,
        "n_train_total": 2 * n_per_class,
        "feature_dim": X_train.shape[1],
        "train_accuracy": train_accuracy,
        "train_precision_macro": train_precision,
        "train_recall_macro": train_recall,
        "train_f1_macro": train_f1,
        "val_accuracy": val_accuracy,
        "val_precision_macro": val_precision,
        "val_recall_macro": val_recall,
        "val_f1_macro": val_f1,
        "fit_time_sec": fit_time,
        "train_predict_time_sec": train_predict_time,
        "val_predict_time_sec": val_predict_time,
        "seed": seed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def _chunk_header(csv_path: Path) -> List[str]:
    with open(csv_path, newline="") as f:
        return next(csv.reader(f), [])


def _quarantine_stale_chunk(csv_path: Path) -> None:
    """Move a chunk file with an outdated column schema out of the way.

    Triggered whenever RESULT_FIELDS changes (e.g. new metrics added) and an
    old chunk file predates that change: appending new-schema rows to an
    old-schema file would misalign columns, and treating its rows as
    "already completed" would silently skip recomputing them with the new
    fields. Moving it into a `_stale/` subdirectory (excluded from the
    top-level `*.csv` glob used everywhere else) makes every affected combo
    look not-yet-run, so the next `sweep` naturally regenerates it.
    """
    stale_dir = csv_path.parent / "_stale"
    stale_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = stale_dir / f"{csv_path.stem}.{timestamp}{csv_path.suffix}"
    csv_path.rename(dest)
    logger.warning(
        "%s used an outdated results schema (columns changed); moved it to %s. "
        "Its combos will be recomputed with the current schema.",
        csv_path,
        dest,
    )


def _load_all_chunk_keys(chunks_dir: Path) -> set:
    """Union of (preprocess_level, feature_level, classifier, train_size_key, seed) already recorded anywhere.

    Chunk files whose columns don't match the current RESULT_FIELDS are
    ignored here (their combos are treated as not-yet-run) rather than
    trusted at face value — see `_quarantine_stale_chunk`.
    """
    keys = set()
    if not chunks_dir.exists():
        return keys
    for csv_path in chunks_dir.glob("*.csv"):
        if _chunk_header(csv_path) != RESULT_FIELDS:
            logger.warning(
                "Ignoring %s for resume purposes: its columns don't match the current results "
                "schema. Re-run `sweep` for the levels/features it covers to regenerate it.",
                csv_path,
            )
            continue
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                keys.add(tuple(row[field] for field in _KEY_FIELDS))
    return keys


def _chunk_filename(preprocess_levels: List[str], feature_levels: List[str], classifiers: List[str], sizes: List[str]) -> str:
    slug = lambda items: "+".join(items)
    name = f"{slug(preprocess_levels)}__{slug(feature_levels)}__{slug(classifiers)}__{slug(sizes)}.csv"
    if len(name) > 200:
        import hashlib

        name = hashlib.sha256(name.encode()).hexdigest() + ".csv"
    return name


def run_extract_features(args: argparse.Namespace) -> None:
    split = _load_split(args)
    levels = _resolve_preprocess_levels(args.preprocess_level)
    parts = ["train_pool", "val", "test"] if args.split_part == "all" else [args.split_part]
    fc = FeatureConfig()
    pool_max = min(len(split.cat_train_pool), len(split.dog_train_pool))

    for level in levels:
        for part in parts:
            if part == "train_pool":
                records = training_subset(split, pool_max)
            elif part == "val":
                records = split.val_records
            elif part == "test":
                records = split.test_records
            else:
                raise ValueError(f"Unknown split_part {part!r}")
            compute_or_load_raw_features(
                records, level, fc, part, Path(args.cache_dir), args.force_recompute_features
            )


def run_sweep_chunk(args: argparse.Namespace) -> None:
    split = _load_split(args)
    levels = _resolve_preprocess_levels(args.preprocess_level)
    feature_levels = _resolve_feature_levels(args.feature_level)
    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    for c in classifiers:
        if c not in CLASSIFIER_BUILDERS:
            raise ValueError(f"Unknown classifier {c!r}; choices: {sorted(CLASSIFIER_BUILDERS)}")

    fc = FeatureConfig()
    cache_dir = Path(args.cache_dir)
    chunks_dir = Path(args.results_dir) / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    existing_keys = _load_all_chunk_keys(chunks_dir)

    pool_max = min(len(split.cat_train_pool), len(split.dog_train_pool))
    chunk_name = _chunk_filename(
        [lvl.name for lvl in levels], feature_levels, classifiers, [str(s) for s in split.train_sizes_resolved]
    )
    chunk_path = chunks_dir / chunk_name
    if chunk_path.exists() and _chunk_header(chunk_path) != RESULT_FIELDS:
        _quarantine_stale_chunk(chunk_path)
    write_header = not chunk_path.exists()

    with open(chunk_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if write_header:
            writer.writeheader()
            f.flush()

        for level in levels:
            full_pool_records = training_subset(split, pool_max)
            train_hog, train_lbp, train_labels, _ = compute_or_load_raw_features(
                full_pool_records, level, fc, "train_pool", cache_dir, args.force_recompute_features
            )
            val_hog, val_lbp, val_labels, _ = compute_or_load_raw_features(
                split.val_records, level, fc, "val", cache_dir, args.force_recompute_features
            )

            for feature_level in feature_levels:
                X_train_full = build_feature_vector(train_hog, train_lbp, feature_level)
                X_val = build_feature_vector(val_hog, val_lbp, feature_level)

                for size_key, n in split.train_sizes_resolved.items():
                    cat_rows = X_train_full[:n]
                    dog_rows = X_train_full[pool_max : pool_max + n]
                    X_train = np.concatenate([cat_rows, dog_rows], axis=0)
                    y_train = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])

                    scaler = StandardScaler().fit(X_train)
                    X_train_s = scaler.transform(X_train)
                    X_val_s = scaler.transform(X_val)

                    for clf_name in classifiers:
                        if clf_name.startswith(_SVM_LIKE_PREFIX) and n > args.svm_max_size:
                            logger.info(
                                "Skipping %s at size %s (%d/class > --svm-max-size %d)",
                                clf_name,
                                size_key,
                                n,
                                args.svm_max_size,
                            )
                            continue

                        key = (level.name, feature_level, clf_name, size_key, str(args.seed))
                        if key in existing_keys:
                            logger.info("Skipping already-completed combo %s", key)
                            continue

                        row = run_single_combo(
                            level.name,
                            feature_level,
                            clf_name,
                            size_key,
                            n,
                            X_train_s,
                            y_train,
                            X_val_s,
                            val_labels,
                            args.seed,
                        )
                        writer.writerow(row)
                        f.flush()
                        existing_keys.add(key)
                        logger.info(
                            "%s/%s/%s/size=%s: train_acc=%.4f val_acc=%.4f (gap=%.4f, fit %.1fs)",
                            level.name,
                            feature_level,
                            clf_name,
                            size_key,
                            row["train_accuracy"],
                            row["val_accuracy"],
                            row["train_accuracy"] - row["val_accuracy"],
                            row["fit_time_sec"],
                        )


def merge_chunks(results_dir: Path) -> pd.DataFrame:
    chunks_dir = results_dir / "chunks"
    all_chunk_paths = sorted(chunks_dir.glob("*.csv"))

    frames = []
    skipped = []
    for p in all_chunk_paths:
        if _chunk_header(p) != RESULT_FIELDS:
            skipped.append(p)
            continue
        frames.append(pd.read_csv(p))
    if skipped:
        # Concatenating a chunk with different columns would silently
        # introduce NaN for whatever's missing (e.g. an old chunk from
        # before a metric was added) — excluded outright instead, with a
        # pointer to the fix, rather than merging it in unnoticed.
        logger.warning(
            "Skipped %d chunk file(s) with an outdated results schema during merge (not included "
            "in sweep_results.csv): %s. Re-run `sweep` for the levels/features they cover to "
            "regenerate them with the current fields.",
            len(skipped),
            [str(p) for p in skipped],
        )
    if not frames:
        raise RuntimeError(
            f"No chunk CSVs with the current results schema found in {chunks_dir}; run `sweep` at least once first."
        )

    combined = pd.concat(frames, ignore_index=True)
    key_cols = list(_KEY_FIELDS)
    combined = combined.sort_values("timestamp").drop_duplicates(subset=key_cols, keep="last")
    combined = combined.sort_values(
        ["preprocess_level", "feature_level", "classifier", "n_train_per_class"]
    ).reset_index(drop=True)

    out_path = results_dir / "sweep_results.csv"
    combined.to_csv(out_path, index=False)
    logger.info("Merged %d chunk file(s) into %s (%d rows)", len(frames), out_path, len(combined))
    return combined


def run_final_eval(args: argparse.Namespace) -> None:
    import reporting

    results_path = Path(args.results_dir) / "sweep_results.csv"
    if not results_path.exists():
        raise RuntimeError(f"{results_path} not found — run `merge` after at least one `sweep` chunk first.")
    results_df = pd.read_csv(results_path)

    split = _load_split(args)
    fc = FeatureConfig()
    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)

    best_row = reporting.select_best_combo(results_df)
    logger.info("Best validated combo:\n%s", best_row)

    report = reporting.evaluate_final_on_test(best_row, split, fc, cache_dir)
    reporting.save_final_report(report, results_dir / "final_report.json")
    reporting.plot_confusion_matrix(np.array(report["confusion_matrix"]), results_dir / "final_confusion_matrix.png")
    reporting.plot_learning_curves(
        results_df,
        results_dir / "learning_curve.png",
        fixed_preprocess=best_row["preprocess_level"],
        fixed_feature=best_row["feature_level"],
    )
    reporting.plot_train_val_gap(
        results_df,
        results_dir / "train_val_gap.png",
        fixed_preprocess=best_row["preprocess_level"],
        fixed_feature=best_row["feature_level"],
    )
    logger.info("Final test accuracy: %.4f", report["accuracy"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cat-dir", type=Path, default=DEFAULT_CAT_DIR)
    parser.add_argument("--dog-dir", type=Path, default=DEFAULT_DOG_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-per-class", type=int, default=500)
    parser.add_argument("--test-per-class", type=int, default=500)
    parser.add_argument("--train-sizes", type=str, default="80,200,500,1000,5000,10000,max")
    parser.add_argument("--cache-dir", type=Path, default=Path("results/cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--force-recompute-features", action="store_true")

    subparsers = parser.add_subparsers(dest="command", required=True)

    level_choices = [*PREPROCESS_LEVELS_BY_NAME, "all"]

    p_extract = subparsers.add_parser("extract-features", help="Compute/cache HOG+LBP features.")
    p_extract.add_argument("--preprocess-level", choices=level_choices, default="all")
    p_extract.add_argument("--split-part", choices=["train_pool", "val", "test", "all"], default="all")

    p_sweep = subparsers.add_parser("sweep", help="Train+validate classifiers; resumable, writes its own chunk CSV.")
    p_sweep.add_argument("--preprocess-level", choices=level_choices, default="all")
    p_sweep.add_argument("--feature-level", choices=[*FEATURE_LEVELS, "all"], default="all")
    p_sweep.add_argument(
        "--classifiers",
        type=str,
        default="decision_tree,random_forest,svm_rbf",
        help=(
            "Comma-separated list. Available: "
            + ",".join(CLASSIFIER_BUILDERS)
            + ". The *_capped/*_low_c variants are capacity-limited counterparts of the "
            "originals, added to compare against the unconstrained versions' overfitting "
            "(see experiment_log.md)."
        ),
    )
    p_sweep.add_argument("--svm-max-size", type=int, default=1000)

    subparsers.add_parser("merge", help="Combine chunk CSVs into results/sweep_results.csv.")
    subparsers.add_parser("final-eval", help="Evaluate the best validated combo once on the test set.")

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_arg_parser().parse_args()

    if args.command == "extract-features":
        run_extract_features(args)
    elif args.command == "sweep":
        run_sweep_chunk(args)
    elif args.command == "merge":
        merge_chunks(Path(args.results_dir))
    elif args.command == "final-eval":
        run_final_eval(args)
    else:
        raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    main()
