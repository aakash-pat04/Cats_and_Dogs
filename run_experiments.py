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
  pool                Evidence pooling: evaluate "juries" of saved models
                      (majority / soft / weighted vote) on validation.
  blur-report         Diagnostic: how blurry are the images? (Laplacian
                      variance histogram + the blurriest examples.)
  final-eval          Evaluate the best validated combo (or, with --jury,
                      the chosen jury) once on the held-out test set
                      (accuracy, confusion matrix, classification report).

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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

from dataset_split import DatasetSplit, SplitConfig, TrainSizeSpec, build_or_load_split, training_subset
from features import (
    DEFAULT_FEATURE_LEVELS,
    DEFAULT_PREPROCESS_LEVELS,
    FEATURE_LEVELS,
    Blocks,
    PreprocessLevel,
    PREPROCESS_LEVELS_BY_NAME,
    blocks_for,
    build_feature_vector,
    load_feature_blocks,
    training_matrix,
)
from pooling import RULES, decision_scores, records_signature, save_scores, score_path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CAT_DIR = _PROJECT_ROOT / "data" / "PetImages" / "Cat"
DEFAULT_DOG_DIR = _PROJECT_ROOT / "data" / "PetImages" / "Dog"

# libsvm's kernel cache, in MB. The default (200) is far too small once the
# training set reaches thousands of rows, forcing kernel values to be
# recomputed over and over; this only affects speed, never results.
_SVC_CACHE_MB = 2000


def _hellinger(X: np.ndarray) -> np.ndarray:
    """Elementwise sqrt ("Hellinger map"): the standard fix for comparing histogram features.

    HOG, LBP and color features are all (normalized) histograms. Euclidean
    distance on raw histograms over-weights large bins; on their square roots
    it approximates the Hellinger/Bhattacharyya distance, which usually
    suits histograms much better.
    """
    return np.sqrt(np.maximum(X, 0.0))


class CappedPCA(TransformerMixin, BaseEstimator):
    """Randomized PCA to `n_components`, capped at what the training slice supports.

    PCA can't keep more components than there are training rows, and the
    smallest sweep sizes (80/class) have fewer rows than 512. Above that the
    cap is inactive and this is exactly `PCA(n_components, svd_solver="randomized")`
    (including using its `fit_transform`), so results are unchanged.
    """

    def __init__(self, n_components: int = 512, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state

    def _make(self, X: np.ndarray) -> PCA:
        k = min(self.n_components, X.shape[0], X.shape[1])
        return PCA(n_components=k, svd_solver="randomized", random_state=self.random_state)

    def fit(self, X, y=None):
        self.pca_ = self._make(X).fit(X)
        return self

    def fit_transform(self, X, y=None):
        self.pca_ = self._make(X)
        return self.pca_.fit_transform(X)

    def transform(self, X):
        return self.pca_.transform(X)


def _scaled(model) -> Pipeline:
    """`model` behind a StandardScaler that is fit on the training slice only (the homework's rule).

    Putting the scaler inside the pipeline means it's refit on every single
    training-size slice — exactly as the sweep always did explicitly — and
    that `final-eval` can never scale differently from the sweep.
    """
    return make_pipeline(StandardScaler(), model)


def _hellinger_scaled(model) -> Pipeline:
    return make_pipeline(FunctionTransformer(_hellinger), StandardScaler(), model)


# Each builder takes the feature dimension (some models' defaults depend on it).
#
# The unconstrained (default-hyperparameter) originals, plus capacity-limited
# variants added after the P0_minimal sweep showed near-100% train accuracy
# at every size for all three — see experiment_log.md for the full story.
# Adding these as NEW classifier names (not replacing the originals) is
# purely additive to sweep_results.csv/chunk CSVs: no existing row's schema
# changes, so no rerun of already-completed combos is triggered.
CLASSIFIER_BUILDERS: Dict[str, Callable[[int], object]] = {
    "decision_tree": lambda d: _scaled(DecisionTreeClassifier(random_state=42)),
    "random_forest": lambda d: _scaled(RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    "svm_rbf": lambda d: _scaled(SVC(kernel="rbf", C=10, random_state=42, cache_size=_SVC_CACHE_MB)),
    "decision_tree_capped": lambda d: _scaled(
        DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)
    ),
    "random_forest_capped": lambda d: _scaled(
        RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1)
    ),
    "svm_rbf_low_c": lambda d: _scaled(SVC(kernel="rbf", C=0.1, random_state=42, cache_size=_SVC_CACHE_MB)),
    # Fills in the C grid between svm_rbf_low_c (C=0.1) and svm_rbf (C=10):
    # svm_rbf_low_c underperformed at every size tested and got worse with
    # more data, suggesting C=0.1 overshot into underfitting rather than
    # just removing memorization — these narrow the search toward wherever
    # the real optimum sits. See experiment_log.md.
    "svm_rbf_c1": lambda d: _scaled(SVC(kernel="rbf", C=1, random_state=42, cache_size=_SVC_CACHE_MB)),
    "svm_rbf_c3": lambda d: _scaled(SVC(kernel="rbf", C=3, random_state=42, cache_size=_SVC_CACHE_MB)),
    # --- Round 2: faster / different model families (see experiment_log.md). ---
    # Linear SVM: seconds even at max size + flips, so it makes large feature
    # stacks affordable. Expected a few points below RBF on raw HOG.
    "linsvc_c1e-3": lambda d: _scaled(LinearSVC(C=1e-3, random_state=42, max_iter=5000)),
    "linsvc_c1e-2": lambda d: _scaled(LinearSVC(C=1e-2, random_state=42, max_iter=5000)),
    "hellinger_linsvc": lambda d: _hellinger_scaled(LinearSVC(C=1e-2, random_state=42, max_iter=5000)),
    # Approximate RBF kernel (Nystroem) + linear SVM: close to the exact RBF
    # SVM at near-linear cost. gamma=1/d is sklearn's gamma="scale" for
    # standardized features (overall variance ~1).
    "nystroem_svm_c1": lambda d: _scaled(
        Pipeline([
            ("nystroem", Nystroem(kernel="rbf", gamma=1.0 / d, n_components=3000, random_state=42)),
            ("linsvc", LinearSVC(C=1, random_state=42, max_iter=5000)),
        ])
    ),
    # Exact RBF SVM (the current best family) on PCA-reduced features:
    # kernel evaluations cost O(dims), so 8,100 -> 512 dims is ~16x less
    # work per kernel value.
    "pca_svm_c3": lambda d: _scaled(
        Pipeline([
            ("pca", CappedPCA(n_components=512, random_state=42)),
            ("svc", SVC(kernel="rbf", C=3, random_state=42, cache_size=_SVC_CACHE_MB)),
        ])
    ),
    # Tuning neighbors of pca_svm_c3, added once it became the best model
    # (round 2): more retained dimensions, and a stronger C.
    "pca1024_svm_c3": lambda d: _scaled(
        Pipeline([
            ("pca", CappedPCA(n_components=1024, random_state=42)),
            ("svc", SVC(kernel="rbf", C=3, random_state=42, cache_size=_SVC_CACHE_MB)),
        ])
    ),
    "pca_svm_c10": lambda d: _scaled(
        Pipeline([
            ("pca", CappedPCA(n_components=512, random_state=42)),
            ("svc", SVC(kernel="rbf", C=10, random_state=42, cache_size=_SVC_CACHE_MB)),
        ])
    ),
    "hellinger_svm_c3": lambda d: _hellinger_scaled(
        SVC(kernel="rbf", C=3, random_state=42, cache_size=_SVC_CACHE_MB)
    ),
    # Gradient-boosted trees (multithreaded). Early stopping uses an internal
    # split of the *training* slice, never the validation set.
    "hgb": lambda d: _scaled(
        HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.1, early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=20, random_state=42,
        )
    ),
}

try:
    from lightgbm import LGBMClassifier

    CLASSIFIER_BUILDERS["lightgbm"] = lambda d: _scaled(
        LGBMClassifier(
            n_estimators=500, learning_rate=0.1, num_leaves=63, colsample_bytree=0.3,
            subsample=0.8, subsample_freq=1, random_state=42, n_jobs=-1, verbose=-1,
        )
    )
except ImportError:  # optional dependency
    pass

try:
    from xgboost import XGBClassifier

    CLASSIFIER_BUILDERS["xgboost"] = lambda d: _scaled(
        XGBClassifier(
            n_estimators=500, learning_rate=0.1, max_depth=6, tree_method="hist",
            colsample_bytree=0.3, subsample=0.8, random_state=42, n_jobs=-1,
        )
    )
except ImportError:  # optional dependency
    pass

# Exact kernel SVMs, whose training cost scales roughly quadratically in
# sample count, respect --svm-max-size. (Linear/Nystroem SVMs scale
# linearly and aren't gated.)
_KERNEL_SVM_PREFIXES = ("svm_rbf", "pca_svm", "pca1024_svm", "hellinger_svm")


def is_kernel_svm(clf_name: str) -> bool:
    return clf_name.startswith(_KERNEL_SVM_PREFIXES)


def build_classifier(clf_name: str, n_features: int):
    return CLASSIFIER_BUILDERS[clf_name](n_features)


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


def _resolve_preprocess_levels(spec: str) -> List[PreprocessLevel]:
    """Comma-separated level names; "all" = the original three (see DEFAULT_PREPROCESS_LEVELS)."""
    names = DEFAULT_PREPROCESS_LEVELS if spec == "all" else [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in PREPROCESS_LEVELS_BY_NAME]
    if unknown:
        raise ValueError(f"Unknown preprocess level(s) {unknown}; choices: {sorted(PREPROCESS_LEVELS_BY_NAME)}")
    return [PREPROCESS_LEVELS_BY_NAME[n] for n in names]


def _resolve_feature_levels(spec: str) -> List[str]:
    """Comma-separated feature level names; "all" = the original two (see DEFAULT_FEATURE_LEVELS)."""
    names = list(DEFAULT_FEATURE_LEVELS) if spec == "all" else [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in FEATURE_LEVELS]
    if unknown:
        raise ValueError(f"Unknown feature level(s) {unknown}; choices: {sorted(FEATURE_LEVELS)}")
    return names


def load_train_pool(
    split: DatasetSplit,
    level: PreprocessLevel,
    cache_dir: Path,
    block_names: List[str],
    force: bool = False,
) -> Tuple[Blocks, Optional[Blocks], int]:
    """Cached feature blocks for the full training pool (plus its mirrored copy if the level flips).

    Returns `(blocks, flip_blocks_or_None, pool_max)`; slice per size step with
    `features.training_matrix`.
    """
    pool_max = min(len(split.cat_train_pool), len(split.dog_train_pool))
    records = training_subset(split, pool_max)
    blocks, _, _ = load_feature_blocks(records, level, level.fc, "train_pool", cache_dir, block_names, force)
    flip_blocks = None
    if level.train_flip:
        flip_blocks, _, _ = load_feature_blocks(
            records, level, level.fc, "train_pool", cache_dir, block_names, force, flip=True
        )
    return blocks, flip_blocks, pool_max


def load_eval_features(
    records, level: PreprocessLevel, part: str, cache_dir: Path, block_names: List[str], force: bool = False
) -> Tuple[Blocks, Blocks, np.ndarray]:
    """Feature blocks for an evaluation set (val or test), plus its mirrored copy for flip TTA.

    The mirrored copy is only ever used at prediction time (averaging each
    image's score with its mirror's) — never for training.
    """
    blocks, labels, _ = load_feature_blocks(records, level, level.fc, part, cache_dir, block_names, force)
    flip_blocks, _, _ = load_feature_blocks(records, level, level.fc, part, cache_dir, block_names, force, flip=True)
    return blocks, flip_blocks, labels


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
    train_acc_sample: int = 0,
    X_val_flip: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, object], Dict[str, np.ndarray]]:
    """Fit one classifier on one (preprocessing, feature, size) slice and score it on train + validation.

    Scoring on the training slice itself (not just validation) is what lets
    the train/val gap be plotted directly as an overfitting signal across
    the size sweep — see `reporting.plot_train_val_gap`. With
    `train_acc_sample > 0`, train metrics are computed on a fixed seeded
    random subsample of that many training rows instead of all of them: at
    large sizes, predicting on the whole training set costs as much as
    fitting, and a few thousand rows estimate train accuracy to within
    about a point.

    Also returns the val decision scores (and, given `X_val_flip`, the
    flip-TTA scores averaged with each image's mirror) for evidence pooling.
    """
    import time

    clf = build_classifier(clf_name, X_train.shape[1])

    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    if 0 < train_acc_sample < len(X_train):
        idx = np.random.default_rng(seed).choice(len(X_train), size=train_acc_sample, replace=False)
        X_train_eval, y_train_eval = X_train[idx], y_train[idx]
    else:
        X_train_eval, y_train_eval = X_train, y_train

    t0 = time.perf_counter()
    y_train_pred = clf.predict(X_train_eval)
    train_predict_time = time.perf_counter() - t0
    train_accuracy = accuracy_score(y_train_eval, y_train_pred)
    train_precision, train_recall, train_f1, _ = precision_recall_fscore_support(
        y_train_eval, y_train_pred, average="macro", zero_division=0
    )

    t0 = time.perf_counter()
    y_val_pred = clf.predict(X_val)
    val_predict_time = time.perf_counter() - t0
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average="macro", zero_division=0
    )

    row = {
        "preprocess_level": preprocess_level,
        "feature_level": feature_level,
        "classifier": clf_name,
        "train_size_key": train_size_key,
        "n_train_per_class": n_per_class,
        # Rows actually trained on — 4 * n_per_class for flip-augmented levels.
        "n_train_total": len(X_train),
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
    scores = decision_scores(clf, X_val)
    scores_tta = (scores + decision_scores(clf, X_val_flip)) / 2 if X_val_flip is not None else scores
    return row, {"scores": scores, "scores_tta": scores_tta}


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
    block_names = blocks_for(_resolve_feature_levels(args.feature_level))
    parts = ["train_pool", "val", "test"] if args.split_part == "all" else [args.split_part]
    cache_dir = Path(args.cache_dir)

    for level in levels:
        for part in parts:
            if part == "train_pool":
                load_train_pool(split, level, cache_dir, block_names, args.force_recompute_features)
                continue
            if part == "val":
                records = split.val_records
            elif part == "test":
                records = split.test_records
            else:
                raise ValueError(f"Unknown split_part {part!r}")
            load_eval_features(records, level, part, cache_dir, block_names, args.force_recompute_features)


def run_sweep_chunk(args: argparse.Namespace) -> None:
    split = _load_split(args)
    levels = _resolve_preprocess_levels(args.preprocess_level)
    feature_levels = _resolve_feature_levels(args.feature_level)
    block_names = blocks_for(feature_levels)
    classifiers = [c.strip() for c in args.classifiers.split(",") if c.strip()]
    for c in classifiers:
        if c not in CLASSIFIER_BUILDERS:
            raise ValueError(f"Unknown classifier {c!r}; choices: {sorted(CLASSIFIER_BUILDERS)}")

    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    chunks_dir = results_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    existing_keys = _load_all_chunk_keys(chunks_dir)
    val_signature = records_signature(split.val_records)

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
            train_blocks, flip_blocks, pool_max = load_train_pool(
                split, level, cache_dir, block_names, args.force_recompute_features
            )
            val_blocks, val_flip_blocks, val_labels = load_eval_features(
                split.val_records, level, "val", cache_dir, block_names, args.force_recompute_features
            )

            for feature_level in feature_levels:
                X_val = build_feature_vector(val_blocks, feature_level)
                X_val_flip = build_feature_vector(val_flip_blocks, feature_level)

                tasks = []
                for size_key, n in split.train_sizes_resolved.items():
                    for clf_name in classifiers:
                        if is_kernel_svm(clf_name) and n > args.svm_max_size:
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
                            has_scores = score_path(results_dir, level.name, feature_level, clf_name, size_key).exists()
                            if not (args.backfill_scores and not has_scores):
                                logger.info("Skipping already-completed combo %s", key)
                                continue
                            logger.info("Re-running %s only to save its val scores (--backfill-scores)", key)
                        tasks.append((size_key, n, clf_name))

                def run_task(size_key: str, n: int, clf_name: str):
                    X_train, y_train = training_matrix(train_blocks, flip_blocks, feature_level, pool_max, n)
                    return run_single_combo(
                        level.name,
                        feature_level,
                        clf_name,
                        size_key,
                        n,
                        X_train,
                        y_train,
                        X_val,
                        val_labels,
                        args.seed,
                        args.train_acc_sample,
                        X_val_flip,
                    )

                # Threads, not processes: libsvm releases the GIL while
                # fitting, so single-threaded SVC fits genuinely run in
                # parallel while sharing the (large) cached feature matrices
                # instead of copying them into each worker. Rows are written
                # (and flushed) as each fit finishes, in completion order.
                results = Parallel(n_jobs=args.n_jobs, prefer="threads", return_as="generator_unordered")(
                    delayed(run_task)(*task) for task in tasks
                )
                for row, val_scores in results:
                    # Scores first: a row in the CSV means "done", so its
                    # scores must already be on disk by then.
                    _, acc_tta = save_scores(
                        score_path(results_dir, row["preprocess_level"], row["feature_level"], row["classifier"], row["train_size_key"]),
                        val_scores["scores"],
                        val_scores["scores_tta"],
                        val_labels,
                        val_signature,
                    )
                    writer.writerow(row)
                    f.flush()
                    existing_keys.add(
                        (row["preprocess_level"], row["feature_level"], row["classifier"], row["train_size_key"], str(args.seed))
                    )
                    logger.info(
                        "%s/%s/%s/size=%s: train_acc=%.4f val_acc=%.4f val_acc_flipTTA=%.4f (gap=%.4f, fit %.1fs)",
                        row["preprocess_level"],
                        row["feature_level"],
                        row["classifier"],
                        row["train_size_key"],
                        row["train_accuracy"],
                        row["val_accuracy"],
                        acc_tta,
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


def _guard_test_set(results_dir: Path, archive_previous: bool) -> None:
    """Refuse to look at the test set again unless explicitly told to archive the previous look.

    The homework allows exactly one test-set evaluation. An earlier run
    (2026-09-23, an end-to-end smoke test — see experiment_log.md) already
    produced final_report.json; the real final evaluation has to be a
    deliberate, logged decision, not an accident.
    """
    previous = results_dir / "final_report.json"
    if not previous.exists():
        return
    if not archive_previous:
        raise RuntimeError(
            f"{previous} already exists, i.e. the test set has been evaluated before. The test set "
            "may only be used once for the real final evaluation. If this IS that one deliberate run, "
            "re-run with --archive-previous (the old report is kept, renamed, for the write-up). "
            "To check the pipeline without touching the test set, use --dry-run-on-val."
        )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    dest = results_dir / f"final_report.previous_{stamp}.json"
    previous.rename(dest)
    logger.warning("Archived the previous test-set report to %s", dest)


def run_final_eval(args: argparse.Namespace) -> None:
    import json

    import reporting

    results_dir = Path(args.results_dir)
    cache_dir = Path(args.cache_dir)
    split = _load_split(args)

    if args.jury is not None:
        jury = json.loads(Path(args.jury).read_text())
        if args.dry_run_on_val:
            report = reporting.evaluate_jury(jury, split, cache_dir, part="val")
            out = results_dir / "jury_dryrun_val.json"
            reporting.save_final_report(report, out)
            logger.info(
                "DRY RUN on val (test set untouched): pooled val accuracy %.4f; `pool` reported %.4f for this jury.",
                report["accuracy"],
                jury["val_accuracy"],
            )
            return
        _guard_test_set(results_dir, args.archive_previous)
        report = reporting.evaluate_jury(jury, split, cache_dir, part="test")
        reporting.save_final_report(report, results_dir / "final_report.json")
        reporting.plot_confusion_matrix(np.array(report["confusion_matrix"]), results_dir / "final_confusion_matrix.png")
        logger.info("Final test accuracy (jury): %.4f", report["accuracy"])
        return

    if args.dry_run_on_val:
        raise RuntimeError("--dry-run-on-val is only implemented together with --jury")

    results_path = results_dir / "sweep_results.csv"
    if not results_path.exists():
        raise RuntimeError(f"{results_path} not found — run `merge` after at least one `sweep` chunk first.")
    results_df = pd.read_csv(results_path)

    best_row = reporting.select_best_combo(results_df)
    logger.info("Best validated combo:\n%s", best_row)

    _guard_test_set(results_dir, args.archive_previous)
    report = reporting.evaluate_final_on_test(best_row, split, cache_dir)
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


def run_pool(args: argparse.Namespace) -> None:
    import pooling

    member_keys = None if args.members == "auto" else [m.strip() for m in args.members.split(",") if m.strip()]
    rules = [r.strip() for r in args.rules.split(",") if r.strip()]
    ks = [int(k) for k in args.k.split(",") if k.strip()]
    result = pooling.run_pool(Path(args.results_dir), member_keys, rules, ks, min_size=args.min_size)

    df = pd.DataFrame(result["rows"])
    cols = ["rule", "k", "tta", "val_accuracy", "delta", "only_jury_right", "only_single_right", "mcnemar_p"]
    print(df.sort_values("val_accuracy", ascending=False)[cols].head(15).to_string(index=False))
    jury = result["jury"]
    print(f"\nBest jury -> results/jury.json: rule={jury['rule']} tta={jury['use_tta']} val_acc={jury['val_accuracy']:.4f}")
    for m in jury["members"]:
        print(f"  {m['preprocess_level']}/{m['feature_level']}/{m['classifier']}/{m['train_size_key']}  (val {m['val_accuracy']:.4f})")


def _laplacian_variance(path: Path, size: int) -> float:
    from PIL import Image
    from scipy import ndimage

    from subject import center_square_box

    with Image.open(path) as img:
        img = img.convert("L")
        img = img.crop(center_square_box(*img.size)).resize((size, size), Image.Resampling.BILINEAR)
        return float(ndimage.laplace(np.asarray(img, dtype=np.float64)).var())


def run_blur_report(args: argparse.Namespace) -> None:
    """How blurry is the data, at the resolutions the features actually see?

    Laplacian variance (low = few sharp edges = blurry) on a seeded random
    sample of the training pool, at 128 and 256 px. Saves a histogram and a
    grid of the blurriest images to results/debug/ so it's visible whether
    "blurry" images really are blurred or just smooth (e.g. a white cat on
    a white sofa).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    split = _load_split(args)
    pool = [p for p, _ in training_subset(split, min(len(split.cat_train_pool), len(split.dog_train_pool)))]
    rng = np.random.default_rng(args.seed)
    sample = [pool[i] for i in rng.choice(len(pool), size=min(args.n, len(pool)), replace=False)]

    out_dir = Path(args.results_dir) / "debug"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, size in zip(axes, (128, 256)):
        v = np.array(Parallel(n_jobs=-1)(delayed(_laplacian_variance)(p, size) for p in sample))
        q = np.percentile(v, [5, 25, 50, 75, 95])
        print(f"{size}px Laplacian variance percentiles 5/25/50/75/95: " + " / ".join(f"{x:.0f}" for x in q))
        ax.hist(np.log10(v + 1), bins=60)
        ax.set_xlabel("log10(Laplacian variance + 1)   (left = blurrier)")
        ax.set_title(f"{size}x{size}, n={len(v)}")
        if size == 128:
            v128 = v
    fig.tight_layout()
    fig.savefig(out_dir / "blur_hist.png", dpi=110)
    plt.close(fig)

    order = np.argsort(v128)
    fig, axes = plt.subplots(3, 8, figsize=(16, 6.5))
    for ax, i in zip(axes.ravel(), order[:24]):
        with Image.open(sample[i]) as img:
            ax.imshow(img.convert("RGB"))
        ax.set_title(f"{v128[i]:.0f}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Blurriest 24 of the sample (Laplacian variance at 128px)")
    fig.tight_layout()
    fig.savefig(out_dir / "blur_examples.png", dpi=80)
    plt.close(fig)
    print(f"Saved {out_dir / 'blur_hist.png'} and {out_dir / 'blur_examples.png'}")


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

    level_help = (
        "Comma-separated preprocessing level(s), or 'all' (= the original "
        + ",".join(DEFAULT_PREPROCESS_LEVELS)
        + "). Available: "
        + ",".join(PREPROCESS_LEVELS_BY_NAME)
    )
    feature_help = (
        "Comma-separated feature level(s), or 'all' (= the original "
        + ",".join(DEFAULT_FEATURE_LEVELS)
        + "). Available: "
        + ",".join(FEATURE_LEVELS)
    )

    p_extract = subparsers.add_parser("extract-features", help="Compute/cache feature blocks.")
    p_extract.add_argument("--preprocess-level", default="all", help=level_help)
    p_extract.add_argument("--feature-level", default="all", help=feature_help + " (extracts the blocks these need)")
    p_extract.add_argument("--split-part", choices=["train_pool", "val", "test", "all"], default="all")

    p_sweep = subparsers.add_parser("sweep", help="Train+validate classifiers; resumable, writes its own chunk CSV.")
    p_sweep.add_argument("--preprocess-level", default="all", help=level_help)
    p_sweep.add_argument("--feature-level", default="all", help=feature_help)
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
    p_sweep.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help=(
            "Classifier fits to run concurrently (threads). Worth it for single-threaded models "
            "(SVC, LinearSVC, decision trees); keep at 1 for multithreaded ones (random forest, "
            "hgb, lightgbm, xgboost). Each concurrent max-size fit needs ~1.5-3 GB of RAM."
        ),
    )
    p_sweep.add_argument(
        "--train-acc-sample",
        type=int,
        default=0,
        help="Compute train metrics on a seeded random subsample of this many training rows (0 = all rows).",
    )
    p_sweep.add_argument(
        "--backfill-scores",
        action="store_true",
        help=(
            "Also re-run already-completed combos that have no saved val scores (results/val_scores/), "
            "so older results can join a jury. Their CSV rows are re-written; `merge` keeps the newest."
        ),
    )

    subparsers.add_parser("merge", help="Combine chunk CSVs into results/sweep_results.csv.")

    p_pool = subparsers.add_parser("pool", help="Evidence pooling: evaluate juries of saved models on val.")
    p_pool.add_argument(
        "--members",
        default="auto",
        help="'auto' (greedy diverse selection) or comma-separated score-file stems from results/val_scores/.",
    )
    p_pool.add_argument("--rules", default=",".join(RULES), help="Comma-separated: " + ",".join(RULES))
    p_pool.add_argument("--k", default="3,5,7", help="Jury sizes to try (with --members auto).")
    p_pool.add_argument(
        "--min-size", type=int, default=None, help="Only consider members trained on at least this many images/class."
    )

    p_blur = subparsers.add_parser("blur-report", help="Diagnostic: blur (Laplacian variance) histogram.")
    p_blur.add_argument("--n", type=int, default=3000, help="Number of training-pool images to sample.")

    p_final = subparsers.add_parser("final-eval", help="Evaluate the best validated combo (or a jury) once on the test set.")
    p_final.add_argument("--jury", type=Path, default=None, help="Evaluate this jury (from `pool`) instead of the best single combo.")
    p_final.add_argument(
        "--dry-run-on-val",
        action="store_true",
        help="With --jury: run the full evaluation code path on the VAL set instead (test set untouched).",
    )
    p_final.add_argument(
        "--archive-previous",
        action="store_true",
        help="Required when a previous final_report.json exists: archives it and runs the (one) real test evaluation.",
    )

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
    elif args.command == "pool":
        run_pool(args)
    elif args.command == "blur-report":
        run_blur_report(args)
    elif args.command == "final-eval":
        run_final_eval(args)
    else:
        raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    main()
