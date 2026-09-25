"""Post-sweep analysis: pick the winning combo, plot a learning curve, and
run the single, one-time held-out test-set evaluation (accuracy, confusion
matrix, classification report) for the homework report.

`evaluate_final_on_test` and `evaluate_jury` (with part="test") are the
only functions in this codebase that read `DatasetSplit.test_records` —
everything else in the sweep only ever touches the validation set, so the
test set is never used for repeated tuning.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from dataset_split import DatasetSplit
from features import FEATURE_LEVELS, PREPROCESS_LEVELS_BY_NAME, build_feature_vector, load_feature_blocks, training_matrix
from pooling import combine, decision_scores

logger = logging.getLogger(__name__)


def select_best_combo(results_df: pd.DataFrame) -> pd.Series:
    """Highest validation accuracy across the whole sweep; fit time breaks exact ties.

    This is the only place model/feature/classifier selection happens, and
    it only ever looks at validation columns — never at anything test-set
    related, per the homework's "don't tune on the test set" rule.
    """
    if results_df.empty:
        raise ValueError("results_df is empty — nothing to select from.")
    ordered = results_df.sort_values(["val_accuracy", "fit_time_sec"], ascending=[False, True])
    return ordered.iloc[0]


def plot_learning_curves(
    results_df: pd.DataFrame, out_path: Path, fixed_preprocess: str, fixed_feature: str
) -> None:
    """Train + validation accuracy vs. training size (log-x), one color per classifier.

    Solid lines are validation accuracy, dashed lines (same color) are
    training accuracy for the same combo — plotting both on one axis makes
    the train/val gap visible directly, not just inferable. Restricted to
    the winning preprocessing+feature combo so the plot reads as a single,
    clean "does more data help" comparison across classifiers.
    """
    subset = results_df[
        (results_df["preprocess_level"] == fixed_preprocess) & (results_df["feature_level"] == fixed_feature)
    ].copy()
    if subset.empty:
        logger.warning("No rows match preprocess_level=%s, feature_level=%s; skipping plot", fixed_preprocess, fixed_feature)
        return

    fig, ax = plt.subplots(figsize=(7, 5))
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for (clf_name, group), color in zip(subset.groupby("classifier"), color_cycle):
        group = group.sort_values("n_train_per_class")
        ax.plot(group["n_train_per_class"], group["val_accuracy"], marker="o", color=color, label=f"{clf_name} (val)")
        ax.plot(
            group["n_train_per_class"],
            group["train_accuracy"],
            marker="x",
            linestyle="--",
            color=color,
            alpha=0.6,
            label=f"{clf_name} (train)",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Training images per class (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Learning curve — {fixed_preprocess} / {fixed_feature}")
    ax.legend(fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved learning curve to %s", out_path)


def plot_train_val_gap(
    results_df: pd.DataFrame, out_path: Path, fixed_preprocess: str, fixed_feature: str
) -> None:
    """Train-minus-validation accuracy vs. training size — an explicit overfitting-gap chart.

    A shrinking gap as training size grows is the classic "more data helps
    generalization" signal; a gap that stays wide (or widens) points at a
    model that's overfitting regardless of how much data it gets.
    """
    subset = results_df[
        (results_df["preprocess_level"] == fixed_preprocess) & (results_df["feature_level"] == fixed_feature)
    ].copy()
    if subset.empty:
        logger.warning("No rows match preprocess_level=%s, feature_level=%s; skipping plot", fixed_preprocess, fixed_feature)
        return

    subset["train_val_gap"] = subset["train_accuracy"] - subset["val_accuracy"]

    fig, ax = plt.subplots(figsize=(7, 5))
    for clf_name, group in subset.groupby("classifier"):
        group = group.sort_values("n_train_per_class")
        ax.plot(group["n_train_per_class"], group["train_val_gap"], marker="o", label=clf_name)

    ax.axhline(0, color="gray", linewidth=1, linestyle=":")
    ax.set_xscale("log")
    ax.set_xlabel("Training images per class (log scale)")
    ax.set_ylabel("Train accuracy − validation accuracy")
    ax.set_title(f"Overfitting gap — {fixed_preprocess} / {fixed_feature}")
    ax.legend(fontsize="small")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved train/val gap plot to %s", out_path)


def evaluate_final_on_test(best_row: pd.Series, split: DatasetSplit, cache_dir: Path) -> Dict[str, object]:
    """Refit the winning combo on its exact training slice and score it once on the test set.

    The training data is rebuilt with the same helpers the sweep uses
    (including flip augmentation, if the winning level has it), and the
    classifier pipeline carries its own train-only StandardScaler.
    """
    from run_experiments import build_classifier, load_train_pool

    level = PREPROCESS_LEVELS_BY_NAME[best_row["preprocess_level"]]
    feature_level = best_row["feature_level"]
    block_names = list(FEATURE_LEVELS[feature_level])
    n = int(best_row["n_train_per_class"])

    train_blocks, flip_blocks, pool_max = load_train_pool(split, level, cache_dir, block_names)
    X_train, y_train = training_matrix(train_blocks, flip_blocks, feature_level, pool_max, n)
    test_blocks, test_labels, test_paths = load_feature_blocks(
        split.test_records, level, level.fc, "test", cache_dir, block_names
    )
    X_test = build_feature_vector(test_blocks, feature_level)

    clf = build_classifier(best_row["classifier"], X_train.shape[1])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    cm = confusion_matrix(test_labels, y_pred, labels=[0, 1])
    report_dict = classification_report(test_labels, y_pred, target_names=["cat", "dog"], output_dict=True)
    accuracy = float((y_pred == test_labels).mean())

    n_test_cat = int((test_labels == 0).sum())
    n_test_dog = int((test_labels == 1).sum())
    cat_as_dog = int(cm[0, 1])
    dog_as_cat = int(cm[1, 0])

    return {
        "best_combo": {
            "preprocess_level": best_row["preprocess_level"],
            "feature_level": best_row["feature_level"],
            "classifier": best_row["classifier"],
            "train_size_key": best_row["train_size_key"],
            "n_train_per_class": n,
            "val_accuracy": float(best_row["val_accuracy"]),
        },
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "n_test_cat": n_test_cat,
        "n_test_dog": n_test_dog,
        "cat_as_dog_count": cat_as_dog,
        "cat_as_dog_rate": cat_as_dog / n_test_cat if n_test_cat else None,
        "dog_as_cat_count": dog_as_cat,
        "dog_as_cat_rate": dog_as_cat / n_test_dog if n_test_dog else None,
        "test_paths": test_paths,
    }


def _size_to_n(split: DatasetSplit, size_key: str) -> int:
    pool_max = min(len(split.cat_train_pool), len(split.dog_train_pool))
    return pool_max if size_key == "max" else int(size_key)


def evaluate_jury(jury: Dict[str, object], split: DatasetSplit, cache_dir: Path, part: str) -> Dict[str, object]:
    """Refit every jury member on its training slice, score `part` ("val" or "test"), and pool the votes.

    `part="val"` is the dry run: the same code path as the real evaluation,
    checked against the pooled val accuracy `pool` reported, without
    touching the test set. `part="test"` is the one-time final evaluation.
    Pooling uses the rule, TTA setting and (for "weighted") the member val
    accuracies fixed in `jury` — nothing is re-tuned on the evaluation set.
    """
    from run_experiments import build_classifier, load_eval_features, load_train_pool

    if part not in ("val", "test"):
        raise ValueError(f"part must be 'val' or 'test', got {part!r}")
    records = split.val_records if part == "val" else split.test_records
    use_tta = bool(jury["use_tta"])

    member_scores: List[np.ndarray] = []
    member_accs: List[float] = []
    per_member = []
    labels = None
    for m in jury["members"]:
        level = PREPROCESS_LEVELS_BY_NAME[m["preprocess_level"]]
        feature_level = m["feature_level"]
        block_names = list(FEATURE_LEVELS[feature_level])
        n = _size_to_n(split, str(m["train_size_key"]))
        logger.info("Jury member %s/%s/%s/%s: fitting on %d/class", level.name, feature_level, m["classifier"], m["train_size_key"], n)

        train_blocks, flip_blocks, pool_max = load_train_pool(split, level, cache_dir, block_names)
        X_train, y_train = training_matrix(train_blocks, flip_blocks, feature_level, pool_max, n)
        clf = build_classifier(m["classifier"], X_train.shape[1])
        clf.fit(X_train, y_train)
        del X_train, train_blocks, flip_blocks

        eval_blocks, eval_flip_blocks, labels = load_eval_features(records, level, part, cache_dir, block_names)
        scores = decision_scores(clf, build_feature_vector(eval_blocks, feature_level))
        if use_tta:
            scores = (scores + decision_scores(clf, build_feature_vector(eval_flip_blocks, feature_level))) / 2
        member_scores.append(scores)
        member_accs.append(float(m["val_accuracy"]))
        per_member.append({**m, f"{part}_accuracy": float(((scores > 0).astype(int) == labels).mean())})

    y_pred = (combine(member_scores, member_accs, jury["rule"]) > 0).astype(int)
    cm = confusion_matrix(labels, y_pred, labels=[0, 1])
    report_dict = classification_report(labels, y_pred, target_names=["cat", "dog"], output_dict=True)
    n_cat, n_dog = int((labels == 0).sum()), int((labels == 1).sum())
    return {
        "evaluated_on": part,
        "jury": {"rule": jury["rule"], "use_tta": use_tta, "members": per_member},
        "accuracy": float((y_pred == labels).mean()),
        "confusion_matrix": cm.tolist(),
        "classification_report": report_dict,
        "n_test_cat": n_cat,
        "n_test_dog": n_dog,
        "cat_as_dog_count": int(cm[0, 1]),
        "cat_as_dog_rate": int(cm[0, 1]) / n_cat if n_cat else None,
        "dog_as_cat_count": int(cm[1, 0]),
        "dog_as_cat_rate": int(cm[1, 0]) / n_dog if n_dog else None,
    }


def plot_confusion_matrix(cm: np.ndarray, out_path: Path, labels=("cat", "dog")) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Cat vs. Dog Confusion Matrix (test set)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("Saved confusion matrix to %s", out_path)


def save_final_report(report: Dict[str, object], out_path: Path) -> None:
    # test_paths can be long; keep it out of the human-readable JSON summary.
    slim = {k: v for k, v in report.items() if k != "test_paths"}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(slim, indent=2))
    logger.info("Saved final report to %s", out_path)
