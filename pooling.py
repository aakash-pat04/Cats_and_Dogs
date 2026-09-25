"""Evidence pooling: a "jury" of several trained models voting on each image.

Every sweep run saves its validation-set decision scores to
`results/val_scores/<preprocess>__<feature>__<classifier>__<size>.npz`
(plus flip test-time-augmentation scores: the average over each val image
and its mirror). This module combines those saved scores without
retraining anything:

- majority:  each member votes cat/dog; the majority wins (use odd K).
- soft:      average of each member's decision score, each divided by its
             own val-score standard deviation so SVM margins and
             tree-model log-odds are on a comparable scale (no labels used).
- weighted:  like majority, but each vote counts log(acc / (1 - acc)) of
             that member's validation accuracy (the classic weighted
             majority vote).

`select_jury` greedily adds the member that most improves the pooled val
accuracy, requiring each new member to differ from every existing one in
preprocessing, features, or model family — a jury of five near-copies of
one SVM would just repeat the same mistakes. Choosing the jury on val is
optimistic in the same way picking any single best model on val is; the
one-time test-set evaluation (`final-eval --jury`) is the unbiased number.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest

logger = logging.getLogger(__name__)

RULES = ("majority", "soft", "weighted")
SCORES_DIRNAME = "val_scores"


def decision_scores(clf, X: np.ndarray) -> np.ndarray:
    """Real-valued score per row, > 0 meaning "dog" (label 1).

    SVMs (and HistGradientBoosting) expose `decision_function`; models that
    only have `predict_proba` (LightGBM, XGBoost, random forest) are mapped
    to log-odds so every model's score is centered on 0 at its decision
    boundary.
    """
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(X), dtype=np.float64).ravel()
    proba = np.clip(clf.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(proba / (1 - proba))


def records_signature(records: Iterable[Tuple[Path, int]]) -> str:
    hasher = hashlib.sha256()
    for path, label in records:
        hasher.update(f"{Path(path).name}|{label}\n".encode())
    return hasher.hexdigest()


def score_path(results_dir: Path, preprocess_level: str, feature_level: str, classifier: str, size_key: str) -> Path:
    return Path(results_dir) / SCORES_DIRNAME / f"{preprocess_level}__{feature_level}__{classifier}__{size_key}.npz"


def save_scores(
    path: Path, scores: np.ndarray, scores_tta: np.ndarray, labels: np.ndarray, signature: str
) -> Tuple[float, float]:
    """Write one member's val scores; returns (accuracy, accuracy with flip TTA)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    acc = float(((scores > 0).astype(int) == labels).mean())
    acc_tta = float(((scores_tta > 0).astype(int) == labels).mean())
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        scores=scores.astype(np.float32),
        scores_tta=scores_tta.astype(np.float32),
        labels=labels.astype(np.int8),
        accuracy=np.array(acc),
        accuracy_tta=np.array(acc_tta),
        signature=np.array(signature),
    )
    tmp.replace(path)
    return acc, acc_tta


@dataclass
class Member:
    preprocess_level: str
    feature_level: str
    classifier: str
    train_size_key: str
    scores: np.ndarray
    scores_tta: np.ndarray
    labels: np.ndarray
    accuracy: float
    accuracy_tta: float
    signature: str

    @property
    def key(self) -> str:
        return f"{self.preprocess_level}__{self.feature_level}__{self.classifier}__{self.train_size_key}"

    def as_dict(self, use_tta: bool) -> Dict[str, object]:
        return {
            "preprocess_level": self.preprocess_level,
            "feature_level": self.feature_level,
            "classifier": self.classifier,
            "train_size_key": self.train_size_key,
            "val_accuracy": self.accuracy_tta if use_tta else self.accuracy,
        }

    def get(self, use_tta: bool) -> Tuple[np.ndarray, float]:
        return (self.scores_tta, self.accuracy_tta) if use_tta else (self.scores, self.accuracy)


def load_members(results_dir: Path, keys: Optional[Sequence[str]] = None) -> List[Member]:
    """Load saved val scores (all of them, or only `keys` = file stems)."""
    score_dir = Path(results_dir) / SCORES_DIRNAME
    paths = sorted(score_dir.glob("*.npz")) if keys is None else [score_dir / f"{k}.npz" for k in keys]
    members = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"No saved val scores at {p}")
        parts = p.stem.split("__")
        if len(parts) != 4:
            logger.warning("Skipping unexpected file name %s", p)
            continue
        with np.load(p, allow_pickle=False) as d:
            members.append(
                Member(
                    *parts,
                    scores=d["scores"].astype(np.float64),
                    scores_tta=d["scores_tta"].astype(np.float64),
                    labels=d["labels"].astype(int),
                    accuracy=float(d["accuracy"]),
                    accuracy_tta=float(d["accuracy_tta"]),
                    signature=str(d["signature"]),
                )
            )
    if members:
        # Every member must have been scored on the same val images in the same order.
        ref = members[0].signature
        bad = [m.key for m in members if m.signature != ref]
        if bad:
            raise ValueError(f"Val scores were computed on different val sets (signature mismatch): {bad}")
    return members


def model_family(classifier: str) -> str:
    if classifier.startswith(("svm_rbf", "pca_svm", "pca1024_svm", "hellinger_svm", "nystroem")):
        return "rbf_svm"
    if classifier.startswith(("linsvc", "hellinger_linsvc")):
        return "linear_svm"
    if classifier in ("lightgbm", "xgboost", "hgb"):
        return "boosted_trees"
    if classifier.startswith("random_forest"):
        return "random_forest"
    if classifier.startswith("decision_tree"):
        return "decision_tree"
    return classifier


def combine(scores: Sequence[np.ndarray], accuracies: Sequence[float], rule: str) -> np.ndarray:
    """Pooled decision per image (> 0 = dog) from members' scores under `rule`."""
    S = np.stack(scores)  # (members, images)
    if rule == "majority":
        return np.sign(S).sum(axis=0)
    if rule == "soft":
        std = S.std(axis=1, keepdims=True)
        return (S / np.where(std > 0, std, 1.0)).mean(axis=0)
    if rule == "weighted":
        acc = np.clip(np.asarray(accuracies, dtype=np.float64), 0.501, 0.999)
        weights = np.log(acc / (1 - acc))
        return (weights[:, None] * np.sign(S)).sum(axis=0)
    raise ValueError(f"Unknown pooling rule {rule!r}; expected one of {RULES}")


def pooled_accuracy(members: Sequence[Member], rule: str, use_tta: bool) -> Tuple[float, np.ndarray]:
    scores, accs = zip(*(m.get(use_tta) for m in members))
    pred = (combine(scores, accs, rule) > 0).astype(int)
    return float((pred == members[0].labels).mean()), pred


def mcnemar(labels: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> Tuple[int, int, float]:
    """Exact McNemar test: (#only A right, #only B right, two-sided p-value)."""
    a_right = pred_a == labels
    b_right = pred_b == labels
    b = int((a_right & ~b_right).sum())
    c = int((~a_right & b_right).sum())
    p = 1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return b, c, p


def _diverse(candidate: Member, chosen: Sequence[Member]) -> bool:
    return all(
        (candidate.preprocess_level, candidate.feature_level, model_family(candidate.classifier))
        != (m.preprocess_level, m.feature_level, model_family(m.classifier))
        for m in chosen
    )


def select_jury(members: Sequence[Member], rule: str, k: int, use_tta: bool) -> List[Member]:
    """Greedy forward selection of up to `k` diverse members maximizing pooled val accuracy."""
    ranked = sorted(members, key=lambda m: m.get(use_tta)[1], reverse=True)
    chosen = [ranked[0]]
    while len(chosen) < k:
        best, best_acc = None, -1.0
        for cand in ranked:
            if cand in chosen or not _diverse(cand, chosen):
                continue
            acc, _ = pooled_accuracy([*chosen, cand], rule, use_tta)
            if acc > best_acc:
                best, best_acc = cand, acc
        if best is None:
            break
        chosen.append(best)
    return chosen


def run_pool(
    results_dir: Path,
    member_keys: Optional[Sequence[str]],
    rules: Sequence[str],
    ks: Sequence[int],
    min_size: Optional[int] = None,
) -> Dict[str, object]:
    """Evaluate pooling rules/jury sizes on val; write pooling_results.csv and jury.json (the best)."""
    members = load_members(results_dir, member_keys)
    if min_size is not None:
        members = [m for m in members if m.train_size_key == "max" or int(m.train_size_key) >= min_size]
    if not members:
        raise RuntimeError(f"No saved val scores found under {Path(results_dir) / SCORES_DIRNAME}")
    labels = members[0].labels
    logger.info("Pooling over %d candidate member(s)", len(members))

    rows = []
    seen = set()
    best_entry = None
    for use_tta in (False, True):
        single = max(members, key=lambda m: m.get(use_tta)[1])
        single_pred = (single.get(use_tta)[0] > 0).astype(int)
        for rule in rules:
            for k in ks:
                jury = list(members) if member_keys is not None else select_jury(members, rule, k, use_tta)
                signature = (rule, use_tta, tuple(m.key for m in jury))
                if signature in seen:
                    continue  # fewer diverse members than k: same jury as a smaller k
                seen.add(signature)
                acc, pred = pooled_accuracy(jury, rule, use_tta)
                only_jury, only_single, p = mcnemar(labels, pred, single_pred)
                row = {
                    "rule": rule,
                    "k_requested": k,
                    "k": len(jury),
                    "tta": use_tta,
                    "val_accuracy": acc,
                    "best_single": single.key,
                    "best_single_accuracy": single.get(use_tta)[1],
                    "delta": acc - single.get(use_tta)[1],
                    "only_jury_right": only_jury,
                    "only_single_right": only_single,
                    "mcnemar_p": p,
                    "members": ";".join(m.key for m in jury),
                }
                rows.append(row)
                candidate = (acc, -len(jury))
                if best_entry is None or candidate > best_entry[0]:
                    best_entry = (
                        candidate,
                        {
                            "rule": rule,
                            "use_tta": use_tta,
                            "val_accuracy": acc,
                            "members": [m.as_dict(use_tta) for m in jury],
                            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        },
                    )
                if member_keys is not None:
                    break  # explicit member list: k doesn't apply

    out_csv = Path(results_dir) / "pooling_results.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    jury = best_entry[1]
    (Path(results_dir) / "jury.json").write_text(json.dumps(jury, indent=2))
    logger.info("Wrote %s (%d rows) and %s", out_csv, len(rows), Path(results_dir) / "jury.json")
    return {"rows": rows, "jury": jury}
