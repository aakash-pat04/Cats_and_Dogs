# Experiment log

Running notes on what was tried, what was observed, and why the pipeline
changed as a result — written to be usable more or less directly in the
homework report's "what worked well and what did not" discussion.

## 2026-09-23 — P0_minimal sweep shows severe overfitting; added capacity-limited classifiers

### What was run

The first full sweep chunk: preprocessing level `P0_minimal` (resize 128x128,
grayscale, no denoise/equalize), both feature levels (`hog`, `hog_lbp`), all
7 per-class training sizes (80, 200, 500, 1000, 5000, 10000, max=11,498), with
the three classifiers from the homework's example table — all at their
**default, unconstrained** hyperparameters:

- `decision_tree`: `DecisionTreeClassifier(random_state=42)` — no `max_depth`.
- `random_forest`: `RandomForestClassifier(n_estimators=100, random_state=42)` — no `max_depth`.
- `svm_rbf`: `SVC(kernel="rbf", C=10, random_state=42)` — the homework's suggested `C`, default `gamma="scale"`.

### What we saw

Training accuracy was ~100% for every classifier, at every training size,
including the largest (22,996 total training images):

| classifier | feature | size=80 train/val | size=1000 train/val | size=max train/val |
|---|---|---|---|---|
| decision_tree | hog | 1.000 / 0.567 | 1.000 / 0.559 | 0.99996 / 0.572 |
| random_forest | hog | 1.000 / 0.638 | 1.000 / 0.685 | 0.99996 / 0.704 |
| svm_rbf | hog | 1.000 / 0.654 | 1.000 / 0.713 | *(not run past 1,000/class — see below)* |

Full numbers in `results/sweep_results.csv` after merging.

Two things stood out:

1. **Decision Tree's validation accuracy never really moved** — it stayed
   in the 53-59% range from 80 training images per class all the way to
   11,498 per class, while its training accuracy stayed pinned at ~100%
   the entire time. More data didn't help it generalize; it just kept
   perfectly memorizing a larger dataset.
2. **Random Forest and SVM did improve with more data** (RF: 63.8% val
   accuracy at size=80 -> 70.4% at size=max; SVM: 65.4% at size=80 -> 71.3%
   at size=1000) even though *their* training accuracy also stayed at
   ~100% throughout. So more data clearly helped — just not by reducing
   the train/val gap, which stayed large (RF: ~30 points at size=max;
   SVM: ~29 points at size=1000).

### Why this happens (not a bug)

We checked the metric-computation code specifically to rule out a labeling
or data-leakage bug before concluding this was real: the cat/dog label
assignment for each training slice, the `StandardScaler` fit-train-only
rule, and the train-accuracy predict call were all verified correct (see
`dataset_split.py`'s `training_subset` ordering guarantee and
`run_experiments.py::run_single_combo`). The overfitting is a genuine
consequence of the classifiers' hyperparameters, not a measurement error:

- `DecisionTreeClassifier` and `RandomForestClassifier` default to
  `max_depth=None`, i.e. "grow until every leaf is pure." With 8,100 (HOG)
  or 8,110 (HOG+LBP) continuous-valued features, it's essentially always
  possible to carve out a leaf that isolates any single training example,
  so near-100% training accuracy is the *expected* behavior of an
  unconstrained tree on this kind of data, not an anomaly.
- `SVC(kernel="rbf", C=10)` with the default `gamma="scale"`, evaluated
  where the sample count (<=2,000 for the sizes SVM was run at) is small
  relative to the 8,100+ feature dimensions, sits in the classic
  "more dimensions than data" regime where an RBF kernel can effectively
  interpolate the training set.

A side finding worth noting: Random Forest's fit time at the largest size
(~78s) was much lower than a single unconstrained Decision Tree's (~407s),
despite training 100 trees. This is because `RandomForestClassifier`
defaults to considering only `sqrt(n_features)` (~90 of 8,100) features at
each split, versus `DecisionTreeClassifier`'s default of considering all of
them — cheaper per-split search more than offsets building 100 trees.

We also confirmed the earlier decision to cap RBF-SVM to the four smaller
training sizes by default was justified on runtime grounds alone: fit time
at 1,000/class was already 16-20s, with *predict* time on top of that
reaching 26-29s (predicting on the training set to compute train accuracy
adds a second O(n) SVM inference pass) — consistent with RBF-SVM's known
poor scaling well before considering overfitting.

### What we changed

Added three capacity-limited classifier variants to `CLASSIFIER_BUILDERS`
in `run_experiments.py`, **alongside** (not replacing) the originals, so
both can be compared directly in the same `sweep_results.csv`:

- `decision_tree_capped`: `DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=42)`
- `random_forest_capped`: `RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42)`
- `svm_rbf_low_c`: `SVC(kernel="rbf", C=0.1, random_state=42)` — 100x smaller `C` than the original (more regularization, same `gamma`)

These are a first-pass, reasonable-but-not-tuned choice (not a grid search)
— `max_depth=10`/`min_samples_leaf=5` is a common starting point for
capping tree capacity, and `C=0.1` is a large step down from the homework's
suggested `C=10` to make the regularization effect clearly visible. The
validation set could be used for a proper hyperparameter sweep over these
later if there's time.

This is purely additive: existing rows in `sweep_results.csv`/chunk CSVs
are untouched (new classifier *names*, not a schema change), and the
original overfitting data stays intact for comparison.

### How to reproduce / extend

```bash
# Run just the new capacity-limited variants (features already cached):
python run_experiments.py sweep --preprocess-level P0_minimal --feature-level all \
    --classifiers decision_tree_capped,random_forest_capped,svm_rbf_low_c

# Or run everything (originals + capped) in one pass for a new preprocessing level:
python run_experiments.py sweep --preprocess-level P1_equalize --feature-level all \
    --classifiers decision_tree,decision_tree_capped,random_forest,random_forest_capped,svm_rbf,svm_rbf_low_c
```

### What to look for next

The actual hypothesis being tested: do the capped variants show a
**smaller train/val gap** (via `results/train_val_gap.png` once `merge` +
`final-eval` are re-run) while holding validation accuracy roughly flat or
better — i.e. does constraining capacity trade away memorization without
costing real generalization? If `*_capped`/`*_low_c` validation accuracy
comes in close to or above the originals with a visibly smaller gap, that's
a clean "regularization helped" result for the report. If it comes in
noticeably lower, that's also worth reporting — it would mean these
particular models needed the extra capacity to fit the signal, and the
overfitting we saw was a more unavoidable side effect of the feature
representation than of hyperparameter choice alone.
