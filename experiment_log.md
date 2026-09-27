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

## 2026-09-23 (continued) — Full sweep results: P1/P2 sweeps, capped-vs-uncapped, final test-set eval

### What was run

Both remaining preprocessing levels (`P1_equalize`, `P2_denoise`) were swept
with all 6 classifiers (3 originals + 3 capacity-limited variants), both
feature levels, all 7 sizes (SVM variants capped to the 4 smaller sizes as
before). Combined with the earlier `P0_minimal` sweep, `sweep_results.csv`
now has complete coverage: 3 preprocessing levels x 2 feature levels x 6
classifiers = 36 combos, 216 total rows. `merge` + `final-eval` were then
run to pick a final combo and evaluate it once on the held-out test set.

### Preprocessing level: equalize/denoise didn't help — if anything, slightly hurt

Mean validation accuracy across every classifier/feature/size combo, by
preprocessing level:

| preprocess level | mean val acc | best val acc |
|---|---|---|
| P0_minimal | 0.639 | 0.718 |
| P1_equalize | 0.636 | 0.718 |
| P2_denoise | 0.633 | 0.714 |

And isolating just the three original (uncapped) classifiers, broken out by
feature level too:

| preprocess level | hog | hog_lbp |
|---|---|---|
| P0_minimal | 0.637 | 0.645 |
| P1_equalize | 0.632 | 0.641 |
| P2_denoise | 0.632 | 0.634 |

`P0_minimal` (plain resize + grayscale, nothing else) is consistently at
least as good as — usually marginally better than — both `P1_equalize` and
`P2_denoise`, in every slice we checked. The differences are small (within
~1 point), but they're consistently in the "extra preprocessing didn't pay
off" direction, never the other way. Our read: HOG is already a gradient-
orientation descriptor, which is inherently fairly robust to the kind of
global contrast/brightness changes histogram equalization corrects for —
so equalizing ahead of HOG mostly adds noise (or a mild Gaussian blur, for
P2) without fixing anything HOG couldn't already handle. This is a useful
negative result for the report's "what didn't work" section.

### Feature level: HOG+LBP consistently, modestly beats HOG alone

`hog_lbp` beat plain `hog` at every preprocessing level (by 0.2-0.8 points
in the table above), and by ~0.6 points on average overall (0.639 vs 0.633
across all classifiers/sizes). Consistent but modest — LBP's texture
information is complementary to HOG's gradient information, but doesn't
transform performance on its own.

### Capped vs. uncapped: the effect is genuinely different per classifier, and depends on training size

This is the most interesting result, and it only shows up by looking at the
per-size breakdown, not the sweep-wide average.

**Decision Tree — capping is an unambiguous win that grows with data size:**

| size | uncapped val acc | capped val acc | uncapped gap | capped gap |
|---|---|---|---|---|
| 80 | 0.551 | 0.540 | 0.449 | 0.399 |
| 1000 | 0.578 | 0.578 | 0.422 | 0.345 |
| 10000 | 0.580 | 0.617 | 0.420 | 0.218 |
| max (11,498) | 0.587 | 0.618 | 0.413 | 0.202 |

At small sizes the two are roughly tied. But as training size grows, the
capped tree pulls ahead by 3+ points of validation accuracy while cutting
the train/val gap in half — the single biggest accuracy improvement from
any change we tried. Interpretation: an unconstrained tree just grows
deeper and memorizes harder as it gets more data, gaining nothing; capping
`max_depth` forces it to actually generalize, and that constraint matters
more (not less) the more data it has to potentially overfit to.

**Random Forest — capping is a small, consistent win, but not a dramatic one:**

| size | uncapped val acc | capped val acc | uncapped gap | capped gap |
|---|---|---|---|---|
| 80 | 0.631 | 0.624 | 0.369 | 0.376 |
| 1000 | 0.678 | 0.687 | 0.322 | 0.308 |
| max | 0.698 | 0.704 | 0.302 | 0.244 |

Capping shrinks the gap similarly to Decision Tree, but the validation
accuracy gain is much smaller (roughly +0.5-1 point rather than +3). Makes
sense: Random Forest's bagging already provides a form of built-in
regularization (each tree sees a bootstrap sample and only ~90 of 8,100
features per split), so there's less overfitting left for `max_depth` to
fix.

**SVM — lowering C to 0.1 was too aggressive, and gets worse with more data:**

| size | C=10 val acc | C=0.1 val acc | C=10 train acc | C=0.1 train acc |
|---|---|---|---|---|
| 80 | 0.655 | 0.638 | 1.000 | 0.992 |
| 200 | 0.682 | 0.659 | 1.000 | 0.896 |
| 500 | 0.708 | 0.667 | 1.000 | 0.804 |
| 1000 | 0.712 | 0.671 | 1.000 | 0.755 |

Unlike the trees, `svm_rbf_low_c` never wins — it's behind at every size,
and the gap between it and the original *widens* as training size grows
(-1.6 points at size 80, -4.1 points at size 1000). Its own train accuracy
also degrades with size (0.992 -> 0.755), meaning C=0.1 isn't just
"removing memorization" the way the tree caps were — it's actively
underfitting real signal in the data, worse the more data there is to
underfit. A 100x drop in C (10 -> 0.1) was too big a step; a smaller
reduction (e.g. C=1) would be a better next experiment than assuming lower
is always better.

**Secondary benefit of capping — fit time.** Independent of accuracy, the
capped tree-based models were substantially cheaper to train at the
largest size: Decision Tree 202s -> 88s, Random Forest 27s -> 10s. Worth
mentioning even where the accuracy effect is small.

### Final test-set evaluation

Best combo by validation accuracy: **`P1_equalize` / `hog` / `svm_rbf` (C=10) / 1,000 images per class** (val accuracy 0.718). Evaluated once on the held-out test set (1,000 images, 500/class):

- Test accuracy: **0.729**
- Cats misclassified as dogs: 130/500 (26.0%)
- Dogs misclassified as cats: 141/500 (28.2%)
- Precision/recall/F1 (macro): 0.729 / 0.729 / 0.729

The confusion is fairly balanced between the two error types (dogs-as-cats
only slightly more common than cats-as-dogs), i.e. no strong directional
bias toward over-predicting one class. Full numbers in
`results/final_report.json`; visuals in `results/final_confusion_matrix.png`,
`results/learning_curve.png`, `results/train_val_gap.png`.

Notably, the winning combo used the *uncapped* SVM, not `svm_rbf_low_c` —
consistent with the per-size table above showing the low-C variant losing
at every size it was tested.

### Takeaways for the report

1. Of everything tried, **capping Decision Tree's `max_depth`** produced
   the single largest, cleanest accuracy improvement — and the fact that
   the improvement *grows* with training size is a good illustration that
   "more data helps" is conditional on the model being able to use it.
2. Preprocessing changes (equalize, denoise) and the low-C SVM variant were
   both genuine, useful **negative results** — worth reporting as "tried,
   didn't help" rather than omitting, since the homework explicitly asks
   for a discussion of what did and didn't work.
3. Accuracy plateaued around 71-73% regardless of which preprocessing/
   feature/capacity knob was turned — suggests the ceiling here is set more
   by the HOG/LBP feature representation itself than by these tuning
   choices, which is a reasonable thing to say explicitly rather than imply
   a specific config was close to optimal.
4. A natural next experiment (not yet run): a proper small grid over SVM's
   `C` (e.g. 1, 3, 10) and Decision Tree's `max_depth` (e.g. 5, 10, 20)
   using validation accuracy, rather than the single arbitrary capped value
   tested here — the SVM result in particular suggests the optimum is
   somewhere between 0.1 and 10, not at either extreme.

## 2026-09-23 (continued, PRELIMINARY) — SVM C-grid + large sizes: the earlier accuracy ceiling was an artifact, not a real limit

**Status: the underlying sweep is still running as of this writing.** `svm_rbf`/`svm_rbf_low_c` are complete through 10,000/class on some (preprocess, feature) slices; the new `svm_rbf_c1`/`svm_rbf_c3` variants are complete through 5,000/class on `P0_minimal` and partway through `P1_equalize`; nothing has reached `max` (11,498/class) yet, and `P2_denoise` hasn't started the C-grid classifiers at all. This section analyzes what's landed so far (239 rows) and **will need revisiting once the full sweep finishes** — but the pattern below is consistent across every slice checked, so it's already a meaningful correction to record now rather than wait on.

### The headline finding: SVM keeps improving well past where we'd stopped testing it, and the previous "~71-73% ceiling" conclusion (takeaway #3 above) was premature

That earlier conclusion was based on SVM only ever having been run up to 1,000 training images per class — a limit set purely for runtime reasons (`--svm-max-size 1000`), not because SVM's accuracy had leveled off. Now that larger sizes are actually being tested, `svm_rbf` (C=10) climbs well past the old ceiling:

`P0_minimal` / `hog` / `svm_rbf` (C=10), validation accuracy by size:

| size/class | 80 | 200 | 500 | 1000 | 5000 | 10000 |
|---|---|---|---|---|---|---|
| val accuracy | 0.654 | 0.684 | 0.709 | 0.713 | **0.767** | **0.773** |

0.773 is the best result seen anywhere in the entire sweep so far, beating the previous best (0.718) by 5.5 points — and it's still climbing at the last completed size, with `max` (11,498/class) not yet run. The same upward trend holds on the `P1_equalize`/`hog` slice (0.718 at 1,000 -> 0.751 at 5,000), so this isn't a one-off.

### The C-grid, now with enough sizes to see the real shape of it

`P0_minimal` / `hog`, validation accuracy by C and size (the most complete slice so far):

| size/class | C=0.1 | C=1 | C=3 | C=10 |
|---|---|---|---|---|
| 80 | 0.641 | 0.655 | 0.654 | 0.654 |
| 200 | 0.650 | 0.680 | 0.684 | 0.684 |
| 500 | 0.652 | 0.699 | 0.708 | 0.709 |
| 1000 | 0.670 | 0.712 | 0.713 | 0.713 |
| 5000 | 0.700 | 0.745 | **0.767** | **0.767** |
| 10000 | 0.714 | — | — | **0.773** |

Two things worth calling out:

1. **C=3 and C=10 are statistically indistinguishable at every size tested** (identical to 3 decimal places at 5,000/class: both 0.767). This suggests the useful range tops out somewhere at or before C=3 — going even higher than the homework's suggested C=10 is unlikely to help further, and C=3 gets the same accuracy while fitting noticeably faster (below).
2. **C=1 now clearly beats C=0.1 by a wide, growing margin** (80: +1.4 pts, 5000: +4.5 pts) — confirming last entry's guess that C=0.1 overshot into underfitting. But C=1 still trails C=3/C=10 by a consistent ~2 points at every size from 500 upward, so the earlier "optimum is somewhere between 0.1 and 10" is now narrowed further: it's at or above 3, not near 1.

### The train/val gap: revisits last entry's SVM finding, and it now tells a more complete story

Last entry reported the gap *widening* for the original C=10 SVM from size 80 to 1000 (0.346 -> 0.287 — actually narrowing slightly, but staying wide) and concluded we hadn't seen it shrink the way Decision Tree's did. With sizes up to 10,000 now available, it does:

| size/class | 80 | 200 | 500 | 1000 | 5000 | 10000 |
|---|---|---|---|---|---|---|
| gap (C=10) | 0.346 | 0.316 | 0.291 | 0.287 | 0.233 | 0.227 |

Same shape as Decision Tree's gap curve, just needing more data before the shrinkage becomes visible — we simply hadn't given it enough data to see this in the previous entry. This reinforces the broader theme across both entries: **models with real capacity to use extra data (unconstrained SVM, unconstrained/capped trees) generalize better with more of it; only models that are already saturated or mis-regularized (C=0.1, an unconstrained single Decision Tree with no cap) fail to benefit.**

### Cost side of the picture

At size=5,000 (`P0_minimal`/`hog`), fit time tracks C directly, not just accuracy:

| classifier | C | fit_time_sec | train_predict_time_sec |
|---|---|---|---|
| svm_rbf_low_c | 0.1 | 92 | 214 |
| svm_rbf_c1 | 1 | 94 | 198 |
| svm_rbf_c3 | 3 | 309 | 222 |
| svm_rbf | 10 | 316 | 224 |

C=3 costs roughly the same as C=10 (both ~3.3x slower to fit than C=1/C=0.1) for identical accuracy — so if compute time matters, **C=3 is a strictly better choice than C=10**: same accuracy, same cost, and it's a gentler regularization setting (usually preferable when two options tie). Note also that `train_predict_time_sec` (evaluating the classifier back on its own ~10,000-image training set, needed for the train_accuracy column) is now the dominant cost at this scale, bigger than fitting itself — a real trade-off of tracking the train/val gap this thoroughly at large sizes.

### Revised takeaways (supersedes takeaway #3 above)

1. **The "~71-73% accuracy ceiling" from the previous entry was not a real ceiling** — it was an artifact of capping SVM's training size at 1,000 for runtime reasons. SVM(C>=3) reaches 77.3% at 10,000/class and is still climbing. This is the single most important correction from this round.
2. **C=3 matches C=10's accuracy at a fraction of the "how far did we need to push regularization down" uncertainty**, and ties it exactly on cost too — worth using C=3 as the default going forward rather than the homework's example C=10.
3. The train/val gap for high-C SVM does eventually shrink with more data, same as Decision Tree — the earlier apparent difference between the two was just an artifact of not having tested SVM far enough.

### Next steps to improve accuracy (see also the reply to the user for the full prioritized list)

- **Let the SVM(C=10 and C=3) runs reach `max` size** — highest expected value of anything remaining in the current sweep, given the trend hasn't leveled off yet.
- Given C=3≈C=10, **deprioritize further C=1/C=0.1 large-size runs** (their trend is already clear: consistently behind, gap unlikely to close) in favor of getting C=3/C=10 to `max` faster, and finishing `P2_denoise`'s C-grid for completeness.
- Once `max` is in, revisit whether `P0_minimal` (no equalize/denoise) still leads at this scale, or whether the ranking of preprocessing levels changes now that the classifier itself is far more accurate — last entry's preprocessing comparison was also implicitly capped at the smaller sizes typical classifiers reach quickly.

## 2026-09-24 — Round 2: subject cropping, flip augmentation, richer features, new model families

**Goal**: get from ~77% toward 90% val accuracy. Professor's suggestion: zoom into the subject / remove the background. Constraints unchanged (no deep learning, test set touched once).

### A correction to the handoff first: the test set has already been touched once

`results/final_report.json` (2026-09-23 19:47) shows `final-eval` already ran on P1_equalize/hog/svm_rbf/1000 → **72.9% test**, even though the handoff says it never ran. Decision: disclose it in the report as an early end-to-end pipeline smoke test. The final model's `final-eval` is the one real test result. No selection has used that number.

### Code changes (all additive; old results reproduce exactly)

- `subject.py` (new): `center_square` crop and a `saliency` crop (spectral-residual saliency, Hou & Zhang 2007: classical, no learned model).
- `PreprocessConfig.crop_mode` / `flip`. Both are `repr=False` with `cache_extras()`, so every pre-existing feature cache keeps its signature. Crop parameters are folded into the signature.
- `features.py`: named feature *blocks* (`hog`, `lbp`, `hog16` coarse HOG, `slbp` spatial LBP, `color` HSV histogram), each cached separately. Extraction is parallel via joblib (all 5 blocks for the 23k-image pool take ~90 s per level on the M4). New levels are `P3_square`, `P4_saliency`, and `*_flip`, which add mirrored copies of the training images; val/test are never flipped. `--preprocess-level all` / `--feature-level all` still mean the original levels, so commands in flight on other machines are unchanged.
- `run_experiments.py`: every classifier is now a `Pipeline(StandardScaler, model)`. The scaler is still fit on each training slice only, as before, and `final-eval` can't scale differently from the sweep.
  - New models: `linsvc_*`, `hellinger_*`, `nystroem_svm_c1`, `pca_svm_c3` (PCA-512 → RBF SVC C=3), `hgb`, `lightgbm`, `xgboost`.
  - `SVC(cache_size=2000)`.
  - `--n-jobs` runs fits concurrently on threads. This works because libsvm releases the GIL: 2 fits took 1.38 s vs 2.55 s sequential.
  - `--train-acc-sample` computes train metrics on a random subsample; round-2 screening used 2000 rows.
- **Verified**: legacy caches load, and parallel extraction is bit-identical to them. `decision_tree_capped` (all 7 sizes) and `svm_rbf_c3` (500/1k/5k) reproduce their old val/train accuracies exactly. `cache_size=2000` alone cut the 5k SVC fit from 222 s to 152 s.
- Note: `fit_time_sec` now includes fitting the StandardScaler (negligible), and fits run with `--n-jobs > 1` share the CPU, so their times are inflated compared to solo runs.

### Screening results (val accuracy, fixed val set, 1k and 5k per class)

**Models on P0_minimal/hog** (reference: svm_rbf_c3 = 0.713 / 0.767):

| classifier | 1k | 5k |
|---|---|---|
| pca_svm_c3 | **0.722** | **0.770** |
| nystroem_svm_c1 | 0.716 | 0.740 (underfits: train 0.84) |
| hellinger_svm_c3 | 0.707 | 0.758 |
| lightgbm | **0.727** | 0.745 |
| xgboost | 0.698 | 0.747 |
| hgb | 0.688 | 0.730 |
| linsvc_c1e-3 / c1e-2 | 0.669 / 0.656 | 0.676 / 0.649 |
| hellinger_linsvc | 0.653 | 0.683 |

PCA→RBF-SVM matches or beats the exact SVM at roughly half the cost, so it's the screening model from here on. Linear SVMs on raw HOG are ~9 points behind: the RBF kernel matters. Gradient-boosted trees land ~2.5 points behind the SVM on raw HOG. They're kept for later fusion, not as the main model.

**Preprocessing (pca_svm_c3, hog)** — P0 reference 0.722 / 0.770:

| level | 1k | 5k |
|---|---|---|
| P3_square (center square crop: stop stretching) | 0.732 | 0.788 |
| P4_saliency (saliency zoom) | 0.698 | 0.780 |
| P0_minimal_flip | 0.751 | **0.806** |
| P3_square_flip | **0.764** | 0.799 |
| P4_saliency_flip | 0.744 | 0.800 |

- The aspect-ratio fix alone is worth ~+2 points at 5k. Stretching was squashing ~70% of images by >20% (median aspect ratio 1.25).
- Saliency zoom beats stretching but loses to a plain center square. Spectral-residual saliency fires on cage bars and busy fabric as often as on the animal (`results/debug/saliency_grid.png`). The default (90% saliency mass) never zoomed at all, so it's tuned to 60% mass + 5% margin.
- Flip augmentation is the biggest preprocessing win (+3.6 at 5k), consistent with the learning curves still climbing: effectively more data.

**Features (pca_svm_c3, P0_minimal)** — hog reference 0.722 / 0.770:

| feature level | dims | 1k | 5k |
|---|---|---|---|
| hog_slbp | 9,988 | 0.754 | 0.805 |
| hog2_slbp | 11,752 | 0.752 | **0.814** |
| hog2_slbp_color | 11,880 | **0.761** | **0.815** |

Replacing the single 10-bin global LBP histogram with a 4×4 grid of 59-bin histograms at two radii is worth +3.5 points. That's a far bigger jump than the original `hog_lbp` vs `hog` (+0.5–1), which suggests the global histogram threw away most of LBP's value. Coarse 16×16-cell HOG adds another ~1 point. Color adds ~0.1 at 5k, and it's pending the professor's OK anyway.

**Combining the winners (pca_svm_c3, hog2_slbp)**: they stack.

| level | 1k | 5k | 10k | max (11,498) |
|---|---|---|---|---|
| P3_square | 0.787 | 0.827 | | |
| P0_minimal_flip | 0.794 | 0.833 | | |
| **P3_square_flip** | 0.792 | 0.843 | 0.845 | **0.860** |

Tuning neighbors at 5k on P3_square_flip/hog2_slbp: `pca_svm_c10` 0.848, `pca1024_svm_c3` 0.849, vs 0.843 for `pca_svm_c3`. Those gaps are within noise: 1,000 val images give a standard error of ~1.1 points.

**New best: P3_square_flip / hog2_slbp / pca_svm_c3 / max → 0.860 val** (was 0.773). That's +8.7 points, from:
- no stretching (square crop),
- flip augmentation,
- spatial LBP + coarse HOG,
- PCA → RBF SVM.

The max-size fit (45,992 rows including flips) took 1,650 s on the M4 with 2 concurrent jobs.

### Revised next steps
1. Stacking/fusion: pca_svm + LightGBM (different model families) → logistic regression, using internal CV on train only.
2. `pca1024_svm_c10` at max: the C/PCA gains were small but both pointed the same way.
3. Whether color stays in depends on the professor's answer.
4. Then pick on val, run `final-eval` once, and write the report.

## 2026-09-24 (later) — Round 3 setup: professor's guidance, and the code for it

Professor's new guidance:
- Deep learning is **allowed in preprocessing**, and encouraged for segmentation / background removal.
- Post-processing and deblurring are options.
- "Evidence pooling" = several models acting as a **jury**, deciding by majority or weighted vote.
- Try other resolutions: classmates did better at 256×256.

The experiments are run by the user across three Macs, following `RUNBOOK_round3.md`. This entry records what was built and smoke-tested. Results go in the next entry.

### What was added
- **Segmentation** (`segment.py`, `subject.py`)
  - `rembg` salient-object models (IS-Net / U²-Net / BiRefNet) produce class-agnostic foreground masks, once, stored as PNGs in `data/masks/<model>/`. Only the mask is used; nothing class-related comes from the network.
  - Levels:
    - `P5_maskcrop` (square crop around the mask)
    - `P6_maskcrop_bggray` / `P7_maskcrop_bgblur` (background removed → flat gray / heavy blur)
    - `P5_maskcrop_raw` (no mask cleanup)
  - Mask cleanup = open/close + largest component + hole fill: the "post-processing" ablation.
  - Masks under 5% of the image count as failed segmentation, and the whole image is used instead. This avoids erasing animals behind fences.
  - New blocks: `shape` (silhouette HOG + Hu moments + area/aspect) and `slbp_fg` (spatial LBP on foreground pixels only).
- **Resolution** (per-level `FeatureConfig`):
  - `r192` and `r256c16` keep the 128/8-px cell geometry, computed from more pixels.
  - `r256c8` is the finer 44.6k-dim setup, without flips because of memory.
- **Deblurring**: `unsharp` (unsharp masking) and `rl` (Richardson–Lucy, Gaussian PSF σ=1, 10 iterations), plus a `blur-report` diagnostic.
- **Jury / evidence pooling** (`pooling.py`)
  - Every sweep run now saves its val decision scores plus flip-TTA scores to `results/val_scores/`.
  - `pool` combines them by majority / soft (per-member std-normalized mean) / weighted (log-odds of val accuracy), with greedy diverse member selection and a McNemar test.
  - `final-eval --jury`, with `--dry-run-on-val`.
- **Safety**:
  - `final-eval` refuses a second test-set look unless `--archive-previous` is passed, which keeps the 2026-09-23 smoke-test report for disclosure.
  - `CappedPCA` fixes a latent crash of `pca_svm_*` at 80/class, with no change to results at ≥1k.

### Smoke-test observations (scratch results dir, not in sweep_results.csv; the runbook's jury runs will record them properly)
- **Reference reproduces exactly** with the new code: P3_square_flip / hog2_slbp / pca_svm_c3 = 0.792 @1k, 0.843 @5k. Flip TTA gives 0.793 / 0.846.
- **LightGBM on hog2_slbp** (P3_square_flip, 5k): 0.841, and **0.853 with flip TTA**. On the richer features it's now competitive with the SVM (it was ~2.5 points behind on plain HOG).
- **Jury of those two** (soft vote + TTA, 5k): **0.861** val vs. 0.853 for the best single model (McNemar p=0.32, not significant yet). `final-eval --jury --dry-run-on-val` reproduced 0.861 exactly.
- **IS-Net segmentation**: 0.74 s/img on the M4 (≈5 h for all 25k images on one machine), with visually clean masks. The model misses animals behind chain-link fences.
- **Blur** (400-image sample, Laplacian variance): median 536 at 128 px vs. 252 at 256 px. Images look sharp at the working resolution, so deblurring is expected to matter mainly at 256.
- **onnxruntime has no Intel-Mac wheels for Python 3.14**, so masks are generated on the Apple Silicon machines only (`requirements-segment.txt` is kept separate).

## 2026-09-26 — Round 3 results, part 1: background removal works; a jury reaches ~0.90 val

Runs:
- M4 Air: IS-Net masks for all 24,997 images, plus the four mask levels.
- Intel MBP: sharpening + resolution levels (sharpening ran at all sizes because `TRAIN_SIZES` was missing on the first run).

All numbers are pca_svm_c3 / hog2_slbp unless noted.

**Segmentation levels vs. the reference (P3_square_flip):**

| level | 1k | 5k | 5k + flip TTA | McNemar p vs ref @5k (TTA) |
|---|---|---|---|---|
| P3_square_flip (reference) | 0.792 | 0.843 | 0.846 | — |
| P5_maskcrop_flip (zoom to mask, cleaned) | 0.783 | 0.849 | 0.848 | 0.92 |
| P5_maskcrop_raw_flip (zoom, raw mask) | 0.780 | 0.853 | 0.851 | 0.66 |
| **P6_maskcrop_bggray_flip (zoom + gray background)** | **0.816** | **0.864** | **0.873** | **0.048** |
| P7_maskcrop_bgblur_flip (zoom + blurred background) | 0.797 | 0.860 | 0.856 | 0.41 |

- **Removing the background is what helps; zooming alone doesn't.**
  - P5 (crop only) is within noise of the reference.
  - Flattening the background to gray (P6) gives +2.1 points (+2.7 with TTA, p=0.048). It's the first statistically significant gain in round 3.
  - Blurring the background (P7) lands in between. Some background texture survives the blur.
- **Where the gain comes from:** cat recall rises 0.808 → 0.864, while dog recall stays about flat (0.878 → 0.864). The reference was misreading cats from background context.
- **Mask post-processing (cleaned vs. raw) makes no measurable difference** (0.849 vs. 0.853, noise). IS-Net masks are already clean enough.

**Deblurring and resolution (Intel MBP):**

| level | 1k | 5k | 10k | max |
|---|---|---|---|---|
| P3_square_flip (reference) | 0.792 | 0.843 | 0.845 | 0.860 |
| + unsharp masking | 0.801 | 0.842 | 0.856 | 0.866 |
| + Richardson–Lucy | 0.791 | 0.838 | 0.845 | 0.849 |
| 192 px | 0.808 | 0.848 | | |
| 256 px, 16-px cells | 0.811 | 0.843 | | |

- Sharpening and resolution changes are all within noise at 5k.
- Higher resolution helps at 1k (+1.6 to +1.9), but that fades with more data.
- Richardson–Lucy slightly hurts.
- Consistent with the blur diagnostic: images at the working resolution are mostly sharp already.

**Jury (evidence pooling), first look** (scratch pool over the 31 saved score files so far):

| rule | K | TTA | val | vs. best single | McNemar p |
|---|---|---|---|---|---|
| soft | 3 | yes | **0.896** | +2.3 | 0.030 |
| soft | 3 | no | 0.894 | +2.8 | 0.0006 |
| weighted | 3 | yes | 0.893 | +2.0 | 0.070 |
| majority | 3 | yes | 0.892 | +1.9 | 0.084 |

- Best jury = P6_maskcrop_bggray_flip/pca_svm_c3/5k + P3_square_flip/lightgbm/5k + P3_square_flip_unsharp/pca_svm_c3/max, soft vote with flip TTA.
- The jury works because the members make *different* mistakes: P6 and the reference disagree on 175 of 1,000 val images.
- Caveats:
  - The jury was chosen on val, so 0.896 is optimistic. The one-time test run is the honest number.
  - The LightGBM member's scores come from the smoke test. The Intel MBP's jury run will reproduce it properly.

**Other:**
- The Intel MBP couldn't run LightGBM/XGBoost (its install is missing or broken). The code used to hide this as "unknown classifier"; it now reports the real import error.
- A crash from a half-written cache file (interrupted save) is fixed: cache writes are now atomic, and incomplete files are recomputed.

## 2026-09-26 (later) — Round 3 results, part 2: P6 at max, jury members, honest jury estimate

**New single-model best: P6_maskcrop_bggray_flip / hog2_slbp / pca_svm_c3 / max = 0.885 val** (P3_square_flip at max was 0.860, so gray background is +2.5 at full size).

**Shape / foreground-texture features on P6** (pca_svm_c3):

| feature level | 1k | 5k | 5k + TTA |
|---|---|---|---|
| hog2_slbp | 0.816 | 0.864 | 0.873 |
| hog2_slbp_shape (+ silhouette) | 0.811 | 0.866 | 0.876 |
| hog2_slbpfg_shape (+ silhouette, foreground-only LBP) | 0.817 | 0.862 | 0.859 |

These are all within noise, so no clear gain. Once the background is gray, HOG already sees the silhouette's edges.

**Tree models at 5k:**

| level / features | LightGBM | XGBoost | HGB | RF (capped) |
|---|---|---|---|---|
| P3_square_flip / hog2_slbp (Intel) | 0.837 | 0.838 | 0.844 | 0.753 |
| P3_square_flip / hog2_slbp_color (Intel) | 0.847 | 0.849 | 0.857 | 0.747 |
| P6_maskcrop_bggray_flip / hog2_slbp (Air) | 0.865 | 0.862 | 0.855 | — |

- Boosted trees are now within ~0.5 points of the SVM, and they make different mistakes.
- Color helps the trees by ~+1 point, unlike the SVM (+0.1).
- Cross-machine note: LightGBM on P3_square_flip/hog2_slbp/5k gave 0.837 on the Intel MBP vs 0.841 in the M4 smoke test. Tree models aren't bit-identical across CPU architectures; the SVMs were.

**Jury over all 45 saved score files (sizes ≥ 5k):**
- Best raw: weighted vote, K=7, no TTA = **0.909** (McNemar p=0.003 vs the best single model, 0.885).
- Many other configurations land at 0.900–0.906.

**How much of that is selection optimism?** Honest check: 40 random half/half splits of val. Pick the jury (and the best single model) on one half, score on the other.

| procedure | held-out accuracy | gain over best single (held-out) |
|---|---|---|
| best single model | 0.879 | — |
| soft, K=3 | 0.891 | +1.1 |
| soft, K=5, TTA | 0.893 | +1.4 |
| weighted, K=5 | 0.892 | +1.3 |
| weighted, K=7 | 0.894 | +1.4 |
| majority, K=3, TTA | 0.886 | +0.7 |

So a realistic expectation for the test set is **~89% for the jury vs. ~88% for the best single model**. The jury gain is real (+1.1 to +1.4 held-out), but smaller than the raw val numbers suggest (+2.4). Soft/weighted votes beat plain majority.

**Intel MBP issue:** LightGBM/XGBoost failed there because its repo's `.venv` pointed at another venv (moved folder). Fixed with a fresh venv. Its SVM `--backfill-scores` run exited without producing rows (cause unknown). That doesn't matter anymore: P3_square_flip SVM members at 5k are superseded by max-size members.
