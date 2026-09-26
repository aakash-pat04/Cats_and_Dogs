# Round 3 runbook: segmentation, resolution, deblurring, jury pooling

Two machines:
- **A** = M4 MacBook Air: segmentation, mask experiments, 256×256 fine cells, pooling, final evaluation
- **B** = Intel MacBook Pro: deblurring, resolution, jury members

Run every command from inside `Cats_and_Dogs/` with the venv active (`source .venv/bin/activate`). Everything is resumable: re-running the same command skips work that's already done.

**Reference to beat** (P3_square_flip / hog2_slbp / pca_svm_c3): **0.792 @1k, 0.843 @5k** (flip-TTA: 0.793 / 0.846).
- With 1,000 val images, ±1.1 points is noise.
- `pool` reports a McNemar p-value for jury vs. best single model.

**Golden rules**
- Never run the same (level, feature, classifier, size) on both machines. Both would write the same `results/val_scores/*.npz` and git would conflict. Stick to the assignments below.
- Use `./run_and_sync.sh` for sweeps. It pulls first, then commits and pushes chunk CSVs + val scores (even on Ctrl-C). `--train-sizes` goes in the `TRAIN_SIZES` env var.
- Don't run two heavy jobs on the same machine at once.
- **Don't run `final-eval` without `--dry-run-on-val` until step 6.** The code refuses a second test-set look unless you pass `--archive-previous`.
- `CATDOG_MASK_ROOT` must NOT be set on real runs; it's only for tests. Masks live in `../data/masks/isnet-general-use/`.

---

## Step 0: Sync the code (A) — DONE (commit 5bf24f8, merged into results-sync)

B (Intel): stop any old `svm_rbf`/`svm_rbf_c3` runs (Ctrl-C). Then:
```bash
git checkout results-sync && git pull origin results-sync
brew install libomp            # needed by LightGBM/XGBoost
pip install -r requirements.txt
```
B does **not** install `requirements-segment.txt`: onnxruntime has no Intel-Mac wheels for Python 3.14. B only reads finished masks, which needs nothing extra.

## Step 1: Segmentation masks (A only)

- Pilot: DONE.
- Shard 0/2: DONE (12,499 masks).
- Remaining: the other half, ~2.6 h. Run nothing else heavy on A meanwhile.
```bash
python segment.py run --model isnet-general-use --shard 1/2
python segment.py check --model isnet-general-use     # expect 24997/24997 present; note the fingerprint
python segment.py preview                             # look at results/debug/mask_levels_preview.png
```
Copy the masks to B (~550 MB). On A:
```bash
cd ../data && zip -rq ../masks_all.zip masks && cd ../Cats_and_Dogs
```
AirDrop `Cats and Dogs/masks_all.zip` to B. On B:
```bash
cd ../data && unzip -nq ~/Downloads/masks_all.zip && cd ../Cats_and_Dogs
python segment.py check --model isnet-general-use     # must match A's count AND fingerprint
```

## Step 2: B — deblurring + resolution (start now, no masks needed)

```bash
python run_experiments.py blur-report
TRAIN_SIZES=1000,5000 ./run_and_sync.sh --preprocess-level P3_square_flip_unsharp,P3_square_flip_rl,P3_square_flip_r192,P3_square_flip_r256c16 \
    --feature-level hog2_slbp --classifiers pca_svm_c3 --svm-max-size 999999 --n-jobs 2 --train-acc-sample 2000
```
Then B's jury members on the current best level. `--backfill-scores` re-runs round-2 combos that finished before val scores were saved.
```bash
TRAIN_SIZES=5000 ./run_and_sync.sh --preprocess-level P3_square_flip --feature-level hog2_slbp,hog2_slbp_color \
    --classifiers lightgbm,xgboost,hgb,random_forest_capped --n-jobs 1 --train-acc-sample 2000
TRAIN_SIZES=5000 ./run_and_sync.sh --preprocess-level P3_square_flip --feature-level hog2_slbp,hog2_slbp_color \
    --classifiers pca_svm_c3,pca1024_svm_c3,pca_svm_c10 --svm-max-size 999999 --n-jobs 2 --train-acc-sample 2000 --backfill-scores
```

## Step 3: A — mask experiments (after step 1's check passes)

```bash
TRAIN_SIZES=1000,5000 ./run_and_sync.sh --preprocess-level P5_maskcrop_flip,P6_maskcrop_bggray_flip,P7_maskcrop_bgblur_flip,P5_maskcrop_raw_flip \
    --feature-level hog2_slbp --classifiers pca_svm_c3 --svm-max-size 999999 --n-jobs 2 --train-acc-sample 2000
```
Then take the best of P5/P6/P7 (`BEST`) and add the silhouette/foreground-texture features:
```bash
TRAIN_SIZES=1000,5000 ./run_and_sync.sh --preprocess-level BEST --feature-level hog2_slbp_shape,hog2_slbpfg_shape \
    --classifiers pca_svm_c3 --svm-max-size 999999 --n-jobs 2 --train-acc-sample 2000
```
Then the 256×256 fine-cell run. It's memory-heavy: `--n-jobs 1`, nothing else running on A.
```bash
TRAIN_SIZES=1000,5000 ./run_and_sync.sh --preprocess-level P3_square_r256c8 --feature-level hog2_slbp \
    --classifiers pca_svm_c3 --svm-max-size 999999 --n-jobs 1 --train-acc-sample 2000
```
It has no flips (memory), so compare it with **P3_square**: 0.787 @1k / 0.827 @5k.

## Step 4: B — jury members on the best mask level (after B has the masks and A has named BEST)

```bash
TRAIN_SIZES=5000 ./run_and_sync.sh --preprocess-level BEST --feature-level hog2_slbp \
    --classifiers lightgbm,xgboost,hgb --n-jobs 1 --train-acc-sample 2000
```

## Step 5: A — pool

```bash
git pull origin results-sync && python run_experiments.py merge
python run_experiments.py pool --members auto --rules majority,soft,weighted --k 3,5,7
```
This writes `results/pooling_results.csv` and `results/jury.json`. Then:
1. Combine the winners into one level, and I'll add it.
2. Run the chosen members at `TRAIN_SIZES=max`: SVMs on A (~30 min each with flips), tree models on B.
3. Pool only the max-size members: `pool --members auto --min-size 11000`.

## Step 6: The one test-set evaluation (A)

```bash
python run_experiments.py final-eval --jury results/jury.json --dry-run-on-val   # val only: must reproduce pool's number
python run_experiments.py final-eval --jury results/jury.json --archive-previous # THE one real test look
```
- Use plain `final-eval --archive-previous` instead if a single model wins.
- The 2026-09-23 smoke-test report is kept as `final_report.previous_<time>.json` for the write-up's disclosure.

---

## What each experiment tests (for the report)

| Idea (professor) | Levels / features | Compare against |
|---|---|---|
| Segmentation: zoom to the animal | `P5_maskcrop_flip` | P3_square_flip |
| Background removal | `P6_maskcrop_bggray_flip` (flat gray), `P7_maskcrop_bgblur_flip` (blurred) | P5 |
| Post-processing (mask cleanup) | `P5_maskcrop_flip` (cleaned) vs `P5_maskcrop_raw_flip` (raw) | each other |
| Shape / foreground-only texture | `hog2_slbp_shape`, `hog2_slbpfg_shape` on the best mask level | hog2_slbp on the same level |
| Deblurring | `P3_square_flip_unsharp`, `P3_square_flip_rl` | P3_square_flip |
| Resolution (same geometry) | `P3_square_flip_r192`, `P3_square_flip_r256c16` | P3_square_flip (128) |
| Resolution (finer cells, 256×256) | `P3_square_r256c8` (no flips: memory) | P3_square (128, no flips) |
| Evidence pooling (jury) | `pool` over all saved val scores | best single model (McNemar) |
| Flip test-time augmentation | `val_acc_flipTTA` in logs / `tta=True` in `pool` | the same model without TTA |
