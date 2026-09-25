# Round 3 runbook: segmentation, resolution, deblurring, jury pooling

Machines:
- **A** = M4 MacBook Air
- **B** = Intel MacBook Pro
- **C** = second Apple Silicon Mac

Run every command from inside `Cats_and_Dogs/` with the venv active (`source .venv/bin/activate`). Everything is resumable: re-running the same command skips work that's already done.

**Reference to beat** (P3_square_flip / hog2_slbp / pca_svm_c3): **0.792 @1k, 0.843 @5k** (flip-TTA: 0.793 / 0.846).
- With 1,000 val images, ±1.1 points is noise.
- `pool` reports a McNemar p-value for jury vs. best single model.

**Golden rules**
- Never run the same (level, feature, classifier, size) on two machines. Both would write the same `results/val_scores/*.npz` and git would conflict. Stick to the assignments below.
- Use `./run_and_sync.sh` for sweeps. It pulls first, then commits and pushes chunk CSVs + val scores (even on Ctrl-C). `--train-sizes` goes in the `TRAIN_SIZES` env var.
- **Don't run `final-eval` without `--dry-run-on-val` until step 5.** The test set gets exactly one real look. The code now refuses a second look unless you pass `--archive-previous`.
- `CATDOG_MASK_ROOT` must NOT be set on real runs. It's only for tests. Masks live in `../data/masks/<model>/`.

---

## Step 0: Sync the code (A, ~10 min)

All round-2/3 code is uncommitted on `results-sync` (the code is identical on both branches, so the changes carry over):
```bash
git checkout main
git add subject.py segment.py pooling.py features.py preprocess.py run_experiments.py reporting.py \
        run_and_sync.sh .gitignore requirements.txt requirements-segment.txt experiment_log.md RUNBOOK_round3.md
git commit -m "Round 2+3: square crop, flips, feature blocks, new models, segmentation masks, resolution, sharpening, jury pooling"
git push origin main
git checkout results-sync
git merge main
git add results/chunks/*.csv
git commit -m "Round 2 screening results (M4 Air)"
git push origin results-sync
```

**B** (Intel): stop the old `svm_rbf`/`svm_rbf_c3` large-size runs first (Ctrl-C; the sync script still pushes what finished). Then:
```bash
git checkout results-sync && git pull origin results-sync
brew install libomp            # needed by LightGBM/XGBoost
pip install -r requirements.txt
```
B does **not** install `requirements-segment.txt`. onnxruntime has no Intel-Mac wheels for Python 3.14, and B only reads finished masks, which needs nothing extra.

**C** (fresh setup): follow `../Cats and Dogs Dataset Setup.rtf` steps 1–3 (clone, check out `results-sync`, unzip `PetImages.zip` as a sibling folder `data/`, create the venv). Then:
```bash
pip install -r requirements.txt -r requirements-segment.txt   # brew install libomp first if pip/LightGBM complains
```
A also needs the segmentation packages (they may already be installed there):
```bash
pip install -r requirements-segment.txt
```

---

## Step 1: Segmentation masks (A and C, ~3 h each, can run overnight)

**1a. Pilot on A (~10 min, downloads ~180 MB of weights per model once).**
```bash
python segment.py pilot --models u2net,isnet-general-use,birefnet-general-lite --n 200
```
- It prints seconds per image and projected hours, and writes `results/debug/seg_pilot_<model>.png`.
- My smoke test measured isnet-general-use at **0.74 s/img on the M4**, which is about 2.6 h per machine with 2 machines. Its masks are clean. It misses animals behind fences/cage bars, but those automatically fall back to the whole image.
- Pick the model with the best-looking masks at an acceptable speed. The default is `isnet-general-use`.

**1b. Generate. Use the same model on both machines.**
```bash
export CATDOG_MASK_MODEL=isnet-general-use      # or the pilot winner — set it in EVERY terminal that runs mask levels, on every machine
python segment.py run --model $CATDOG_MASK_MODEL --shard 0/2      # on A
python segment.py run --model $CATDOG_MASK_MODEL --shard 1/2      # on C
```
If a run gets interrupted, re-run the same command: masks already written are skipped.

**1c. Exchange the masks (~275 MB per shard).** On each of A and C:
```bash
cd ../data && zip -rq ../masks_$(hostname -s).zip masks && cd ../Cats_and_Dogs
```
Copy each zip to the other two machines (AirDrop or cloud drive). On every receiving machine (A, B, C):
```bash
cd ../data && unzip -nq ~/Downloads/masks_<other-host>.zip && cd ../Cats_and_Dogs
```

**1d. Verify on all three machines.** The counts and fingerprint must match everywhere:
```bash
python segment.py check --model $CATDOG_MASK_MODEL
```
Expected: `24997/24997 masks present`, and the same fingerprint on A, B and C.

**1e. Look before sweeping (A).**
```bash
python segment.py preview
```
Open `results/debug/mask_levels_preview.png`. It shows what P3 (current best), the P5 mask crop, P5 raw, P6 gray background, and P7 blurred background actually feed the features.

---

## Step 2: Screening sweeps (in parallel)

Set these in each terminal first:
```bash
export CATDOG_MASK_MODEL=isnet-general-use       # same as step 1
export TRAIN_SIZES=1000,5000
```

**B (Intel) — deblurring + resolution. Can start right after step 0, no masks needed.**
```bash
python run_experiments.py blur-report            # diagnostic -> results/debug/blur_hist.png, blur_examples.png
./run_and_sync.sh --preprocess-level P3_square_flip_unsharp,P3_square_flip_rl,P3_square_flip_r192,P3_square_flip_r256c16 \
    --feature-level hog2_slbp --classifiers pca_svm_c3 --svm-max-size 999999 --n-jobs 2 --train-acc-sample 2000
```
Feature extraction is slower on Intel, and 256-px levels take ~4× longer than 128-px. Expect a few hours in total.

**A (M4) — segmentation levels. Needs all masks (step 1d).**
```bash
./run_and_sync.sh --preprocess-level P5_maskcrop_flip,P6_maskcrop_bggray_flip,P7_maskcrop_bgblur_flip,P5_maskcrop_raw_flip \
    --feature-level hog2_slbp --classifiers pca_svm_c3 --svm-max-size 999999 --n-jobs 2 --train-acc-sample 2000
```
Then take the best of P5/P6/P7 (call it `<BEST_MASK>`) and add the silhouette/foreground-texture features:
```bash
./run_and_sync.sh --preprocess-level <BEST_MASK> --feature-level hog2_slbp_shape,hog2_slbpfg_shape \
    --classifiers pca_svm_c3 --svm-max-size 999999 --n-jobs 2 --train-acc-sample 2000
```

**C — fine-cell 256×256 (the classmates' setup). Memory-heavy: `--n-jobs 1`, and nothing else heavy running.**
```bash
./run_and_sync.sh --preprocess-level P3_square_r256c8 --feature-level hog2_slbp \
    --classifiers pca_svm_c3 --svm-max-size 999999 --n-jobs 1 --train-acc-sample 2000
```
This level has no flips (memory), so compare it with **P3_square** (no flip): 0.787 @1k / 0.827 @5k.

**C, afterwards — jury members on the current best level (diversity for pooling).**
`--backfill-scores` re-runs round-2 combos that finished before val scores existed.
```bash
export TRAIN_SIZES=5000
./run_and_sync.sh --preprocess-level P3_square_flip --feature-level hog2_slbp,hog2_slbp_color \
    --classifiers lightgbm,xgboost,hgb,random_forest_capped --n-jobs 1 --train-acc-sample 2000
./run_and_sync.sh --preprocess-level P3_square_flip --feature-level hog2_slbp,hog2_slbp_color \
    --classifiers pca_svm_c3,pca1024_svm_c3,pca_svm_c10 --svm-max-size 999999 --n-jobs 2 --train-acc-sample 2000 --backfill-scores
```

**B, afterwards (once masks are on B and A has named `<BEST_MASK>`) — jury members on the best mask level.**
```bash
export TRAIN_SIZES=5000
./run_and_sync.sh --preprocess-level <BEST_MASK> --feature-level hog2_slbp \
    --classifiers lightgbm,xgboost,hgb --n-jobs 1 --train-acc-sample 2000
```
(`<BEST_MASK>` / hog2_slbp / pca_svm_c3 @5k is already covered by A's run above.)

**Reading results anywhere:**
```bash
git pull origin results-sync && python run_experiments.py merge
python -c "import pandas as pd; d=pd.read_csv('results/sweep_results.csv'); print(d.sort_values('val_accuracy',ascending=False)[['preprocess_level','feature_level','classifier','train_size_key','val_accuracy']].head(20).to_string(index=False))"
```
The sweep log lines also show `val_acc_flipTTA` for every run.

---

## Step 3: Pool (A, a few seconds)

```bash
git pull origin results-sync && python run_experiments.py merge
python run_experiments.py pool --members auto --rules majority,soft,weighted --k 3,5,7
```
- It prints the best juries with their McNemar p-value vs. the best single model, and writes `results/pooling_results.csv` and `results/jury.json` (the best one).
- Only runs that saved val scores (round 3 onward, or backfilled) can be members.
- To test a specific jury: `--members <stem1>,<stem2>,...`, using file names from `results/val_scores/` without `.npz`.

---

## Step 4: Combine winners and scale to max size (day 3; split across machines)

1. Tell me the step 2/3 results. If more than one of mask / resolution / sharpening helps, I'll add a combined level for them.
2. Run the chosen members at full size: `TRAIN_SIZES=max ./run_and_sync.sh ...` (SVMs on A, tree models on B/C).
   - A max-size SVM with flips took ~28 min on the M4.
3. Pool only the max-size members:
   ```bash
   python run_experiments.py pool --members auto --min-size 11000
   ```
   Decide: best single model vs. best jury (report both).

---

## Step 5: The one test-set evaluation (A, overnight)

```bash
python run_experiments.py final-eval --jury results/jury.json --dry-run-on-val   # val only: must reproduce pool's val accuracy
python run_experiments.py final-eval --jury results/jury.json --archive-previous # THE one real test look
```
- Use plain `final-eval --archive-previous` instead if a single model wins.
- `--archive-previous` keeps the 2026-09-23 smoke-test report as `final_report.previous_<time>.json` for the write-up's disclosure.
- Outputs: `results/final_report.json`, `results/final_confusion_matrix.png`.

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
| Flip test-time augmentation | `val_acc_flipTTA` column in logs / `tta=True` in `pool` | the same model without TTA |
