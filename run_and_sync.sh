#!/usr/bin/env bash
# Run a sweep chunk and sync the resulting chunk CSV(s) to GitHub via the
# results-sync branch — for running an assigned chunk on a second/third
# machine without having to remember every git step by hand.
#
# Usage (run from inside Cats_and_Dogs/, on a clone checked out to
# results-sync, with data/PetImages present alongside it):
#   ./run_and_sync.sh --preprocess-level P1_equalize --feature-level all \
#       --classifiers svm_rbf,svm_rbf_c3 --svm-max-size 999999
#
# Any arguments given are passed straight through to `run_experiments.py
# sweep`. Syncs whatever chunk progress exists even if interrupted
# (Ctrl-C) partway — nothing completed is lost.
#
# --train-sizes is a top-level option of run_experiments.py (it must come
# before `sweep`), so pass it through the TRAIN_SIZES environment variable:
#   TRAIN_SIZES=1000,5000 ./run_and_sync.sh --preprocess-level ... 

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ORIGINAL_ARGS="$*"

sync_results() {
    echo
    echo "==> Syncing chunk results to results-sync..."
    git add results/chunks/*.csv 2>/dev/null
    # Saved val scores feed evidence pooling (`run_experiments.py pool`), which
    # needs every machine's scores in one place. Separate `git add` so an
    # empty glob here can't block adding the chunk CSVs above.
    git add results/val_scores/*.npz 2>/dev/null
    if git diff --cached --quiet; then
        echo "No new/changed chunk files to commit."
        return
    fi
    git commit -m "Add sweep results from $(hostname) [${ORIGINAL_ARGS}]"
    if git push origin results-sync; then
        echo "==> Pushed to results-sync."
    else
        echo "==> Push failed (network? someone else pushed first?). Chunk files are"
        echo "    committed locally — retry with:"
        echo "    git pull --rebase origin results-sync && git push origin results-sync"
    fi
}
trap sync_results EXIT

if [ -d ".venv" ]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

echo "==> Pulling latest results-sync before starting (avoids redoing already-completed combos)..."
git pull origin results-sync

TOP_LEVEL_ARGS=()
if [ -n "${TRAIN_SIZES:-}" ]; then
    TOP_LEVEL_ARGS+=(--train-sizes "$TRAIN_SIZES")
    ORIGINAL_ARGS="TRAIN_SIZES=${TRAIN_SIZES} ${ORIGINAL_ARGS}"
fi

echo "==> Running: python run_experiments.py ${TOP_LEVEL_ARGS[*]:-} sweep ${ORIGINAL_ARGS}"
python run_experiments.py ${TOP_LEVEL_ARGS[@]+"${TOP_LEVEL_ARGS[@]}"} sweep "$@"
