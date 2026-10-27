#!/bin/bash
set -e

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

CKPT="$1"
CKPT_DIR=$(dirname "$CKPT")
CKPT_STEM=$(basename "$CKPT" .ckpt)

# Find the predictions file (PredictionWriter suffix __test_*.h5).
PREDS=$(ls -t "$CKPT_DIR/${CKPT_STEM}__test"*.h5 2>/dev/null | head -1)
[[ -n "$PREDS" && -f "$PREDS" ]] || { echo "ERROR: no predictions file under $CKPT_DIR" >&2; exit 2; }
echo "Plotting from predictions: $PREDS"

# Plots go to <run_dir>/plots/
RUN_DIR=$(dirname "$CKPT_DIR")
OUT_DIR="$RUN_DIR/plots"
mkdir -p "$OUT_DIR"

export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg

# The auxiliary-head plots (track origin/type, b/c charge, vertexing) need the
# task set, which salt writes as the resolved config at the top of the run dir.
MODEL_CONFIG="$RUN_DIR/config.yaml"
CFG_ARG=()
if [[ -f "$MODEL_CONFIG" ]]; then
    CFG_ARG=(--model-config "$MODEL_CONFIG")
else
    echo "WARNING: no $MODEL_CONFIG -- auxiliary-head plots will be skipped" >&2
fi

python "${FCC_PUMA_SCRIPT}" \
    --preds "$PREDS" \
    --out   "$OUT_DIR" \
    "${CFG_ARG[@]}"
echo "Plots in: $OUT_DIR"
