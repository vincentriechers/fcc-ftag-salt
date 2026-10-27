#!/bin/bash
set -e

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

CKPT="$1"
[[ -f "$CKPT" ]] || { echo "ERROR: ckpt not found: $CKPT" >&2; exit 2; }

# As during training: block the user site-packages.
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Comet is not needed for testing, but salt may still initialise it -- sourcing
# the key is harmless. Same override as salt_commands.sh.
COMET_KEY_FILE="${COMET_KEY_FILE:-${HOME}/flav_tag/submission_scripts/scripts/API_Keys.sh}"
[[ -f "${COMET_KEY_FILE}" ]] && source "${COMET_KEY_FILE}" || true

salt test \
    --config "${FCC_SALT_CONFIG}" \
    --ckpt_path "$CKPT" \
    --force

# PredictionWriter writes to <ckpt_dir>/<ckpt_name>__test_*.h5
CKPT_DIR=$(dirname "$CKPT")
CKPT_STEM=$(basename "$CKPT" .ckpt)
echo "--- predictions written: ---"
ls -la "$CKPT_DIR/${CKPT_STEM}__test"* 2>&1 || echo "no __test_*.h5 produced"
