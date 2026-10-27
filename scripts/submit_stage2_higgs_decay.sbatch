#!/usr/bin/env bash
# SLURM array runner: one task = one stage1 file from $MANIFEST, flattened to a
# per-jet stage2 ntuple. Submit via scripts/submit_stage2_higgs_decay.sh (which
# builds the manifest and sizes the --array). Do NOT sbatch this directly.
#SBATCH --job-name=stage2-higgs
#SBATCH --partition=private-dpnc-cpu
# measured on the 260918 production: ~1.5 min and ~2.5 GB RSS for a full
# 100k-event file. stage2 holds the whole event range in memory at once.
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/stage2_%A_%a.out
#SBATCH --error=logs/stage2_%A_%a.err
# not -u: the key4hep setup.sh reads unset variables of its own.
set -eo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

# The truth-labelling FCCAnalyses checkout (HEP-FCC/FCCAnalyses#549-554), whose
# pinned key4hep stack setup.sh -p brings in.
export FCCANALYSES_DIR="${FCCANALYSES_DIR:-$HOME/flav_tag/fcc/FCCAnalyses}"

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
IFS='|' read -r decay stage1 tag <<< "$line"
out="${OUTDIR}/stage2_${decay}_${tag}.root"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${decay} ${tag}"
echo "  input: ${stage1}"

# Idempotent re-runs of a partial array.
if [[ -s "$out" ]]; then
    echo "  already done, skipping."
    exit 0
fi

cd "$FCCANALYSES_DIR"
source ./setup.sh -p

# stage2 reads the jet flavour out of the input PATH, which is why the sample
# directory name has to stay in it. It writes every branch as float32.
python3 examples/FCCee/weaver/stage2.py "$stage1" "$out" 0 100000000
echo "[$(date)] done: ${out}"
