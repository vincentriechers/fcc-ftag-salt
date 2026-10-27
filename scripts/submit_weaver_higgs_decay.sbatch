#!/usr/bin/env bash
# SLURM array runner: one task = one (decay, Delphes file) from $MANIFEST, run
# through scripts/weaver_higgs_decay.sh. Submit via scripts/submit_weaver_higgs_decay.sh
# (which builds the manifest and sizes the --array). Do NOT sbatch this directly.
#SBATCH --job-name=weaver-higgs
#SBATCH --partition=private-dpnc-cpu
# measured: <3 min/file (4 cpus), ~3 GB RSS
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=logs/weaver_%A_%a.out
#SBATCH --error=logs/weaver_%A_%a.err
set -euo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

REPO=${FCC_FTAG_REPO}
# truth-origin-vertex-524 worktree: PR #524 base (fixed features) + truth branches.
export FCCANALYSES_DIR="${FCCANALYSES_DIR:-$HOME/FCCAnalyses-truth524}"
export NCPUS="${SLURM_CPUS_PER_TASK:-8}"

: "${MANIFEST:?MANIFEST not set (submit via submit_weaver_higgs_decay.sh)}"
: "${OUTDIR:?OUTDIR not set}"

# Per-task scratch for the (large) intermediate stage1 file. /tmp on compute
# nodes is often too small, so stage on beegfs.
export TMPDIR="${FCC_FTAG_DATA}/higgs_decay/_wtmp_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

line=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$MANIFEST")
IFS='|' read -r decay url tag <<< "$line"
echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} on $(hostname): ${decay} tag=${tag}"
echo "  input: ${url}"

# Skip if the chunk already exists (idempotent re-runs of a partial array).
if [[ -s "${OUTDIR}/stage2_${decay}_${tag}.root" ]]; then
    echo "  already done, skipping."
    exit 0
fi

bash "$REPO/scripts/weaver_higgs_decay.sh" "$decay" "$url" "$OUTDIR" "$tag"
