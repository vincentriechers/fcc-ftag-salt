#!/usr/bin/env bash
# Render normalised step-histogram comparisons between the full 2.7M-event
# FCC sample (parts 1+2) and the ATLAS reference H5. One dataset per array
# task so the slow tracks_ghost run does not block the others.
#
# Submit:
#   sbatch scripts/submit_plot_full.sbatch
#
# Output:
#   ${FCC_FTAG_DATA}/plots_full/*.png

#SBATCH --job-name=fcc-ftag-plots
#SBATCH --array=1-5
#SBATCH --partition=shared-cpu
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/plot_%a.%j.out
#SBATCH --error=logs/plot_%a.%j.err

set -euo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

REPO=${FCC_FTAG_REPO}
mkdir -p "$REPO/logs"
cd "$REPO"

# key4hep env
set +u
source /cvmfs/sw.hsf.org/key4hep/setup.sh > /dev/null
set -u
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

case "$SLURM_ARRAY_TASK_ID" in
    1) DS=jets ;;
    2) DS=tracks_ghost ;;
    3) DS=flows ;;
    4) DS=electrons ;;
    5) DS=muons ;;
    *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac

echo "[$(date)] task ${SLURM_ARRAY_TASK_ID} → dataset=$DS on $(hostname)"
time python scripts/plot_full_vs_ref.py --dataset "$DS"
echo "[$(date)] done."
