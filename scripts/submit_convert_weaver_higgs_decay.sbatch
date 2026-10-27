#!/usr/bin/env bash
# Convert OUR weaver stage2 chunks -> Salt-native H5, ONE decay per array task
# (the 7 decays are independent, so this runs them in parallel; wall time =
# the longest single decay instead of the sum of all seven).
#
# Reads LOCAL beegfs stage2_<decay>_*.root (from submit_weaver_higgs_decay.sh
# or submit_stage2_higgs_decay.sh) and writes fcc_zh_<decay>.h5 into the UPP
# ntuple dir. Runs in the normal converter env (ambient python with
# uproot/h5py, inherited via --export=ALL), NOT the UPP container.
#
# Submit (defaults shown):
#   sbatch scripts/submit_convert_weaver_higgs_decay.sbatch \
#       ${FCC_FTAG_DATA}/higgs_decay/weaver_stage2 \
#       ${FCC_FTAG_DATA}/higgs_decay
#
#SBATCH --job-name=convert-zh
#SBATCH --partition=private-dpnc-cpu
# measured on the 260918 production (20M jets/decay, 139M total): 44 min
# (Htautau) to 67 min (Hgg), peak RSS 2.7 GB. The old 2 h limit was set when a
# decay was 10M jets and would have left Hgg very little margin at this size.
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-6
#SBATCH --output=logs/convert_zh_%A_%a.out
#SBATCH --error=logs/convert_zh_%A_%a.err
set -euo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

REPO=${FCC_FTAG_REPO}
STAGE2_DIR="${1:-${FCC_FTAG_DATA}/higgs_decay/weaver_stage2}"
OUTDIR="${2:-${FCC_FTAG_DATA}/higgs_decay}"

DECAYS=(Hbb Hcc Hdd Hss Huu Hgg Htautau)
decay="${DECAYS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$REPO/logs" "$OUTDIR"
cd "$REPO"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"

shopt -s nullglob
inputs=("$STAGE2_DIR"/stage2_"${decay}"_*.root)
if [[ "${#inputs[@]}" -eq 0 ]]; then
    echo "ERROR: no stage2 chunks for ${decay} in ${STAGE2_DIR}" >&2
    exit 1
fi

echo "[$(date)] ${decay}: ${#inputs[@]} chunk(s) -> ${OUTDIR}/fcc_zh_${decay}.h5"
srun python -m fcc_ftag_converter.fcc_higgs_decay "${inputs[@]}" \
    -o "${OUTDIR}/fcc_zh_${decay}.h5" --chunk-size 50000
echo "[$(date)] ${decay} done."
