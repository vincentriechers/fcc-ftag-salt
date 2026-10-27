#!/usr/bin/env bash
# Convert the full FCC Winter2023 IDEA p8_ee_tt_ecm365 sample (27 ROOT files,
# ~2.7M events) into Salt-shaped H5. Splits into 2 parts of ~8.4 GB each so
# the per-file 10 GB limit is respected.
#
# Submit:
#   sbatch scripts/submit_full_convert.sbatch
#
# Output:
#   ${FCC_FTAG_DATA}/fcc_tt_365_part{1,2}.h5
#   ${FCC_FTAG_DATA}/fcc_tt_365_part{1,2}.summary.{md,json}

#SBATCH --job-name=fcc-ftag-convert
#SBATCH --array=1-3
#SBATCH --partition=shared-cpu
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/convert_part%a.%j.out
#SBATCH --error=logs/convert_part%a.%j.err

set -euo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

REPO=${FCC_FTAG_REPO}
OUT_DIR=${FCC_FTAG_DATA}
EOS_PATH=/eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_tt_ecm365

mkdir -p "$REPO/logs"
cd "$REPO"

echo "[$(date)] Part ${SLURM_ARRAY_TASK_ID} on $(hostname), $SLURM_CPUS_PER_TASK CPUs"

# ---- env ------------------------------------------------------------------
# key4hep references positional args without defaults; tolerate under set -u
set +u
source /cvmfs/sw.hsf.org/key4hep/setup.sh > /dev/null
set -u
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
# Use shared scratch for staging (/tmp on compute nodes is often <16 GB and
# we need ~24 GB for 14 ROOT files of ~1.7 GB each). /srv/beegfs/scratch is
# fast enough for sequential reads. TMPDIR is also where the per-worker
# intermediate H5s live before the merge step.
export TMPDIR="${FCC_FTAG_DATA}/_stage_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# ---- enumerate inputs -----------------------------------------------------
mapfile -t EOS_PATHS < <(xrdfs root://eospublic.cern.ch ls "$EOS_PATH" \
    | grep -v 'sys\.v')
echo "[input] discovered ${#EOS_PATHS[@]} ROOT files on EOS"
test ${#EOS_PATHS[@]} -eq 27 || { echo "expected 27 inputs, got ${#EOS_PATHS[@]}" >&2; exit 1; }

# ---- pick this task's slice ------------------------------------------------
# 27 files split into 3 parts of 9 each → ~6.4 GB per output H5 (safe margin
# under the 10 GB per-file budget after slot-size bump 50→75/100).
case "$SLURM_ARRAY_TASK_ID" in
    1) SLICE=("${EOS_PATHS[@]:0:9}");  N_WORKERS=9 ;;
    2) SLICE=("${EOS_PATHS[@]:9:9}");  N_WORKERS=9 ;;
    3) SLICE=("${EOS_PATHS[@]:18}");   N_WORKERS=9 ;;
    *) echo "unknown array task $SLURM_ARRAY_TASK_ID" >&2; exit 1 ;;
esac
OUT="$OUT_DIR/fcc_tt_365_part${SLURM_ARRAY_TASK_ID}.h5"

# ---- stage inputs locally (avoids long-running HTTP streams that flake) ----
# Earlier attempts streamed via uproot+HTTPS and 3/14 workers died on transient
# socket errors. Pre-staging with retry-able xrdcp is much more reliable.
STAGE="$TMPDIR/inputs"
mkdir -p "$STAGE"
LOCAL_FILES=()
for src in "${SLICE[@]}"; do
    name=$(basename "$src")
    dst="$STAGE/$name"
    LOCAL_FILES+=("$dst")
    if [[ -s "$dst" ]]; then echo "[stage] already have $name"; continue; fi
    echo "[stage] xrdcp $name"
    for try in 1 2 3 4 5; do
        if xrdcp -f -s "root://eospublic.cern.ch/$src" "$dst"; then
            break
        fi
        echo "[stage] retry $try for $name" >&2
        sleep $((10 * try))
    done
    test -s "$dst" || { echo "[stage] FAILED to download $src" >&2; exit 2; }
done
echo "[stage] ${#LOCAL_FILES[@]} files in $STAGE"
df -h "$TMPDIR" 2>&1 | tail -2

echo "[run] ${#LOCAL_FILES[@]} local files → $OUT with $N_WORKERS workers"

# ---- convert --------------------------------------------------------------
time python -m fcc_ftag_converter.cli "${LOCAL_FILES[@]}" \
    --output "$OUT" \
    --workers "$N_WORKERS" \
    --chunk-size 5000

echo "[$(date)] Part ${SLURM_ARRAY_TASK_ID} done."
ls -la "$OUT" "${OUT%.h5}.summary.md" || true
