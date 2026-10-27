#!/usr/bin/env bash
# Interleave the unshuffled taujet prefix of each UPP train split, in place,
# one file per array task (see fix_tau_blocks.py for why). Idempotent: files
# without a pure-tau prefix exit immediately.
#
# NOT always needed: the July production had ~64,000-row pure-tau prefixes
# because countup ran out of taujets, but the 261013 production has none
# (longest prefix 72 rows, 0 of 28,000 500-row windows single-class) -- explicit
# per-class num_jets instead of -1 keeps any component from running dry. CHECK
# before running it, rather than mutating 148 GB in place for nothing. env.sh
# derives FCC_PP_OUT from the UPP config's out_dir; override to target another
# production.
#
#   cd fcc-ftag-salt-converter
#   N=$(ls $FCC_PP_OUT/train/pp_output_train_split_*.h5 | wc -l)
#   sbatch --array=1-$N scripts/submit_fix_tau_blocks.sbatch
#
#SBATCH --job-name=fix-tau-blocks
#SBATCH --partition=private-dpnc-cpu,shared-cpu
# measured: minutes/file, block-wise reads ~50 MB
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/taublocks_%A_%a.out
set -euo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

D=${FCC_PP_OUT}

mapfile -t FILES < <(ls "$D"/train/pp_output_train_split_*.h5)
f="${FILES[$((SLURM_ARRAY_TASK_ID - 1))]}"
fc=$(fcc_in_container "$f")
echo "[$(date)] interleaving tau prefix in $f"

apptainer exec ${FCC_BINDS} \
    --env PYTHONNOUSERSITE=1 \
    "${FCC_IMAGE}" python -u /repo/scripts/fix_tau_blocks.py "$fc"
