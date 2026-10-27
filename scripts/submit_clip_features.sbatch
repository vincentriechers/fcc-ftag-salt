#!/usr/bin/env bash
# Clip pfcand input features to mean +- 5*std, in place, one preprocessed UPP
# H5 per array task (see clip_pfcand_features.py for why). Idempotent.
#
# NOT part of the default chain any more: the salt configs bound outliers at
# training time with `norm_config.clamp`, which is equivalent and mutates
# nothing. Use this only if you need the stored values clipped.
#
# ALWAYS THROTTLE (`%8`). Unthrottled, slurm packed all 39 tasks onto one node,
# none finished inside the walltime, and every one of the 39 files was left
# corrupt ("filter returned failure during read") -- an interrupted in-place
# rewrite of a blosc-compressed compound dataset is unrecoverable. Recovery is
# `submit_upp.sh merge all` + `normalise` from preprocessed/components/.
#
# Submit (env.sh derives FCC_PP_OUT from the UPP config's out_dir; override it
# to clip a different production):
#   cd fcc-ftag-salt-converter
#   N=$(ls $FCC_PP_OUT/{train,val,test}/pp_output_*_split_*.h5 | wc -l)
#   sbatch --array=1-$N%8 scripts/submit_clip_features.sbatch
#
#SBATCH --job-name=clip-pfcand
#SBATCH --partition=private-dpnc-cpu,shared-cpu
# 7-25 min/file solo; >2 h when the array is unthrottled and slurm co-locates
# the tasks, so keep the generous limit AND throttle the array
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/clip_%A_%a.out
set -euo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

D=${FCC_PP_OUT}

mapfile -t FILES < <(ls "$D"/train/pp_output_train_split_*.h5 \
                        "$D"/val/pp_output_val_split_*.h5 \
                        "$D"/test/pp_output_test_split_*.h5)
f="${FILES[$((SLURM_ARRAY_TASK_ID - 1))]}"
fc=$(fcc_in_container "$f")
echo "[$(date)] clipping $f"

apptainer exec ${FCC_BINDS} \
    --env PYTHONNOUSERSITE=1 \
    "${FCC_IMAGE}" python -u /repo/scripts/clip_pfcand_features.py \
    "$fc" "$(fcc_in_container "$D")/norm_dict.yaml" \
    /repo/configs/upp/variables-fcc-higgs_decay.yaml 5
