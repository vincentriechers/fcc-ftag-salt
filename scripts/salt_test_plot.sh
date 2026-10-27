#!/bin/sh
#SBATCH --job-name=Salt-test-plot
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16000
#SBATCH --gpus=1
#SBATCH --constraint=COMPUTE_TYPE_AMPERE
#SBATCH --partition=shared-gpu,private-dpnc-gpu
#SBATCH --time=0-01:00:00

# Evaluate the trained tagger on the test split and make puma plots.
#   sbatch scripts/salt_test_plot.sh [path/to/your.ckpt]
# With no arg: auto-pick the lowest-val_loss checkpoint of the most recent run
# of the selected pipeline, so it follows the training automatically.

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

if [ -n "$1" ]; then
    CKPT="$1"
else
    RUN=$(ls -dt "${FCC_FTAG_REPO}"/logs/${FCC_RUN_GLOB}/ 2>/dev/null | head -1)
    # ckpt names: epoch=NNN-val_loss=X.XXXXX.ckpt -> sort by val_loss, take min.
    CKPT=$(ls "$RUN"ckpts/*.ckpt 2>/dev/null | sort -t= -k3 -g | head -1)
fi
echo "Using checkpoint: $CKPT"
[ -f "$CKPT" ] || { echo "ERROR: no checkpoint found" >&2; exit 2; }
CKPT_C=$(fcc_in_container "$CKPT")

echo "=== Step 1: salt test ==="
srun apptainer exec --contain -e --nv --pwd /repo ${FCC_BINDS} ${FCC_ENVFLAGS} \
    "${FCC_IMAGE}" \
    /repo/scripts/salt_test_commands.sh "$CKPT_C"

echo "=== Step 2: puma plots ==="
srun apptainer exec --contain -e --pwd /repo ${FCC_BINDS} ${FCC_ENVFLAGS} \
    "${FCC_IMAGE}" \
    /repo/scripts/puma_plot_commands.sh "$CKPT_C"
