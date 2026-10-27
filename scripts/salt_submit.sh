#!/bin/sh
#SBATCH --job-name=Salt-FCC
#SBATCH --ntasks=1
# Reference-setup values (8 cpus, 16G/cpu). NOTE: the reference script also
# sets --mem-per-gpu=20G
# but this Baobab SLURM rejects mem-per-cpu + mem-per-gpu together (mutually
# exclusive), so we keep only mem-per-cpu (8x16G=128G, the functional one).
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16000
#SBATCH --gpus=1
# Reference-setup constraint. VERIFIED empirically (job 9587674) that batch_size 2000
# fits even the 10GB RTX 3080 -- peak 9467/10240 MiB, no OOM -- so we can use the
# cheaper/more-available 3080 pool too. Headroom on the 3080 is slim (~770 MiB):
# fine for tests; for a long unattended production run prefer a >=24GB card via
# --constraint="nvidia_geforce_rtx_3090|nvidia_rtx_a5000".
#SBATCH --constraint=COMPUTE_TYPE_AMPERE
#SBATCH --partition=private-dpnc-gpu
#SBATCH --time=2-00:00:00

# Salt training on the preprocessed Higgs-decay samples. FCC_SALT_CONFIG picks
# which configs/salt/*.yaml to train, defaulting to gn3.yaml:
#   sbatch scripts/salt_submit.sh
#   FCC_SALT_CONFIG=/repo/configs/salt/part.yaml sbatch scripts/salt_submit.sh

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
srun apptainer exec --contain -e --nv --pwd /repo ${FCC_BINDS} ${FCC_ENVFLAGS} \
    "${FCC_IMAGE}" \
    /repo/scripts/salt_commands.sh
