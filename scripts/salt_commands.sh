#!/bin/bash

# Let the script fail if error occurs
set -e

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

# Load Comet API credentials if available (file NOT in git; a plain shell file
# exporting COMET_API_KEY / COMET_WORKSPACE / COMET_PROJECT_NAME). Optional:
# without it salt logs offline instead of to comet.com.
COMET_KEY_FILE="${COMET_KEY_FILE:-${HOME}/flav_tag/submission_scripts/scripts/API_Keys.sh}"
if [[ -f "${COMET_KEY_FILE}" ]]; then
    source "${COMET_KEY_FILE}"
else
    echo "WARNING: ${COMET_KEY_FILE} not found -- Comet logging disabled/offline"
fi

# Ignore the user's ~/.local site-packages -- inside the container we want the
# pre-installed h5py/torch/salt from /usr/local, NOT the user's pip-installed
# stuff that bleeds through the /home bind mount.
export PYTHONNOUSERSITE=1

# Helps against memory fragmentation on small GPUs (the cluster often has
# 10GB Ampere cards). Effectively saves ~1GB during backward.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resuming: set FCC_SALT_CKPT to the LAST checkpoint of the interrupted run.
# The checkpoint carries epoch, optimizer and lr_scheduler state (verified:
# last_epoch 735000 of total_steps 770000 on the gn3_atlaslike run), so lightning
# continues the SAME OneCycle rather than restarting it. Pass the RESOLVED
# config the run wrote to logs/<run>/config.yaml, not the original: with
# use_class_dict: True salt refuses a resume unless the class weights are
# already baked in, and the checkpoint's stored hyper_parameters would override
# the model section of the original anyway.
#   FCC_SALT_CONFIG=/repo/logs/<run>/config.yaml \
#   FCC_SALT_CKPT=/repo/logs/<run>/ckpts/epoch=0NN-val_loss=X.ckpt \
#   sbatch scripts/salt_submit.sh
echo "Training with ${FCC_SALT_CONFIG}"
if [[ -n "${FCC_SALT_CKPT:-}" ]]; then
    echo "Resuming from ${FCC_SALT_CKPT}"
    salt fit --config "${FCC_SALT_CONFIG}" --force --ckpt_path "${FCC_SALT_CKPT}"
else
    salt fit --config "${FCC_SALT_CONFIG}" --force
fi
