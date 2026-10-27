#!/bin/sh
# Shared location defaults for everything under scripts/. Source it as
#
#   . "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"
#
# Under sbatch the script body runs from a spool copy, so $0 cannot locate the
# checkout; SLURM_SUBMIT_DIR is the directory sbatch was called from, which the
# README tells you to make the repo root.
#
# Override any of these from the environment instead of editing scripts:
#   FCC_FTAG_REPO   this checkout
#   FCC_FTAG_DATA   sample / preprocessing / training area
#   FCC_IMAGE       container used for every stage of the chain
#
# Inside the container the checkout is bound to /repo and the sample area to
# /data (see FCC_BINDS below), so both are detected rather than configured.

if [ -z "${FCC_FTAG_REPO:-}" ] && [ -d /repo/scripts ]; then
    FCC_FTAG_REPO=/repo
fi
FCC_FTAG_REPO="${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"

if [ -z "${FCC_FTAG_DATA:-}" ] && [ -d /repo/scripts ]; then
    FCC_FTAG_DATA=/data
fi
# SITE DEFAULT -- change this line, or export FCC_FTAG_DATA, for your cluster.
FCC_FTAG_DATA="${FCC_FTAG_DATA:-/srv/beegfs/scratch/groups/rodem/FCC_FTAG}"

FCC_IMAGE="${FCC_IMAGE:-docker://ghcr.io/vincentriechers/fcc-ftag-salt-converter:latest}"

# The one chain this repo drives: FCC-ee Higgs-decay samples -> UPP -> a 7-class
# Salt tagger. Paths are the in-container ones, see FCC_BINDS below.
FCC_UPP_CONFIG=/repo/configs/upp/fcc-zh.yaml
# Overridable per job to train any other configs/salt/*.yaml on the same data
# and scripts:
#   FCC_SALT_CONFIG=/repo/configs/salt/part.yaml sbatch scripts/salt_submit.sh
FCC_SALT_CONFIG="${FCC_SALT_CONFIG:-/repo/configs/salt/gn3.yaml}"
FCC_PUMA_SCRIPT=/repo/scripts/puma_plot_higgs_decay.py
FCC_RUN_GLOB='FCC_higgs_decay_7class_*'

# Where the UPP config writes its merged output, read out of that config rather
# than repeated here. Any step that rewrites the preprocessed H5 in place needs
# this path, and hard-coding it is how such a step silently rewrites the PREVIOUS
# production after the config has moved on to a new output directory. Override
# FCC_PP_OUT to point those steps somewhere else on purpose.
FCC_PP_OUT="${FCC_PP_OUT:-$(sed -n 's|^[[:space:]]*out_dir:[[:space:]]*/data/|'"${FCC_FTAG_DATA}"'/|p' \
    "${FCC_FTAG_REPO}/configs/upp/${FCC_UPP_CONFIG##*/}" 2>/dev/null | tr -d ' ' | sed 's|/$||')}"
export FCC_PP_OUT

# --contain -e wipes the environment, so anything the in-container scripts read
# has to be handed over explicitly.
FCC_ENVFLAGS="--env FCC_SALT_CONFIG=${FCC_SALT_CONFIG} --env FCC_SALT_CKPT=${FCC_SALT_CKPT:-}"

# Bind flags for every apptainer call. /repo and /data are what the UPP and Salt
# configs are written against: UPP resolves a relative !include against its own
# package config dir and neither tool expands environment variables in a config,
# so identical in-container paths are the only way one set of configs can work
# for every checkout and every site.
FCC_BINDS="-B /home -B /tmp -B /srv -B ${FCC_FTAG_REPO}:/repo -B ${FCC_FTAG_DATA}:/data"

export FCC_FTAG_REPO FCC_FTAG_DATA FCC_IMAGE FCC_BINDS
export FCC_UPP_CONFIG FCC_SALT_CONFIG FCC_PUMA_SCRIPT FCC_RUN_GLOB FCC_ENVFLAGS

# Translate a host path into the same file as seen from inside the container.
# Paths outside the checkout and the data area are passed through unchanged --
# /home, /tmp and /srv are bound at their host locations.
fcc_in_container() {
    case "$1" in
        "${FCC_FTAG_REPO}"/*) echo "/repo${1#"${FCC_FTAG_REPO}"}" ;;
        "${FCC_FTAG_DATA}"/*) echo "/data${1#"${FCC_FTAG_DATA}"}" ;;
        *) echo "$1" ;;
    esac
}

if [ ! -d "${FCC_FTAG_REPO}/scripts" ]; then
    echo "ERROR: FCC_FTAG_REPO=${FCC_FTAG_REPO} is not a checkout of this repo." >&2
    echo "       Submit from the repo root, or export FCC_FTAG_REPO=/path/to/checkout." >&2
    exit 2
fi
