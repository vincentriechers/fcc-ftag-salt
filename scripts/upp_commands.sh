#!/usr/bin/env bash

# Let the script fail if error occurs
set -e

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

echo "Preprocessing with ${FCC_UPP_CONFIG}"

run_preprocess() {
    preprocess --config "${FCC_UPP_CONFIG}" "$@"
}

usage() {
    cat <<EOF
Usage:
  upp_commands.sh
    # default block (prep+resample+merge+norm+plot) with split=all

  upp_commands.sh normalise|normalize
  upp_commands.sh merge <split>
  upp_commands.sh plotting <split>
  upp_commands.sh prepare <component> <split>
  upp_commands.sh resampling <region> <split>
  upp_commands.sh fine_resampling <region> <component> <split>
EOF
}

need_args() {
    # Usage: need_args <n_required> <mode> "$@"
    local n_required="$1"
    local mode="$2"
    shift 2
    if [ "$#" -lt "$n_required" ]; then
        echo "ERROR: '$mode' needs ${n_required} argument(s), got $#." >&2
        usage
        exit 2
    fi
}

# ---- Start Preprocessing --------------------------------------------------
if [ "$#" -eq 0 ]; then
    echo "No input given. Processing default block."
    run_preprocess --prep --split=all
    run_preprocess --resample --split=all
    run_preprocess --merge --split=all
    run_preprocess --norm
    # Plotting is cosmetic. UPP's per-split post-resampling plots are correct,
    # but its "initial" (pre-resampling) plots read our one-flavour-per-file
    # inputs sequentially and can starve flavours / crash puma. Run per split and
    # never let a plot failure fail the job (otherwise the afterok chain blocks
    # training). For correct per-flavour PRE-resampling plots use the standalone
    # scripts/plot_initial_higgs_decay.py instead.
    for sp in train val test; do
        run_preprocess --plot --split "$sp" \
            || echo "WARNING: --plot $sp failed (cosmetic); merged data + norm/class dicts are complete."
    done
    exit 0
fi

MODE="$1"
shift

case "${MODE}" in
    normalise|normalize)
        echo "Normalisation selected. Processing..."
        run_preprocess --norm
        echo "Done!"
        ;;
    merge)
        need_args 1 "merge" "$@"
        echo "Start merging for $1. Processing..."
        run_preprocess --merge --split "$1"
        echo "Done!"
        ;;
    plotting)
        need_args 1 "plotting" "$@"
        echo "Plotting selected ($1). Processing..."
        run_preprocess --plot --split "$1"
        echo "Done!"
        ;;
    prepare)
        need_args 2 "prepare" "$@"
        echo "Start preparation for $1 ($2). Processing..."
        run_preprocess --prep --component "$1" --split "$2"
        echo "Done!"
        ;;
    resampling)
        need_args 2 "resampling" "$@"
        echo "Start resampling for $1 ($2). Processing..."
        run_preprocess --resample --region "$1" --split "$2"
        echo "Done!"
        ;;
    fine_resampling)
        need_args 3 "fine_resampling" "$@"
        echo "Start resampling for $2 ($3). Processing..."
        run_preprocess --resample --region "$1" --component "$2" --split "$3"
        echo "Done!"
        ;;
    -h|--help)
        usage
        ;;
    *)
        echo "Step '${MODE}' not supported!"
        usage
        exit 2
        ;;
esac
