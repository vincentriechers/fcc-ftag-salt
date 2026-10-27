#!/usr/bin/env bash
# Convert OUR weaver stage2 chunks (from scripts/submit_weaver_higgs_decay.sh)
# into Salt-native H5, one file per Higgs decay. Same as convert_higgs_decay.sh
# but reads LOCAL weaver output (stage2_<decay>_*.root, which carry our extended
# truth) instead of streaming the central flat ntuples from eos.
#
# Runs in the NORMAL python env (uproot/h5py), NOT the UPP container. The output
# fcc_zh_<decay>.h5 is the input to UPP (scripts/submit_upp.sh).
#
# Usage:
#   scripts/convert_weaver_higgs_decay.sh <stage2_dir> <outdir> [max_jets_per_decay]
set -euo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

STAGE2_DIR="${1:?need weaver stage2 dir (stage2_<decay>_*.root)}"
OUTDIR="${2:?need outdir for fcc_zh_<decay>.h5}"
MAXJETS="${3:-}"
REPO=${FCC_FTAG_REPO}

DECAYS=(Huu Hdd Hss Hcc Hbb Hgg Htautau)

export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
mkdir -p "$OUTDIR"

maxarg=()
[[ -n "$MAXJETS" ]] && maxarg=(--max-jets "$MAXJETS")

for decay in "${DECAYS[@]}"; do
    mapfile -t inputs < <(ls "${STAGE2_DIR}"/stage2_"${decay}"_*.root 2>/dev/null || true)
    if [[ "${#inputs[@]}" -eq 0 ]]; then
        echo "=== ${decay}: no stage2 chunks in ${STAGE2_DIR}, skipping ==="
        continue
    fi
    out="${OUTDIR}/fcc_zh_${decay}.h5"
    echo "=== ${decay}: ${#inputs[@]} chunk(s) -> ${out} ==="
    python -m fcc_ftag_converter.fcc_higgs_decay "${inputs[@]}" -o "${out}" \
        --chunk-size 50000 "${maxarg[@]}"
done

echo "All decays done -> ${OUTDIR}"
