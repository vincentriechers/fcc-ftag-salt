#!/usr/bin/env bash
# Build a manifest of stage1 files and submit a SLURM array that flattens each
# one to a per-jet stage2 ntuple. Use this when stage1 already exists on disk;
# submit_weaver_higgs_decay.sh runs stage1 and stage2 together from Delphes.
#
# The stage2 chunks land as stage2_<decay>_<file>.root, which is what
# scripts/submit_convert_weaver_higgs_decay.sbatch globs for.
#
# Do NOT point this at a directory that already holds chunks from an older
# weaver. The converter detects the available truth branches from the FIRST
# input file only, so a mixed directory converts under one vintage's names.
#
# Usage:
#   scripts/submit_stage2_higgs_decay.sh [stage1_dir] [outdir] [nfiles_per_sample]
#     [stage1_dir]         default ${FCC_FTAG_DATA}/stage1/260918_stage1
#     [outdir]             default ${FCC_FTAG_DATA}/higgs_decay/stage2_260918
#     [nfiles_per_sample]  0 = all files (default); e.g. 2 for a quick test run
set -eo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

STAGE1_DIR="${1:-${FCC_FTAG_DATA}/stage1/260918_stage1}"
OUTDIR="${2:-${FCC_FTAG_DATA}/higgs_decay/stage2_260918}"
NPER="${3:-0}"
REPO=${FCC_FTAG_REPO}

mkdir -p "$OUTDIR" "$REPO/logs"
MAN="$OUTDIR/manifest.txt"
: > "$MAN"

for dir in "$STAGE1_DIR"/*/; do
    sample=$(basename "$dir")
    [[ "$sample" =~ _(H[a-zA-Z]+)_ ]] || continue
    decay="${BASH_REMATCH[1]}"
    files=("$dir"*.root)
    [[ "$NPER" -gt 0 ]] && files=("${files[@]:0:$NPER}")
    for f in "${files[@]}"; do
        echo "${decay}|${f}|$(basename "$f" .root)" >> "$MAN"
    done
    echo "[$decay] ${#files[@]} files"
done

N=$(wc -l < "$MAN")
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: no stage1 samples found under ${STAGE1_DIR}" >&2
    exit 1
fi
echo "manifest: $MAN  (${N} stage2 jobs)  ->  ${OUTDIR}"
echo "submitting array 1-${N} ..."
sbatch --array=1-"${N}" \
    --export=ALL,MANIFEST="$MAN",OUTDIR="$OUTDIR" \
    "$REPO/scripts/submit_stage2_higgs_decay.sbatch"
