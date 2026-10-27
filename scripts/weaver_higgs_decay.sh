#!/usr/bin/env bash
# Run OUR extended FCCAnalyses weaver (stage1 -> stage2) on ONE Delphes EDM4hep
# file, producing a flat stage2 ntuple that carries the extended per-constituent
# truth (pfcand_truthOriginLabel / truthTypeLabel / truthSourceLabel /
# truthVertexIndex / truthPdgId) and the per-jet ghost and truth-jet labels.
# That ntuple is the input to fcc_ftag_converter.fcc_higgs_decay.
#
# Runs in the el9 key4hep + shims + FCCAnalyses env (see ~/fcc-shims/fccenv.sh),
# NOT the UPP/Salt container.
#
# Usage:
#   scripts/weaver_higgs_decay.sh <decay> <delphes_url> <out_dir> <tag> [nevents]
#     <decay>       Hbb|Hcc|Hdd|Hss|Huu|Hgg|Htautau  (drives the stage2 flavour
#                   label; stage2 infers it from the 'H<decay>' substring in the
#                   stage1 filename)
#     <delphes_url> root://.../events_<N>.root   (one Delphes EDM4hep file)
#     <out_dir>     dir for stage2_<decay>_<tag>.root
#     <tag>         unique tag (e.g. file index)
#     [nevents]     optional event cap (default: all events in the file)
#
# Env knobs: FCCANALYSES_DIR (default ~/FCCAnalyses), NCPUS (default 1; only used
# when nevents is not given, since fccanalysis multithreads only for full runs).
set -euo pipefail

DECAY="${1:?need decay (Hbb|Hcc|...)}"
INPUT="${2:?need delphes url}"
OUTDIR="${3:?need out_dir}"
TAG="${4:?need tag}"
NEV="${5:-}"

# Default: the EDM4hep-1.0 port (reads pre_summer2026* productions). For the
# old pinned-stack fork (winter2023*-era files) set:
#   FCCANALYSES_DIR=$HOME/FCCAnalyses FCC_WEAVER_ENV=$HOME/fcc-shims/fccenv.sh
export FCCANALYSES_DIR="${FCCANALYSES_DIR:-$HOME/FCCAnalyses-edm4hep1}"
FCC_WEAVER_ENV="${FCC_WEAVER_ENV:-$HOME/fcc-shims/fccenv-edm4hep1.sh}"

# shellcheck disable=SC1090
source "$FCC_WEAVER_ENV"
cd "$FCCANALYSES_DIR"

mkdir -p "$OUTDIR"
STAGE1="${TMPDIR:-/tmp}/stage1_${DECAY}_${TAG}.root"
STAGE2="${OUTDIR}/stage2_${DECAY}_${TAG}.root"

nevarg=()
[[ -n "$NEV" ]] && nevarg=(--nevents "$NEV")

echo "[weaver] stage1 ${DECAY} tag=${TAG} <- ${INPUT}"
# NOTE: stage1 output MUST contain the 'H<decay>' tag -- stage2 reads THIS path
# to infer the jet flavour (recojet_is<F>).
fccanalysis run examples/FCCee/weaver/stage1.py \
    --output "$STAGE1" --files-list "$INPUT" --ncpus "${NCPUS:-1}" "${nevarg[@]}"

echo "[weaver] stage2 -> ${STAGE2}"
python examples/FCCee/weaver/stage2.py "$STAGE1" "$STAGE2" 0 100000000

rm -f "$STAGE1"
echo "[weaver] done: ${STAGE2}"
