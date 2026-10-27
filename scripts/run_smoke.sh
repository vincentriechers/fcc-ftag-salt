#!/usr/bin/env bash
# Smoke test: convert N events from a FCC EDM4hep ROOT file and produce
# comparison histograms vs the reference Salt H5.
#
# Usage:
#   scripts/run_smoke.sh                                 # defaults: 1000 events
#   scripts/run_smoke.sh -n 5000                         # 5000 events
#   scripts/run_smoke.sh -n 200 -i /path/to/file.root    # different input
#   scripts/run_smoke.sh -n 1000 --cone 0.3              # ATLAS cone radius
#   scripts/run_smoke.sh --skip-plots                    # only convert
#
# Outputs:
#   outputs/<tag>.h5 + outputs/<tag>.summary.{md,json}
#   plots/<tag>/        with one PNG per converter-filled variable

set -euo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

# ---- defaults --------------------------------------------------------------
INPUT_DEFAULT='root://eospublic.cern.ch//eos/experiment/fcc/ee/generation/DelphesEvents/winter2023/IDEA/p8_ee_tt_ecm365/events_002305720.root'
# The converter defaults to the schema shipped inside the package (FCC-tuned
# slot sizes 75/100); plots compare against the ATLAS reference H5 (50 slots).
REFERENCE_DEFAULT="${FCC_FTAG_DATA}/structure_test_file.h5"
N_EVENTS=1000
CHUNK_SIZE=500
CONE=1.5
TAG=""
INPUT="$INPUT_DEFAULT"
REFERENCE="$REFERENCE_DEFAULT"
SKIP_PLOTS=0
EXTRA_CONVERT_ARGS=()

# ---- arg parsing -----------------------------------------------------------
usage() {
    sed -n '2,17p' "$0"
    exit "${1:-0}"
}
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--max-events)   N_EVENTS="$2"; shift 2 ;;
        -i|--input)        INPUT="$2"; shift 2 ;;
        -r|--reference)    REFERENCE="$2"; shift 2 ;;
        -c|--chunk-size)   CHUNK_SIZE="$2"; shift 2 ;;
        --cone)            CONE="$2"; shift 2 ;;
        -t|--tag)          TAG="$2"; shift 2 ;;
        --skip-plots)      SKIP_PLOTS=1; shift ;;
        -h|--help)         usage 0 ;;
        --)                shift; EXTRA_CONVERT_ARGS+=("$@"); break ;;
        *)                 echo "unknown arg: $1" >&2; usage 1 ;;
    esac
done

# Derive a tag if none was given.
if [[ -z "$TAG" ]]; then
    TAG="tt_${N_EVENTS}"
fi
OUT_H5="outputs/${TAG}.h5"
PLOTS_DIR="plots/${TAG}"

# ---- env setup -------------------------------------------------------------
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${KEY4HEP_LOADED:-}" ]]; then
    echo "[env] sourcing key4hep ..."
    # key4hep's setup.sh references positional args ($1) without a default,
    # which breaks under `set -u`. Disable nounset for the duration of the source.
    set +u
    source /cvmfs/sw.hsf.org/key4hep/setup.sh > /dev/null
    set -u
    export KEY4HEP_LOADED=1
fi
export PYTHONPATH="$REPO_ROOT/src:${PYTHONPATH:-}"

mkdir -p outputs plots

# ---- convert ---------------------------------------------------------------
echo "[convert] $N_EVENTS events  cone=$CONE  →  $OUT_H5"
time python -m fcc_ftag_converter.cli \
    "$INPUT" \
    --output "$OUT_H5" \
    --max-events "$N_EVENTS" \
    --chunk-size "$CHUNK_SIZE" \
    --truth-cone-radius "$CONE" \
    "${EXTRA_CONVERT_ARGS[@]}"

if [[ "$SKIP_PLOTS" -eq 1 ]]; then
    echo "[done] skipped plots; output at $OUT_H5"
    exit 0
fi

# ---- plots -----------------------------------------------------------------
rm -rf "$PLOTS_DIR"
mkdir -p "$PLOTS_DIR"

VARS=(
    jets.pt jets.pt_btagJes jets.energy jets.mass
    jets.eta jets.eta_btagJes jets.absEta_btagJes jets.phi
    jets.n_tracks_ghost jets.n_tracks_ghost_rich_ip jets.n_flow_calo_cal_topo_clusters
    jets.HadronConeExclTruthLabelID jets.HadronConeExclExtendedTruthLabelID
    jets.HadronConeExclTruthLabelPdgId jets.LeptonDecayLabel jets.ptFromTruthDressedWZJet
    jets.n_b_tracks_ghost jets.n_c_tracks_ghost jets.n_hf_tracks_ghost jets.n_bc_tracks_ghost
    tracks_ghost.pt tracks_ghost.d0 tracks_ghost.z0SinTheta tracks_ghost.dphi tracks_ghost.deta tracks_ghost.qOverP
    tracks_ghost.qOverPUncertainty tracks_ghost.d0Uncertainty tracks_ghost.z0SinThetaUncertainty
    tracks_ghost.phiUncertainty tracks_ghost.thetaUncertainty
    tracks_ghost.lifetimeSignedD0Significance tracks_ghost.lifetimeSignedZ0SinThetaSignificance
    tracks_ghost.ftagTruthTypeLabel tracks_ghost.ftagTruthOriginLabel
    tracks_ghost.ftagTruthVertexIndex tracks_ghost.leptonID
    tracks_ghost_rich_ip.pt tracks_ghost_rich_ip.d0 tracks_ghost_rich_ip.z0SinTheta
    tracks_ghost_rich_ip.dphi tracks_ghost_rich_ip.deta tracks_ghost_rich_ip.qOverP
    tracks_ghost_rich_ip.qOverPUncertainty tracks_ghost_rich_ip.d0Uncertainty tracks_ghost_rich_ip.z0SinThetaUncertainty
    tracks_ghost_rich_ip.phiUncertainty tracks_ghost_rich_ip.thetaUncertainty
    tracks_ghost_rich_ip.lifetimeSignedD0Significance tracks_ghost_rich_ip.lifetimeSignedZ0SinThetaSignificance
    tracks_ghost_rich_ip.ftagTruthTypeLabel tracks_ghost_rich_ip.ftagTruthOriginLabel
    tracks_ghost_rich_ip.ftagTruthVertexIndex tracks_ghost_rich_ip.leptonID
    flows.pt flows.energy flows.deta flows.dphi flows.isCharged
    electrons.pt electrons.eta electrons.phi electrons.abs_eta electrons.ftag_et
    electrons.ptfrac electrons.ptrel electrons.dr electrons.qOverP
    electrons.d0RelativeToBeamspot electrons.d0RelativeToBeamspotSignificance
    electrons.ftag_ptVarCone30OverPt electrons.ftag_deltaPOverP electrons.ftag_energyOverP
    muons.pt muons.eta muons.phi muons.ptfrac muons.ptrel muons.dr muons.qOverP
)
ARGS=(); for v in "${VARS[@]}"; do ARGS+=(--variable "$v"); done

echo "[plots] ${#VARS[@]} variables  →  $PLOTS_DIR/"
python -m fcc_ftag_converter.compare \
    "$REFERENCE" \
    "$OUT_H5" \
    --label-a reference --label-b "$TAG" \
    --output-dir "$PLOTS_DIR" \
    "${ARGS[@]}"

echo "[done]"
echo "  H5:      $OUT_H5"
echo "  Summary: ${OUT_H5%.h5}.summary.md"
echo "  Plots:   $PLOTS_DIR/  (PNGs + summary.md + summary.csv)"
