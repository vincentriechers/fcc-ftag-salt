#!/usr/bin/env bash
# Build a (decay, Delphes-file) manifest for the 7 H->qq channels and submit a
# SLURM array that runs OUR extended weaver (stage1->stage2) on each file. The
# stage2 chunks carry the per-constituent truth labels and the per-jet ghost
# and truth-jet labels,
# and are the input to scripts/convert_weaver_higgs_decay.sh.
#
# Source: pre_summer2026_training IDEA wzp6_ee_nunuH_H{uu,dd,ss,cc,bb,gg,tautau}_ecm240
# -- the production upstream of the latest reference ParT training ntuples
# (pre_summer2026/samples_v1 chunks). ~100 files x 100k ev x 2
# jets ~= 20M jets/class available (the reference used ~half). Z(->nunu)H(->jj): both
# jets from the H decay, no prompt photon. NOTE these files are EDM4hep 1.0 --
# they REQUIRE the FCCAnalyses-edm4hep1 worktree (weaver_higgs_decay.sh default),
# not the old pinned-stack fork.
#
# Usage:
#   scripts/submit_weaver_higgs_decay.sh [outdir] [nfiles_per_channel]
#     [outdir]              default /srv/.../FCC_FTAG/higgs_decay/weaver_stage2
#     [nfiles_per_channel]  0 = all files (default); e.g. 2 for a quick test run
set -euo pipefail

. "${FCC_FTAG_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}/scripts/env.sh"

OUTDIR="${1:-${FCC_FTAG_DATA}/higgs_decay/weaver_stage2}"
NPER="${2:-0}"
REPO=${FCC_FTAG_REPO}
EOSHOST=root://eospublic.cern.ch
GENBASE=/eos/experiment/fcc/ee/generation/DelphesEvents/pre_summer2026_training/IDEA
ECM=ecm240

# decay label (drives stage2 flavour) -> Delphes sub-sample name
DECAYS=(Hbb Hcc Hdd Hss Huu Hgg Htautau)
declare -A SUB=( [Hbb]=wzp6_ee_nunuH_Hbb [Hcc]=wzp6_ee_nunuH_Hcc [Hdd]=wzp6_ee_nunuH_Hdd [Hss]=wzp6_ee_nunuH_Hss [Huu]=wzp6_ee_nunuH_Huu [Hgg]=wzp6_ee_nunuH_Hgg [Htautau]=wzp6_ee_nunuH_Htautau )

mkdir -p "$OUTDIR" "$REPO/logs"
MAN="$OUTDIR/manifest.txt"
: > "$MAN"

for decay in "${DECAYS[@]}"; do
    sub="${SUB[$decay]}_${ECM}"
    mapfile -t files < <(xrdfs "$EOSHOST" ls "${GENBASE}/${sub}" \
        | grep -E 'events_.*\.root$' | grep -v 'sys\.v')
    [[ "$NPER" -gt 0 ]] && files=("${files[@]:0:$NPER}")
    i=0
    for f in "${files[@]}"; do
        echo "${decay}|${EOSHOST}/${f}|${i}" >> "$MAN"
        i=$((i + 1))
    done
    echo "[$decay] ${#files[@]} files"
done

N=$(wc -l < "$MAN")
if [[ "$N" -eq 0 ]]; then
    echo "ERROR: empty manifest (xrdfs enumeration failed?)" >&2
    exit 1
fi
echo "manifest: $MAN  (${N} weaver jobs)  ->  ${OUTDIR}"
echo "submitting array 1-${N} ..."
sbatch --array=1-"${N}" \
    --export=ALL,MANIFEST="$MAN",OUTDIR="$OUTDIR" \
    "$REPO/scripts/submit_weaver_higgs_decay.sbatch"
