"""Comparison histograms: full 2.7M-event FCC sample (parts 1+2) vs ATLAS reference.

Normalised step (line-only) histograms. Variables whose name hints at integer
category labels are drawn log-y. Already-rendered PNGs are skipped, so the
script is resume-friendly and can be sharded one dataset per sbatch array task.

Usage:
    python scripts/plot_full_vs_ref.py                 # all datasets
    python scripts/plot_full_vs_ref.py --dataset jets  # just one dataset
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

DATA = Path(os.environ.get("FCC_FTAG_DATA", "/srv/beegfs/scratch/groups/rodem/FCC_FTAG"))
OUTDIR = DATA / "plots_full"
REF = str(DATA / "structure_test_file.h5")
PARTS = [str(DATA / f"fcc_tt_365_part{i}.h5") for i in (1, 2, 3)]
NBINS = 60

# variables filled by the converter (curated; same set as scripts/run_smoke.sh)
VARS: dict[str, list[str]] = {
    "jets": [
        "pt",
        "pt_btagJes",
        "energy",
        "mass",
        "eta",
        "eta_btagJes",
        "absEta_btagJes",
        "phi",
        "n_tracks_ghost",
        "n_tracks_ghost_rich_ip",
        "n_flow_calo_cal_topo_clusters",
        "HadronConeExclTruthLabelID",
        "HadronConeExclExtendedTruthLabelID",
        "HadronConeExclTruthLabelPdgId",
        "HadronGhostInitialTruthLabelPdgId",
        "LeptonDecayLabel",
        "ptFromTruthDressedWZJet",
        "n_b_tracks_ghost",
        "n_c_tracks_ghost",
        "n_hf_tracks_ghost",
        "n_bc_tracks_ghost",
    ],
    "tracks_ghost": [
        "pt",
        "d0",
        "z0SinTheta",
        "dphi",
        "deta",
        "qOverP",
        "qOverPUncertainty",
        "d0Uncertainty",
        "z0SinThetaUncertainty",
        "phiUncertainty",
        "thetaUncertainty",
        "lifetimeSignedD0Significance",
        "lifetimeSignedZ0SinThetaSignificance",
        "ftagTruthTypeLabel",
        "ftagTruthOriginLabel",
        "ftagTruthVertexIndex",
        "leptonID",
    ],
    "flows": ["pt", "energy", "deta", "dphi", "isCharged"],
    "electrons": [
        "pt",
        "eta",
        "phi",
        "abs_eta",
        "ftag_et",
        "ptfrac",
        "ptrel",
        "dr",
        "qOverP",
        "d0RelativeToBeamspot",
        "d0RelativeToBeamspotSignificance",
        "ftag_ptVarCone30OverPt",
    ],
    "muons": ["pt", "eta", "phi", "ptfrac", "ptrel", "dr", "qOverP"],
}

# Substring → log-scale y-axis. Any variable name containing one of these
# strings (case-sensitive — these match ATLAS naming) gets log scale.
LOG_SCALE_HINTS = (
    "PdgId",
    "TruthLabel",
    "LabelID",
    "Label",
    "leptonID",
    "isCharged",
    "ftagTruth",
    "Decay",
)

# Datasets that store data as (N_jets, slots) with a "valid" flag.
_RAGGED = {"tracks_ghost", "flows", "electrons", "muons"}


def _wants_log(varname: str) -> bool:
    return any(hint in varname for hint in LOG_SCALE_HINTS)


def _load_var(handle: h5py.File, dataset: str, variable: str) -> np.ndarray:
    """Return the flat array of values for a single variable, valid-masked.

    Parameters
    ----------
    handle : h5py.File
        Open H5 file.
    dataset : str
        Dataset name inside the file.
    variable : str
        Field name inside the dataset.

    Returns
    -------
    np.ndarray
        Finite values only; empty when the field is absent.
    """
    ds = handle[dataset]
    if variable not in ds.dtype.names:
        return np.empty(0)
    arr = ds[variable][:]
    ragged = dataset in _RAGGED and "valid" in ds.dtype.names
    arr = arr[ds["valid"][:]] if ragged else arr.ravel()
    if arr.dtype.kind == "f":
        arr = arr[np.isfinite(arr)]
    return arr


def _bins(a: np.ndarray, b: np.ndarray, is_integer: bool) -> np.ndarray:
    """Joint binning across both samples. Integer-categorical → tight integer bins.

    Parameters
    ----------
    a : np.ndarray
        First sample.
    b : np.ndarray
        Second sample.
    is_integer : bool
        Use one bin per integer value instead of a fixed bin count.

    Returns
    -------
    np.ndarray
        Bin edges covering both samples.
    """
    combined = np.concatenate([a, b])
    if combined.size == 0:
        return np.array([0.0, 1.0])
    if combined.dtype.kind == "b":
        combined = combined.astype(np.int8)
    lo, hi = np.percentile(combined, [0.1, 99.9])
    if hi <= lo:
        hi = lo + 1.0
    if is_integer:
        lo_i, hi_i = int(np.floor(combined.min())), int(np.ceil(combined.max()))
        return np.arange(lo_i, hi_i + 2) - 0.5
    return np.linspace(lo, hi, NBINS + 1)


def _plot(dataset: str, variable: str, ours: np.ndarray, ref: np.ndarray, outdir: Path) -> None:
    if ours.size == 0 and ref.size == 0:
        return
    out = outdir / f"{dataset}.{variable}.png"
    if out.exists():
        return  # resume-friendly: skip already-rendered plots
    int_like = (ours.dtype.kind in "iub" or ref.dtype.kind in "iub") or _wants_log(variable)
    bins = _bins(ours, ref, int_like)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    edges = bins.tolist()
    if ref.size:
        ax.hist(
            ref,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"ATLAS ref ({ref.size:,})",
            color="C0",
        )
    if ours.size:
        ax.hist(
            ours,
            bins=edges,
            density=True,
            histtype="step",
            linewidth=1.8,
            label=f"FCC tt 2.7M ({ours.size:,})",
            color="C3",
        )
    ax.set_title(f"{dataset}.{variable}")
    ax.set_xlabel(variable)
    ax.set_ylabel("normalised count")
    if _wants_log(variable):
        ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  wrote {out.name}")


def main() -> None:
    """Plot every configured variable, full sample against the reference."""
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        choices=sorted(VARS.keys()),
        help="Only plot this dataset (default: all). Use for sbatch sharding.",
    )
    args = ap.parse_args()
    targets = [args.dataset] if args.dataset else list(VARS.keys())

    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {OUTDIR}")
    print(f"datasets: {targets}")

    fref = h5py.File(REF, "r")
    part_handles = [h5py.File(p, "r") for p in PARTS]
    try:
        for dataset in targets:
            if dataset not in fref:
                print(f"skip {dataset}: not in ref")
                continue
            print(f"\n[{dataset}]")
            for v in VARS[dataset]:
                out = OUTDIR / f"{dataset}.{v}.png"
                if out.exists():
                    print(f"  skip {v} (already done)")
                    continue
                ref_arr = _load_var(fref, dataset, v)
                pieces = [_load_var(h, dataset, v) for h in part_handles]
                ours = np.concatenate(pieces) if any(p.size for p in pieces) else np.empty(0)
                _plot(dataset, v, ours, ref_arr, OUTDIR)
    finally:
        fref.close()
        for h in part_handles:
            h.close()

    print(f"\ndone — {len(list(OUTDIR.glob('*.png')))} plots in {OUTDIR}")


if __name__ == "__main__":
    main()
