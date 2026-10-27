"""Correct pre-resampling per-flavour plots for the Higgs-decay inputs.

UPP's own `initial` plots read the raw input via a single concatenated virtual
file and a sequential (shuffle=False) reader. Our inputs are one flavour per
file (fcc_zh_<decay>.h5), so a sequential read starves all but the first
flavour(s) -> the "only s/b/c" plots and puma bin-edge crashes.

This standalone script sidesteps that by reading an EQUAL sample from every
input file (so all 7 flavours are represented) and plotting the raw jet_pt /
jet_eta distributions per flavour. No reprocessing, no pipeline changes; this is
the lightweight equivalent of UPP's initial plots for per-flavour inputs.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import h5py
import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from puma import Histogram, HistogramPlot

FLAVOURS = {  # flavour_label -> (legend label, colour)
    0: ("ujets", "#a6cee3"),
    1: ("djets", "#1f78b4"),
    2: ("sjets", "#b2df8a"),
    3: ("cjets", "#33a02c"),
    4: ("bjets", "#e31a1c"),
    5: ("gjets", "#ff7f00"),
    6: ("taujets", "#6a3d9a"),
}


def main() -> int:
    """Plot pT, eta and flavour composition of the raw higgs_decay ntuples.

    Returns
    -------
    int
        Process exit status.
    """
    ap = argparse.ArgumentParser()
    default_in = os.environ.get("FCC_FTAG_DATA", "/srv/beegfs/scratch/groups/rodem/FCC_FTAG")
    ap.add_argument("--in-dir", default=f"{default_in}/higgs_decay")
    ap.add_argument("--out", required=True, help="output dir for PNGs")
    ap.add_argument("--per-file", type=int, default=200_000, help="jets read per input file")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read an equal sample from every input file -> all 7 flavours represented.
    pt, eta, lab = [], [], []
    for f in sorted(glob.glob(f"{args.in_dir}/fcc_zh_*.h5")):
        j = h5py.File(f)["jets"]
        n = min(args.per_file, len(j))
        pt.append(j["jet_pt"][:n])
        eta.append(j["jet_eta"][:n])
        lab.append(j["flavour_label"][:n])
        print(f"  {Path(f).name}: {n:,} jets")
    pt = np.concatenate(pt)
    eta = np.concatenate(eta)
    lab = np.concatenate(lab)
    print(
        f"total {len(lab):,} jets; per-flavour:",
        {FLAVOURS[i][0]: int((lab == i).sum()) for i in FLAVOURS},
    )

    for var, values, xlabel, bins in [
        ("jet_pt", pt, "Jet $p_T$ [GeV]", np.linspace(0, 120, 50)),
        ("jet_eta", eta, "Jet $\\eta$", np.linspace(-5, 5, 50)),
    ]:
        hp = HistogramPlot(
            ylabel="Normalised number of jets",
            xlabel=xlabel,
            atlas_first_tag="FCC Higgs-decay",
            atlas_second_tag="Pre-resampling (input)",
            atlas_brand="",
            n_ratio_panels=0,
        )
        for lv, (name, colour) in FLAVOURS.items():
            m = lab == lv
            if m.sum() < 10:
                continue
            hp.add(Histogram(values[m], bins=bins, norm=True, label=name, colour=colour))
        hp.draw()
        hp.savefig(out_dir / f"initial_{var}_allflav.png", dpi=120)
        print(f"  -> initial_{var}_allflav.png")

    print(f"\nPlots in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
