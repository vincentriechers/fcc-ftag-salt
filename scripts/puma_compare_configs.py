r"""Compare several trained configs on the same test jets: ROCs plus a rejection table.

``puma_plot_higgs_decay.py`` evaluates ONE model and draws one figure per signal
with a line per background flavour, colour-coded. That is the right encoding for
a single tagger, but it cannot put models side by side -- which is what the
variable comparison (gn3_atlaslike vs gn3_fccfull) and the architecture
comparison (gn3_fccfull vs part_fccfull vs gn3pi_fccfull) need.

Overlaying every background of every model in one figure would mean 6 x N curves,
and puma can only put ONE rejection class in a ratio panel. So the unit here is
**one figure per (signal, background) pair**, with one curve per model: the
flavour is then fixed, colour is free to identify the model, and the ratio panel
shows exactly the quantity the comparison is about -- how many times better or
worse one model rejects that background at the same signal efficiency.

For a strange-tagging study the three that matter are s vs g, s vs u and s vs d.

The rejections at a fixed signal efficiency go to ``rejections.csv`` as well as
to stdout: scraping a log is not a reproducible way to fill a table.

Usage:
  puma_compare_configs.py --out DIR \
      --preds LABEL=path/to/preds.h5 --preds LABEL=... \
      [--wp 0.7] [--signals sjets,bjets,cjets] [--backgrounds gjets,ujets,djets] \
      [--ratio-ref LABEL] [--n 1000000] [--second-tag TEXT]

Each --preds takes ``LABEL=PATH``; the label appears in the legend and the CSV,
e.g. ``--preds 'ATLAS-like=logs/.../preds.h5'``. The first model is the ratio
reference unless --ratio-ref names another.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib as mpl

mpl.use("Agg")
import numpy as np
from puma import Roc, RocPlot

# Reuse the single-model script's conventions rather than restating them: the
# flavour labels, the stats cut-off inside _bg_rej and the header badge have to
# match the other plots of this analysis exactly, so they are imported, private
# names and all.
from puma_plot_higgs_decay import (
    CLASSES,
    FIG,
    FLAV,
    LABEL_IDS,
    ROC_XMIN,
    Y_SCALE,
    _bg_rej,  # noqa: PLC2701
    _find_class_scores,  # noqa: PLC2701
    _header,  # noqa: PLC2701
    _save,  # noqa: PLC2701
)

#: Colours identifying the MODEL. Exactly one background flavour is drawn per
#: figure and it is named on the axis, so colour here encodes the model alone.
#: Six entries, enough for a baseline plus five variants --
#: the list is indexed modulo its length, so a shorter list silently gives two
#: models the same colour.
MODEL_COLOURS = [
    "#000000",  # black
    "#d62728",  # red
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#9467bd",  # purple
]

#: Signal flavours plotted by default. s first: it is the hardest one.
DEFAULT_SIGNALS = ["sjets", "bjets", "cjets"]


def _load(spec: str, n_cap: int | None) -> tuple[str, dict, np.ndarray]:
    """Read one prediction file given as ``LABEL=PATH``.

    Parameters
    ----------
    spec : str
        ``LABEL=PATH``, where PATH is a Salt prediction H5.
    n_cap : int | None
        Optional cap on the number of jets read.

    Returns
    -------
    tuple[str, dict, np.ndarray]
        The label, the per-class score arrays, and the truth labels.

    Raises
    ------
    ValueError
        If the spec is not of the form ``LABEL=PATH``.
    """
    if "=" not in spec:
        raise ValueError(f"--preds needs LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    with h5py.File(path, "r") as f:
        ds = f["jets"]
        n = ds.shape[0] if n_cap is None else min(n_cap, ds.shape[0])
        jets = ds[:n]
    tagger, scores = _find_class_scores(jets, None)
    print(f"  {label:<18s} {n:>9,d} jets  tagger={tagger}")
    return label, scores, jets["flavour_label"][:]


def _roc_one_pair(
    out_dir: Path,
    models: list[tuple[str, dict, np.ndarray]],
    sig: str,
    bg: str,
    sig_eff: np.ndarray,
    second_tag: str,
    ref_label: str,
) -> bool:
    """One ROC figure for one (signal, background) pair, one curve per model.

    Parameters
    ----------
    out_dir : Path
        Directory to write the figure into.
    models : list[tuple[str, dict, np.ndarray]]
        ``(label, scores, truth)`` per model, in legend order.
    sig : str
        Signal flavour key, e.g. ``"sjets"``.
    bg : str
        Background flavour key, e.g. ``"gjets"``.
    sig_eff : np.ndarray
        Signal efficiencies to evaluate the rejection at.
    second_tag : str
        Free-form header line.
    ref_label : str
        Model label used as the ratio reference.

    Returns
    -------
    bool
        Whether a figure was written.
    """
    sig_name, bg_name = FLAV[sig][0], FLAV[bg][0]
    rp = RocPlot(
        n_ratio_panels=1 if len(models) > 1 else 0,
        ylabel=f"{bg_name} rejection",
        xlabel=f"{sig_name} efficiency",
        xmin=ROC_XMIN,
        xmax=1.0,
        y_scale=Y_SCALE,
        figsize=FIG,
        leg_fontsize=10,
        **_header(second_tag, int(models[0][2].size), extra=f"{sig_name} vs {bg_name}"),
    )
    curves = []
    for mi, (mlabel, scores, truth) in enumerate(models):
        sig_mask = truth == LABEL_IDS[sig]
        m = truth == LABEL_IDS[bg]
        if sig_mask.sum() == 0 or m.sum() == 0:
            continue
        eff, rej = _bg_rej(scores[sig][sig_mask], scores[sig][m], sig_eff)
        if eff.size == 0:
            continue
        curves.append((mi, mlabel, eff, rej, int(m.sum())))
    if not curves:
        return False

    # puma divides the curves point by point for the ratio panel and refuses a
    # pair whose signal efficiencies differ, while _bg_rej drops the points where
    # a model ran out of background jets -- so the grids are per-model. Keep only
    # the efficiencies every model reached.
    common = curves[0][2]
    for _, _, eff, _, _ in curves[1:]:
        common = np.intersect1d(common, eff)
    if common.size == 0:
        print(f"  -- {sig} vs {bg}: no signal efficiency is shared by all models, skipped")
        return False

    for mi, mlabel, eff, rej, n_bg in curves:
        keep = np.isin(eff, common)
        rp.add_roc(
            Roc(
                eff[keep],
                rej[keep],
                n_test=n_bg,
                rej_class=bg,
                signal_class=sig,
                label=mlabel,
                colour=MODEL_COLOURS[mi % len(MODEL_COLOURS)],
            ),
            key=mlabel,
            reference=(mlabel == ref_label),
        )
    if len(models) > 1:
        rp.set_ratio_class(1, bg, bg_name)
    rp.draw()
    _save(rp, out_dir, f"cmp_roc_{sig[:-4]}tag_vs_{bg[:-4]}")
    return True


def _table(
    models: list[tuple[str, dict, np.ndarray]],
    signals: list[str],
    backgrounds: list[str],
    wp: float,
) -> list[dict]:
    """Rejection of every background at a fixed signal efficiency, per model.

    Parameters
    ----------
    models : list[tuple[str, dict, np.ndarray]]
        ``(label, scores, truth)`` per model.
    signals : list[str]
        Signal flavours to tabulate.
    backgrounds : list[str]
        Background flavours to tabulate.
    wp : float
        Signal efficiency working point, e.g. 0.7.

    Returns
    -------
    list[dict]
        Rows with signal, background, model, rejection and the jet counts.
    """
    rows: list[dict] = []
    for sig in signals:
        for bg in backgrounds:
            if bg == sig:
                continue
            for mlabel, scores, truth in models:
                sig_mask = truth == LABEL_IDS[sig]
                m = truth == LABEL_IDS[bg]
                if sig_mask.sum() == 0 or m.sum() == 0:
                    continue
                _eff, rej = _bg_rej(scores[sig][sig_mask], scores[sig][m], np.array([wp]))
                rows.append({
                    "signal": sig,
                    "background": bg,
                    "model": mlabel,
                    "wp": wp,
                    # with fewer than MIN_BG_PASSING survivors the rejection
                    # measures the sample size, not the tagger, so _bg_rej drops
                    # the point and the cell is left empty rather than floored
                    "rejection": f"{rej[0]:.1f}" if rej.size else "",
                    "exhausted_statistics": "" if rej.size else "yes",
                    "n_signal": int(sig_mask.sum()),
                    "n_background": int(m.sum()),
                })
    return rows


def main() -> int:
    """Draw the per-pair ROC comparisons and write the rejection table.

    Returns
    -------
    int
        Process exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", action="append", required=True, help="LABEL=path/to/preds.h5")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--wp", type=float, default=0.7, help="signal efficiency for the table")
    ap.add_argument("--signals", default=",".join(DEFAULT_SIGNALS))
    ap.add_argument("--backgrounds", default="", help="default: every other trained class")
    ap.add_argument("--ratio-ref", default=None, help="model label used as the ratio reference")
    ap.add_argument("--n", type=int, default=None, help="cap on jets read per file")
    ap.add_argument("--second-tag", default="")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading predictions:")
    models = [_load(s, args.n) for s in args.preds]
    labels = [m[0] for m in models]
    ref_label = args.ratio_ref or labels[0]
    if ref_label not in labels:
        print(f"WARNING: --ratio-ref {ref_label!r} is not a model label; using {labels[0]!r}")
        ref_label = labels[0]

    # Every model has to be evaluated on the same test jets, or the curves are
    # not comparable. The truth vector is the cheapest proof of that.
    ref_truth = models[0][2]
    for label, _scores, truth in models[1:]:
        if truth.shape != ref_truth.shape or not np.array_equal(truth, ref_truth):
            print(
                f"WARNING: {label!r} has a different truth vector than {labels[0]!r} -- the "
                "models were evaluated on different jets, so this is not a controlled comparison"
            )

    def _norm(names: str) -> list[str]:
        return [n if n.endswith("jets") else f"{n}jets" for n in names.split(",") if n]

    signals = _norm(args.signals)
    backgrounds = _norm(args.backgrounds) if args.backgrounds else list(CLASSES)
    sig_eff = np.linspace(ROC_XMIN, 0.99, 60)

    print(f"\nFigures (ratio reference: {ref_label}):")
    for sig in signals:
        for bg in backgrounds:
            if bg == sig:
                continue
            if _roc_one_pair(out_dir, models, sig, bg, sig_eff, args.second_tag, ref_label):
                print(f"  -> cmp_roc_{sig[:-4]}tag_vs_{bg[:-4]}")

    rows = _table(models, signals, backgrounds, args.wp)
    if not rows:
        print("no rejections could be computed")
        return 1
    csv_path = out_dir / "rejections.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nRejection at {args.wp:.0%} signal efficiency  ->  {csv_path}")
    for sig in signals:
        print(f"\n  {FLAV[sig][0]} signal:")
        print("    " + "background".ljust(12) + "".join(f"{lab:>16s}" for lab in labels))
        for bg in backgrounds:
            if bg == sig:
                continue
            cells = []
            for lab in labels:
                hit = [
                    r
                    for r in rows
                    if r["signal"] == sig and r["background"] == bg and r["model"] == lab
                ]
                cells.append(hit[0]["rejection"] if hit and hit[0]["rejection"] else "n/a")
            print("    " + bg.ljust(12) + "".join(f"{c:>16s}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
