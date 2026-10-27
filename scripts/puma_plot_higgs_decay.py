"""ROC + score histograms for the FCC-ee higgs_decay 7-class tagger (puma).

Reads the Salt PredictionWriter output (jets group with score columns
``<tagger>_p{u,d,s,c,b,g,tau}`` plus the carried-through ``flavour_label`` truth)
and writes PNGs:

  Score histograms (per output node, all 7 truth flavours overlaid, log-y with
  automatic y-scaling): score_p{u,d,s,c,b,g,tau}.png

  ROCs -- ONE per signal flavour, with a SEPARATE line for EACH other trained
  class (never merged into one "light" line): roc_{u,d,s,c,b,g,tau}tag.png.
  7-class -> 6 lines per plot. rej@70% eff printed to stdout.

  Differential background rejection vs jet pT / eta at a flat 70% per-bin signal
  efficiency: rej_vs_{pt,eta}_{b,c,s}tag.png.

  Row-normalised confusion matrices, one per classification head:
  confmat_flavour.png plus confmat_{track_origin,track_type,jets_bccharge}.png.

  Vertexing efficiency, purity and fake rate vs jet pT: vtx_*_vs_pt.png.

The auxiliary heads are read from the Salt training config written next to the
checkpoint (``model_of_record_config.yaml``), so their class names, truth column
and label map always match what the model was trained on. Without that file the
script still produces everything above it.

Discriminant per signal S = the network's own signal node p_S (equivalently
D = ln(p_S/(1-p_S)) = logit(p_S), monotonic -> identical ROC). Parameter-free
(no fc/fb fractions), so every background is treated on equal footing and can be
shown as its own line.

Class order matches flavour_label (fcc_higgs_decay.ONEHOT_LABELS):
u=0 d=1 s=2 c=3 b=4 g=5 tau=6.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib as mpl

mpl.use("Agg")
import numpy as np
import yaml
from puma import (
    Histogram,
    HistogramPlot,
    PlotBase,
    Roc,
    RocPlot,
    VarVsEff,
    VarVsEffPlot,
    VarVsVtx,
    VarVsVtxPlot,
)
from puma.matshow import MatshowPlot
from puma.utils.vertexing import (
    calculate_vertex_metrics,
    clean_reco_vertices,
    clean_truth_vertices,
)

LABEL_IDS = {
    "ujets": 0,
    "djets": 1,
    "sjets": 2,
    "cjets": 3,
    "bjets": 4,
    "gjets": 5,
    "taujets": 6,
}
# Score-column suffixes as Salt writes them: it honours the _px for the standard
# flavours (pu/pc/pb/ptau) but falls back to p<name> for the non-standard d/s/g.
PX = {
    "ujets": "pu",
    "djets": "pdjets",
    "sjets": "psjets",
    "cjets": "pc",
    "bjets": "pb",
    "gjets": "pgjets",
    "taujets": "ptau",
}
CLASSES = list(LABEL_IDS)

# FCC flavour palette: class name -> (legend label, fixed colour). Fixed,
# visually distinct colours for EVERY class (incl. the non-ATLAS d/s/g nodes),
# used for both score histograms and ROC lines so a class keeps one colour.
FLAV = {
    "ujets": ("$u$-jets", "#7f7f7f"),  # grey
    "djets": ("$d$-jets", "#ff7f0e"),  # orange
    "sjets": ("$s$-jets", "#9467bd"),  # purple
    "cjets": ("$c$-jets", "#2ca02c"),  # green
    "bjets": ("$b$-jets", "#1f77b4"),  # blue
    "gjets": ("$g$-jets", "#8c564b"),  # brown
    "taujets": (r"$\tau$-jets", "#e377c2"),  # pink
}

# Subscript for the output-node axis label, so the tau node reads as a symbol.
NODE_MATH = {
    "ujets": "u",
    "djets": "d",
    "sjets": "s",
    "cjets": "c",
    "bjets": "b",
    "gjets": "g",
    "taujets": r"\tau",
}

# atlasify draws the badge from `brand`, so the ATLAS wordmark is a parameter
# rather than something baked into the style -- same bold-italic badge, our name.
BRAND = "FCC-ee"
FIRST_TAG = "Simulation, Work in Progress"
SAMPLE_TAG = r"$e^+e^- \rightarrow ZH$, $\sqrt{s} = 240$ GeV"

# Smallest number of background jets that may survive a cut before the rejection
# stops meaning anything: past it 1/eff_bg measures the sample size, not the
# tagger, and the curve flattens into a plateau.
MIN_BG_PASSING = 5

# One linestyle for every flavour, so a curve is identified by its colour alone.
LINESTYLE = "-"

# Figure sizes chosen so the fonts, which are absolute, come out large relative
# to the canvas. A wider figure just adds whitespace around the same text.
FIG = (6.0, 4.5)
FIG_MATRIX = (6.8, 5.8)

# Print-quality raster alongside the PDF, so a PNG dropped into a talk or a
# note holds up when it is scaled.
PNG_DPI = 500

# Headroom above the curves for the badge and the legend. y_scale multiplies
# the number of decades a log axis spans, so it has to stay close to 1: at 1.6
# the b-tag ROC ran to 5e6 with no curve above 1e4, over half the canvas empty.
Y_SCALE = 1.3

# Efficiency range shown on every ROC.
ROC_XMIN = 0.5


def _count(n: int) -> str:
    """Jet count as a short label, in millions once it reaches one.

    Parameters
    ----------
    n : int
        Number of jets.

    Returns
    -------
    str
        For example ``2M``, ``4.1M`` or ``200k``.
    """
    value, unit = (n / 1e6, "M") if n >= 1_000_000 else (n / 1e3, "k")
    return f"{value:.1f}".rstrip("0").rstrip(".") + unit


def _save(plot: PlotBase, out_dir: Path, stem: str) -> None:
    """Write one figure as both PNG and PDF.

    Parameters
    ----------
    plot : PlotBase
        A drawn puma plot.
    out_dir : Path
        Directory to write into.
    stem : str
        File name without a suffix.
    """
    plot.savefig(out_dir / f"{stem}.png", dpi=PNG_DPI)
    plot.savefig(out_dir / f"{stem}.pdf")


def _header(second_tag: str, n_jets: int, extra: str = "") -> dict:
    """Badge and tag lines shared by every plot.

    Parameters
    ----------
    second_tag : str
        Free-form line from ``--second-tag``, dropped when empty.
    n_jets : int
        Number of jets entering the plot, reported in the header.
    extra : str
        Additional line describing the specific plot, e.g. the working point.

    Returns
    -------
    dict
        Keyword arguments accepted by every puma plot class.
    """
    lines = [SAMPLE_TAG, f"7-class tagger, {_count(n_jets)} jets"]
    if extra:
        lines.append(extra)
    if second_tag:
        lines.append(second_tag)
    return {
        "atlas_brand": BRAND,
        "atlas_first_tag": FIRST_TAG,
        "atlas_second_tag": "\n".join(lines),
    }


def _find_class_scores(jets: np.ndarray, tagger: str | None) -> tuple[str, dict]:
    names = jets.dtype.names or ()
    if tagger is None:
        cands = []
        for n in names:
            if n.endswith("_pb"):
                prefix = n[:-3]
                if all(f"{prefix}_{s}" in names for s in PX.values()):
                    cands.append(prefix)
        if not cands:
            raise KeyError(f"No tagger prefix with all 7 score columns. Columns: {names}")
        tagger = cands[-1]
        print(f"Auto-detected tagger prefix: {tagger}")
    scores = {}
    for cls in CLASSES:
        col = f"{tagger}_{PX[cls]}"
        if col not in names:
            raise KeyError(f"Column {col} missing. Available: {names}")
        scores[cls] = jets[col][:]
    return tagger, scores


def _bg_rej(
    d_sig: np.ndarray, d_bg: np.ndarray, sig_effs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Background rejection per signal efficiency, cut off where stats run out.

    Parameters
    ----------
    d_sig : np.ndarray
        Discriminant for the signal jets.
    d_bg : np.ndarray
        Discriminant for the background jets.
    sig_effs : np.ndarray
        Signal efficiencies at which to evaluate the rejection.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The retained signal efficiencies and the rejection at each. Working
        points leaving fewer than ``MIN_BG_PASSING`` background jets are
        dropped rather than floored: they say how big the sample is, not how
        well the tagger separates.
    """
    # One sort plus a searchsorted, rather than a full pass over d_bg per
    # threshold: this runs for every (signal, background) pair.
    ordered = np.sort(d_bg)
    thresholds = np.quantile(d_sig, 1.0 - sig_effs)
    passing = len(ordered) - np.searchsorted(ordered, thresholds, side="left")
    keep = passing >= MIN_BG_PASSING
    return sig_effs[keep], len(ordered) / passing[keep]


def _longest_usable_bins(
    bins: np.ndarray, var: np.ndarray, masks: list[np.ndarray], min_count: int = 100
) -> np.ndarray | None:
    """Longest run of bins every selection populates, as an edge array.

    A ``flat_per_bin`` working point needs a signal quantile in each bin, so a
    bin no selection fills raises inside puma. Trimming to the populated range
    keeps that from turning a sparse tail into a crash.

    Parameters
    ----------
    bins : np.ndarray
        Candidate bin edges.
    var : np.ndarray
        Values being binned, one per jet.
    masks : list[np.ndarray]
        Boolean selections that all have to be populated in a kept bin.
    min_count : int
        Smallest per-bin count that still counts as populated.

    Returns
    -------
    np.ndarray | None
        The surviving edges, or ``None`` if no bin qualifies.
    """
    n_bins = len(bins) - 1
    index = np.digitize(var, bins) - 1
    inside = (index >= 0) & (index < n_bins)
    counts = np.stack([np.bincount(index[m & inside], minlength=n_bins) for m in masks])
    ok = np.asarray((counts >= min_count).all(axis=0))

    best_len = best_end = 0
    run = 0
    for i, good in enumerate(ok):
        run = run + 1 if good else 0
        if run > best_len:
            best_len, best_end = run, i + 1
    if best_len == 0:
        return None
    return bins[best_end - best_len : best_end + 1]


def _roc_one_vs_each(out_dir, scores, truth, sig, sig_eff, second_tag, stem) -> None:
    """ROC for one signal flavour: a SEPARATE rejection line for EACH other
    trained class (never merged). Discriminant = the signal node score p_sig.
    """
    sig_mask = truth == LABEL_IDS[sig]
    if sig_mask.sum() == 0:
        return
    d = scores[sig]
    sig_label = FLAV[sig][0]
    # No fixed ymax: puma autoscales to the curves; y_scale leaves headroom for
    # the tags (log-aware). Bump y_scale if the text overlaps the curves.
    rp = RocPlot(
        n_ratio_panels=0,
        ylabel="Background rejection",
        xlabel=f"{sig_label} efficiency",
        xmin=ROC_XMIN,
        xmax=1.0,
        y_scale=Y_SCALE,
        figsize=FIG,
        leg_fontsize=11,
        **_header(second_tag, int(truth.size)),
    )
    parts = []
    added = 0
    for bg in CLASSES:
        if bg == sig:
            continue
        m = truth == LABEL_IDS[bg]
        if m.sum() == 0:
            continue
        lbl, col = FLAV[bg]
        eff, rej = _bg_rej(d[sig_mask], d[m], sig_eff)
        if eff.size == 0:
            parts.append(f"{lbl}=n/a")
            continue
        rp.add_roc(
            Roc(
                eff,
                rej,
                n_test=int(m.sum()),
                rej_class=bg,
                signal_class=sig,
                label=lbl,
                colour=col,
                linestyle=LINESTYLE,
            )
        )
        _, r70 = _bg_rej(d[sig_mask], d[m], np.array([0.7]))
        parts.append(f"{lbl}={r70[0]:.0f}" if r70.size else f"{lbl}>{m.sum() / MIN_BG_PASSING:.0f}")
        added += 1
    if added == 0:
        return
    rp.draw()
    _save(rp, out_dir, stem)
    print(f"  -> {stem}   rej@70% eff: " + ", ".join(parts))


def _diff_plot(
    out_dir, sig, sig_mask, d, var, xlabel, var_bins, bgs, stem, second_tag="", wp=0.7
) -> None:
    """Background rejection vs a jet variable (pT, eta) at a flat per-bin WP.

    bgs = [(bg_mask, label, colour)]. Uses puma VarVsEff (flat_per_bin -> the
    signal efficiency is held at `wp` in every bin, so the rejection variation
    is the physics, not a shifting cut). ``d`` is the signal node score.
    """
    sig_label = FLAV[sig][0]
    plot = VarVsEffPlot(
        mode="bkg_rej",
        grid=False,
        n_ratio_panels=0,
        logy=True,
        ylabel="Background rejection",
        xlabel=xlabel,
        y_scale=Y_SCALE,
        figsize=FIG,
        leg_fontsize=11,
        **_header(
            second_tag,
            int(var.size),
            extra=f"{sig_label}, {int(wp * 100)}% efficiency per bin",
        ),
    )
    added = False
    for bg_mask, label, colour in bgs:
        if bg_mask.sum() == 0:
            continue
        plot.add(
            VarVsEff(
                x_var_sig=var[sig_mask],
                disc_sig=d[sig_mask],
                x_var_bkg=var[bg_mask],
                disc_bkg=d[bg_mask],
                bins=var_bins,
                working_point=wp,
                flat_per_bin=True,
                label=label,
                colour=colour,
                linestyle=LINESTYLE,
            )
        )
        added = True
    if not added:
        return
    plot.draw()
    _save(plot, out_dir, stem)
    print(f"  -> {stem}")


def _tasks_from_config(path: Path) -> dict:
    """Classification heads declared in a Salt training config.

    Reading the heads from the config rather than hard-coding them keeps the
    class names, the truth column and the label map identical to what the model
    was trained on -- ``jets_bccharge`` alone maps 225 PDG ids onto 5 classes.

    Parameters
    ----------
    path : Path
        The ``model_of_record_config.yaml`` written next to the checkpoint.

    Returns
    -------
    dict
        Task name -> its truth column, input group, class names and label map.
    """
    tasks: dict = {}
    stack = [yaml.safe_load(path.read_text())]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if "ClassificationTask" in str(node.get("class_path", "")):
                init = node["init_args"]
                tasks[init["name"]] = {
                    "label": init["label"],
                    "input": init["input_name"],
                    "classes": list(init["class_names"]),
                    "map": {int(k): int(v) for k, v in (init.get("label_map") or {}).items()},
                }
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return tasks


def _truth_classes(raw: np.ndarray, task: dict) -> np.ndarray:
    """Raw truth column turned into class indices, as Salt mapped them.

    Parameters
    ----------
    raw : np.ndarray
        The truth column straight out of the file.
    task : dict
        One entry of :func:`_tasks_from_config`.

    Returns
    -------
    np.ndarray
        Class indices. Values the map does not mention keep their own value,
        which is what Salt does; out-of-range ones are dropped by the caller.
    """
    out = raw.astype(np.int64)
    for src, dst in task["map"].items():
        # Compare against `raw`, never the partly-rewritten copy, so a value that
        # is both a source and a destination is not mapped twice.
        out = np.where(raw == src, dst, out)
    return out


def _predicted_classes(data: np.ndarray, tagger: str, task: dict) -> np.ndarray | None:
    """Argmax over a head's score columns, or None when they are absent.

    Parameters
    ----------
    data : np.ndarray
        Structured array holding the score columns.
    tagger : str
        Tagger prefix the columns are named after.
    task : dict
        One entry of :func:`_tasks_from_config`.

    Returns
    -------
    np.ndarray | None
        Predicted class index per row, or None if the head was not written out.
    """
    names = data.dtype.names or ()
    cols = [f"{tagger}_p{cls}" for cls in task["classes"]]
    if any(col not in names for col in cols):
        return None
    return np.stack([data[col] for col in cols], axis=-1).argmax(axis=-1)


def _confusion_plot(
    out_dir: Path,
    targets: np.ndarray,
    preds: np.ndarray,
    classes: list[str],
    stem: str,
    header: dict,
) -> None:
    """Row-normalised confusion matrix: what each truth class gets called.

    Parameters
    ----------
    out_dir : Path
        Directory the PNG is written to.
    targets : np.ndarray
        Truth class index per entry.
    preds : np.ndarray
        Predicted class index per entry.
    classes : list[str]
        Class names, in label order.
    stem : str
        Output file name, without a suffix.
    header : dict
        Badge and tag keywords from :func:`_header`.
    """
    size = len(classes)
    keep = (targets >= 0) & (targets < size)
    if not keep.any():
        print(f"  -- {stem}: no entry carries a valid label, skipped")
        return
    # Counted here rather than with puma's helper, which sizes the matrix from
    # the labels it is handed: a class that is never the truth but sometimes the
    # prediction would then fall outside it.
    pairs = targets[keep].astype(np.int64) * size + preds[keep]
    matrix = np.bincount(pairs, minlength=size * size).reshape(size, size).astype(float)
    rows = matrix.sum(axis=1, keepdims=True)
    matrix = np.divide(matrix, rows, out=np.zeros_like(matrix), where=rows > 0)
    plot = MatshowPlot(
        x_ticklabels=classes,
        y_ticklabels=classes,
        x_ticks_rotation=45,
        show_entries=True,
        show_percentage=True,
        cbar_label="Fraction of truth class",
        xlabel="Predicted class",
        ylabel="Truth class",
        figsize=FIG_MATRIX,
        # A matrix fills its axes edge to edge, so the tags have to go above it
        # rather than float over the top row.
        atlas_tag_outside=True,
        **header,
    )
    plot.draw(matrix)
    _save(plot, out_dir, stem)
    diagonal = float(np.mean(np.diag(matrix)))
    print(f"  -> {stem}   mean diagonal = {diagonal:.3f} over {int(keep.sum())} entries")


def _track_predictions(
    preds_path: str, tagger: str, tasks: dict, n_jets: int, chunk: int = 20_000
) -> dict:
    """Per-track truth and predictions, accumulated a block of jets at a time.

    The pfcand group is (n_jets, 75) of 222-byte rows, so the whole thing is far
    larger than the jets group. Reducing each block to class indices as it is
    read keeps only what the plots need.

    Parameters
    ----------
    preds_path : str
        Path to the Salt prediction file.
    tagger : str
        Tagger prefix the score columns are named after.
    tasks : dict
        Output of :func:`_tasks_from_config`.
    n_jets : int
        Number of jets to read.
    chunk : int
        Jets per read.

    Returns
    -------
    dict
        ``valid`` mask plus truth/prediction pairs for each track-level head
        that is present in the file.
    """
    wanted = {k: v for k, v in tasks.items() if v["input"] == "pfcand" and v["classes"]}
    parts: dict = {}
    with h5py.File(preds_path, "r") as f:
        tracks = f["pfcand"]
        names = tracks.dtype.names or ()
        n_jets = min(n_jets, tracks.shape[0])
        has_vtx = "VertexIndex" in names and "pfcand_truthVertexIndex" in names
        for start in range(0, n_jets, chunk):
            block = tracks[start : start + chunk]
            got = {"valid": np.asarray(block["valid"])}
            for name, task in wanted.items():
                pred = _predicted_classes(block, tagger, task)
                if pred is None:
                    continue
                truth = _truth_classes(np.asarray(block[task["label"]]), task)
                got[f"{name}_pred"] = pred.astype(np.int8)
                got[f"{name}_truth"] = truth.astype(np.int8)
            if has_vtx:
                got["vtx_reco"] = np.asarray(block["VertexIndex"]).astype(np.int16)
                got["vtx_truth"] = np.asarray(block["pfcand_truthVertexIndex"]).astype(np.int16)
            for key, value in got.items():
                parts.setdefault(key, []).append(value)
    return {k: np.concatenate(v) for k, v in parts.items()}


def _vertex_plots(out_dir: Path, tracks: dict, jet_pt: np.ndarray | None, second_tag: str) -> None:
    """Vertex efficiency, purity and fake rate against jet pT.

    Truth and reco vertices are cleaned with puma's own helpers first, so the
    numbers mean the same thing they do in the ATLAS aux-task plots: truth
    vertices keep only heavy-flavour tracks, and a reco vertex needs at least
    one track the model calls heavy flavour.

    Parameters
    ----------
    out_dir : Path
        Directory the PNGs are written to.
    tracks : dict
        Output of :func:`_track_predictions`.
    jet_pt : np.ndarray | None
        Jet pT, trimmed to the same jets.
    second_tag : str
        Free-form tag line.
    """
    if "vtx_reco" not in tracks or jet_pt is None:
        print("  -- vertexing: no VertexIndex/truth pair in the file, skipped")
        return
    origin = tracks.get("track_origin_truth")
    reco_origin = tracks.get("track_origin_pred")
    if origin is None:
        print("  -- vertexing: needs track_origin truth to define HF vertices, skipped")
        return

    n = len(tracks["vtx_reco"])
    truth_idx = np.copy(tracks["vtx_truth"])
    reco_idx = np.copy(tracks["vtx_reco"])
    for i in range(n):
        truth_idx[i] = clean_truth_vertices(truth_idx[i], origin[i], incl_vertexing=False)
        reco_idx[i] = clean_reco_vertices(
            reco_idx[i],
            None if reco_origin is None else reco_origin[i],
            incl_vertexing=False,
        )
    metrics = calculate_vertex_metrics(reco_idx, truth_idx)

    pt = jet_pt[:n]
    bins = _longest_usable_bins(
        np.array([5, 15, 25, 35, 45, 55, 70.0]), pt, [np.ones(n, dtype=bool)]
    )
    if bins is None:
        print("  -- vertexing: no populated pT bin, skipped")
        return
    for mode, ylabel in (
        ("efficiency", "Vertex efficiency"),
        ("purity", "Vertex purity"),
        ("fakes", "Vertex fake rate"),
    ):
        plot = VarVsVtxPlot(
            mode=mode,
            ylabel=ylabel,
            xlabel="Jet $p_T$ [GeV]",
            y_scale=Y_SCALE,
            figsize=FIG,
            **_header(second_tag, n, extra="Track vertexing, HF vertices"),
        )
        plot.add(
            VarVsVtx(
                x_var=pt,
                n_match=metrics["n_match"],
                n_true=metrics["n_ref"],
                n_reco=metrics["n_test"],
                bins=bins,
                label="GN3",
                colour=FLAV["bjets"][1],
                linestyle=LINESTYLE,
            ),
            reference=True,
        )
        plot.draw()
        _save(plot, out_dir, f"vtx_{mode}_vs_pt")
        print(f"  -> vtx_{mode}_vs_pt")


def main() -> int:
    """Write the puma plots for a higgs_decay Salt prediction file.

    Returns
    -------
    int
        Process exit status.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="Salt prediction H5 (with flavour_label truth)")
    ap.add_argument("--out", required=True, help="Output dir for PNGs")
    ap.add_argument("--tagger", default=None, help="Tagger prefix (auto-detect if omitted)")
    ap.add_argument("--n", type=int, default=None, help="Optional cap on number of jets")
    ap.add_argument("--second-tag", default="")
    ap.add_argument(
        "--model-config",
        default=None,
        help="Salt training config; defaults to model_of_record_config.yaml next to --preds",
    )
    ap.add_argument(
        "--n-tracks",
        type=int,
        default=200_000,
        help="Jets used for the track-level plots (the pfcand group is ~100x the jets group)",
    )
    ap.add_argument(
        "--n-vertex",
        type=int,
        default=50_000,
        help="Jets used for the vertexing plots; puma cleans vertices one jet at a time",
    )
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.preds, "r") as f:
        dataset = f["jets"]
        n = dataset.shape[0] if args.n is None else min(args.n, dataset.shape[0])
        # One read of whole rows. h5py serves a per-field selection with its own
        # pass over the file, so pulling the ten columns used below one at a time
        # costs ~10x this -- and reading `dataset[col][:]` ignored --n entirely.
        jets = dataset[:n]

    truth = jets["flavour_label"].astype(np.int32)
    tagger, scores = _find_class_scores(jets, args.tagger)
    nm = jets.dtype.names or ()
    jet_pt = jets["jet_pt"] if "jet_pt" in nm else None
    jet_eta = jets["jet_eta"] if "jet_eta" in nm else None

    print(f"Jets used: {n}")
    print("Truth: " + ", ".join(f"{c}={int((truth == LABEL_IDS[c]).sum())}" for c in CLASSES))

    # --- argmax accuracy (quick numeric check) ---
    score_mat = np.stack([scores[c] for c in CLASSES], axis=1)  # (n, 7)
    pred = score_mat.argmax(axis=1)
    print("\nPer-class accuracy (argmax of the 7 scores):")
    for c in CLASSES:
        m = truth == LABEL_IDS[c]
        if m.sum():
            print(f"  {c:8s} eff={float((pred[m] == LABEL_IDS[c]).mean()):.3f}  (n={int(m.sum())})")
    print(f"  overall accuracy = {float((pred == truth).mean()):.3f}")

    _confusion_plot(
        out_dir,
        truth,
        pred,
        [FLAV[c][0] for c in CLASSES],
        "confmat_flavour",
        _header(args.second_tag, n, extra="Jet flavour, argmax of the 7 scores"),
    )

    # --- score histograms: one per output node, all 7 truth classes overlaid ---
    # logy + no ylim -> puma autoscales the y-axis.
    bins = np.linspace(0, 1, 50)
    for node in CLASSES:
        sc = scores[node]
        short = node[:-4]
        hp = HistogramPlot(
            ylabel="Normalised number of jets",
            xlabel=f"$p_{{{NODE_MATH[node]}}}$",
            logy=True,
            y_scale=Y_SCALE,
            figsize=FIG,
            leg_fontsize=11,
            **_header(args.second_tag, n),
        )
        for tru in CLASSES:
            m = truth == LABEL_IDS[tru]
            if m.sum() < 10:
                continue
            lbl, col = FLAV[tru]
            hp.add(
                Histogram(sc[m], bins=bins, norm=True, label=lbl, colour=col, linestyle=LINESTYLE)
            )
        hp.draw()
        _save(hp, out_dir, f"score_p{short}")
        print(f"  -> score_p{short}")

    # --- ROCs: one per signal flavour, one line per other class (6 lines) ---
    sig_eff = np.linspace(0.5, 1.0, 100)
    for sig in CLASSES:
        _roc_one_vs_each(
            out_dir, scores, truth, sig, sig_eff, args.second_tag, f"roc_{sig[:-4]}tag"
        )

    # --- differential: background rejection vs jet pT and eta at flat 70% WP.
    # Signal node score as discriminant; here we DO fold u+d+g into a grey
    # "light" reference line (a differential plot, not the per-class ROCs above).
    is_light = np.isin(truth, [LABEL_IDS[c] for c in ("ujets", "djets", "gjets")])
    b = truth == LABEL_IDS["bjets"]
    c = truth == LABEL_IDS["cjets"]
    s = truth == LABEL_IDS["sjets"]
    light = (is_light, "Light ($u$+$d$+$g$)", "#7f7f7f")
    as_bg = {f: (truth == LABEL_IDS[f], *FLAV[f]) for f in ("bjets", "cjets")}
    diff_sets = [
        ("bjets", b, [light, as_bg["cjets"]]),
        ("cjets", c, [light, as_bg["bjets"]]),
        ("sjets", s, [light, as_bg["cjets"]]),
    ]
    for var, wanted, xlabel, axis in (
        (jet_pt, np.array([5, 15, 25, 35, 45, 55, 70, 90.0]), "Jet $p_T$ [GeV]", "pt"),
        (jet_eta, np.linspace(-4, 4, 9), "Jet $\\eta$", "eta"),
    ):
        if var is None:
            continue
        for sig, sm, bgs in diff_sets:
            var_bins = _longest_usable_bins(wanted, var, [sm, *(m for m, _, _ in bgs)])
            if var_bins is None:
                print(f"  -- rej_vs_{axis}_{sig[:-4]}tag: no bin has enough jets, skipped")
                continue
            _diff_plot(
                out_dir,
                sig,
                sm,
                scores[sig],
                var,
                xlabel,
                var_bins,
                bgs,
                f"rej_vs_{axis}_{sig[:-4]}tag",
                second_tag=args.second_tag,
            )

    # --- auxiliary heads: track origin, track type, vertexing, b/c charge ---
    config_path = (
        Path(args.model_config)
        if args.model_config
        else Path(args.preds).parent / "model_of_record_config.yaml"
    )
    if not config_path.is_file():
        print(f"\nNo training config at {config_path}; skipping the auxiliary-head plots.")
        print(f"\nPlots in: {out_dir}")
        return 0

    tasks = _tasks_from_config(config_path)
    print(f"\nAuxiliary heads from {config_path.name}: {', '.join(sorted(tasks))}")

    for name, task in tasks.items():
        if task["input"] != "jets" or task["label"] == "flavour_label":
            continue
        head_pred = _predicted_classes(jets, tagger, task)
        if head_pred is None:
            print(f"  -- {name}: scores not in the file, skipped")
            continue
        extra = f"{name}, argmax of the {len(task['classes'])} scores"
        _confusion_plot(
            out_dir,
            _truth_classes(jets[task["label"]], task),
            head_pred,
            task["classes"],
            f"confmat_{name}",
            _header(args.second_tag, n, extra=extra),
        )

    n_tracks = min(args.n_tracks, n)
    tracks = _track_predictions(args.preds, tagger, tasks, n_tracks)
    valid = tracks["valid"]
    for name, task in tasks.items():
        if task["input"] != "pfcand" or f"{name}_pred" not in tracks:
            continue
        _confusion_plot(
            out_dir,
            tracks[f"{name}_truth"][valid],
            tracks[f"{name}_pred"][valid],
            task["classes"],
            f"confmat_{name}",
            _header(args.second_tag, n_tracks, extra=f"{name}, valid tracks only"),
        )

    n_vertex = min(args.n_vertex, n_tracks)
    _vertex_plots(
        out_dir,
        {k: v[:n_vertex] for k, v in tracks.items()},
        jet_pt,
        args.second_tag,
    )

    print(f"\nPlots in: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
