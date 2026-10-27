r"""Per-class recall of every classification head, several trained configs side by side.

``puma_plot_higgs_decay.py`` draws one row-normalised confusion matrix per head
per model. Those matrices are the right object to read a single model with, but
two heatmaps cannot be overlaid, so comparing models means flipping between
images and eyeballing cells.

The comparable quantity inside a row-normalised matrix is its diagonal: the
fraction of each truth class the model calls correctly. This script extracts it
for every head of every model and writes

* ``confusion_diagonals.csv`` -- one row per (head, class, model), with the
  truth-class population behind each number, and
* ``confrecall_<head>.png/pdf`` -- the same numbers as grouped bars.

The heads, their truth columns and their label maps are read from each run's own
resolved config, so a model is always scored the way it was trained -- which
matters for ``jets_bccharge``, where 225 PDG ids fold onto 5 classes.

The interesting head here is ``track_type``: its Pion and Kaon rows say
whether a model actually uses the hadron-identification inputs, independently of
the jet-level ROCs.

Usage:
  puma_compare_confmats.py --out DIR \
      --preds LABEL=path/to/preds.h5 --preds LABEL=... \
      [--n-tracks 200000] [--second-tag TEXT]

Each ``--preds`` takes ``LABEL=PATH``. The run config is looked up as
``<run>/config.yaml`` next to the prediction file's ``ckpts`` directory; pass
``LABEL=PATH:CONFIG`` to point somewhere else.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Same reasoning as puma_compare_configs: the head definitions, the label
# mapping and the track selection have to match the single-model plots of this
# analysis exactly, so they are imported rather than restated.
from puma_compare_configs import MODEL_COLOURS
from puma_plot_higgs_decay import (
    _find_class_scores,  # noqa: PLC2701
    _predicted_classes,  # noqa: PLC2701
    _tasks_from_config,  # noqa: PLC2701
    _track_predictions,  # noqa: PLC2701
    _truth_classes,  # noqa: PLC2701
)


def _diagonal(truth: np.ndarray, pred: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Row-normalised diagonal of a confusion matrix, and the row populations.

    Parameters
    ----------
    truth : np.ndarray
        Truth class index per entry.
    pred : np.ndarray
        Predicted class index per entry.
    n_classes : int
        Number of classes the head declares.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Recall per class (NaN where the class never occurs), and the number of
        entries in each truth class.
    """
    keep = (truth >= 0) & (truth < n_classes)
    t, p = truth[keep].astype(np.int64), pred[keep].astype(np.int64)
    pairs = t * n_classes + p
    matrix = np.bincount(pairs, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    rows = matrix.sum(axis=1)
    diag = np.divide(
        np.diag(matrix).astype(float), rows, out=np.full(n_classes, np.nan), where=rows > 0
    )
    return diag, rows


def _load(spec: str, n_tracks: int) -> tuple[str, dict]:
    """Per-head recall of one model.

    Parameters
    ----------
    spec : str
        ``LABEL=PATH`` or ``LABEL=PATH:CONFIG``.
    n_tracks : int
        Jets to read for the constituent-level heads.

    Returns
    -------
    tuple[str, dict]
        The label, and head name -> ``{"classes", "recall", "n"}``.

    Raises
    ------
    ValueError
        If the spec has no ``LABEL=``, or the run config cannot be located.
    """
    if "=" not in spec:
        raise ValueError(f"--preds needs LABEL=PATH, got {spec!r}")
    label, rest = spec.split("=", 1)
    if ":" in rest:
        path, config = rest.split(":", 1)
        config_path = Path(config)
    else:
        path = rest
        # <run>/ckpts/<ckpt>__test_*.h5 -> <run>/config.yaml
        config_path = Path(path).parent.parent / "config.yaml"
    if not config_path.is_file():
        raise ValueError(f"no run config at {config_path}; pass LABEL=PATH:CONFIG")

    with h5py.File(path, "r") as f:
        jets = f["jets"][:]
    tagger, _scores = _find_class_scores(jets, None)
    tasks = _tasks_from_config(config_path)

    out: dict = {}
    for name, task in tasks.items():
        if task["input"] != "jets":
            continue
        pred = _predicted_classes(jets, tagger, task)
        if pred is None:
            continue
        truth = _truth_classes(jets[task["label"]], task)
        recall, n = _diagonal(truth, pred, len(task["classes"]))
        out[name] = {"classes": task["classes"], "recall": recall, "n": n}

    if any(t["input"] == "pfcand" for t in tasks.values()):
        tracks = _track_predictions(path, tagger, tasks, min(n_tracks, jets.shape[0]))
        valid = tracks["valid"]
        for name, task in tasks.items():
            if task["input"] != "pfcand" or f"{name}_pred" not in tracks:
                continue
            recall, n = _diagonal(
                tracks[f"{name}_truth"][valid],
                tracks[f"{name}_pred"][valid],
                len(task["classes"]),
            )
            out[name] = {"classes": task["classes"], "recall": recall, "n": n}

    print(f"  {label:<20s} tagger={tagger}  heads={', '.join(sorted(out))}")
    return label, out


def _bars(out_dir: Path, head: str, models: list[tuple[str, dict]], second_tag: str) -> None:
    """Grouped bar chart of per-class recall, one colour per model.

    Parameters
    ----------
    out_dir : Path
        Directory the figure is written to.
    head : str
        Head name, e.g. ``"track_type"``.
    models : list[tuple[str, dict]]
        ``(label, per-head results)`` for the models that have this head.
    second_tag : str
        Free-form line under the title.
    """
    present = [(lab, res[head]) for lab, res in models if head in res]
    if not present:
        return
    classes = present[0][1]["classes"]
    x = np.arange(len(classes), dtype=float)
    width = 0.8 / len(present)

    fig, ax = plt.subplots(figsize=(1.6 + 1.15 * len(classes), 4.2))
    for i, (lab, res) in enumerate(present):
        ax.bar(
            x + (i - (len(present) - 1) / 2) * width,
            np.nan_to_num(res["recall"], nan=0.0),
            width,
            label=lab,
            color=MODEL_COLOURS[i % len(MODEL_COLOURS)],
        )
    ax.set_xticks(x)
    # the population is a property of the data, not of a model, so it is shown
    # once per class rather than per bar
    ax.set_xticklabels(
        [f"{c}\n{present[0][1]['n'][j]:,}" for j, c in enumerate(classes)], fontsize=8
    )
    ax.set_ylabel("Fraction of truth class called correctly")
    ax.set_ylim(0, 1.05)
    ax.set_axisbelow(True)
    ax.grid(axis="y", lw=0.4, alpha=0.5)
    ax.legend(fontsize=8, ncols=min(len(present), 3))
    ax.set_title(f"{head}\n{second_tag}" if second_tag else head, fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"confrecall_{head}.{ext}", dpi=200)
    plt.close(fig)
    print(f"  -> confrecall_{head}")


def main() -> int:
    """Write the per-class recall table and one bar chart per head.

    Returns
    -------
    int
        Process exit status.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preds", action="append", required=True, help="LABEL=path[:config.yaml]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-tracks", type=int, default=200_000, help="jets for the track-level heads")
    ap.add_argument("--second-tag", default="")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Loading predictions:")
    models = [_load(s, args.n_tracks) for s in args.preds]

    heads: list[str] = []
    for _lab, res in models:
        heads.extend(h for h in res if h not in heads)

    rows = []
    for head in heads:
        for lab, res in models:
            if head not in res:
                continue
            r = res[head]
            for j, cls in enumerate(r["classes"]):
                rows.append({
                    "head": head,
                    "class": cls,
                    "model": lab,
                    "recall": "" if np.isnan(r["recall"][j]) else f"{r['recall'][j]:.4f}",
                    "n_truth": int(r["n"][j]),
                })
    csv_path = out_dir / "confusion_diagonals.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nFigures and {csv_path.name}:")
    for head in heads:
        _bars(out_dir, head, models, args.second_tag)

    for head in heads:
        present = [(lab, res[head]) for lab, res in models if head in res]
        if not present:
            continue
        print(f"\n  {head} -- fraction of each truth class called correctly:")
        print("    " + "class".ljust(16) + "".join(f"{lab:>20s}" for lab, _ in present))
        for j, cls in enumerate(present[0][1]["classes"]):
            cells = [
                "n/a" if np.isnan(res["recall"][j]) else f"{res['recall'][j]:.3f}"
                for _lab, res in present
            ]
            print("    " + cls.ljust(16) + "".join(f"{c:>20s}" for c in cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
