from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import h5py
import numpy as np


def _values(handle: h5py.File, dataset: str, variable: str) -> np.ndarray:
    if dataset not in handle:
        return np.asarray([], dtype=np.float64)
    data = handle[dataset]
    if variable not in (data.dtype.names or ()):
        return np.asarray([], dtype=np.float64)

    values = data[variable][...]
    if values.ndim > 1 and "valid" in (data.dtype.names or ()):
        valid = data["valid"][...].astype(bool)
        values = values[valid] if valid.shape == values.shape else values.reshape(-1)
    else:
        values = values.reshape(-1)

    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _variables(handle_a: h5py.File, handle_b: h5py.File) -> list[tuple[str, str]]:
    def _fields(handle: h5py.File, name: str) -> tuple[str, ...]:
        if name not in handle:
            return ()
        obj = handle[name]
        if not isinstance(obj, h5py.Dataset):
            return ()
        return obj.dtype.names or ()

    pairs: list[tuple[str, str]] = []
    for dataset in sorted(set(handle_a.keys()) | set(handle_b.keys())):
        names_a = _fields(handle_a, dataset)
        names_b = _fields(handle_b, dataset)
        for variable in sorted(set(names_a) & set(names_b)):
            if variable == "valid":
                continue
            pairs.append((dataset, variable))
    return pairs


def _parse_variables(items: list[str] | None) -> list[tuple[str, str]] | None:
    if not items:
        return None
    pairs = []
    for item in items:
        dataset, sep, variable = item.partition(".")
        if not sep:
            raise ValueError(f"variable must look like dataset.variable, got {item!r}")
        pairs.append((dataset, variable))
    return pairs


def _summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return dict.fromkeys(("min", "p50", "p95", "max", "mean", "zero_frac"), math.nan)
    return {
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "zero_frac": float(np.mean(values == 0)),
    }


def _ks(a_values: np.ndarray, b_values: np.ndarray) -> float:
    if a_values.size == 0 or b_values.size == 0:
        return math.nan
    a_values = np.sort(a_values)
    b_values = np.sort(b_values)
    points = np.sort(np.concatenate([a_values, b_values]))
    a_cdf = np.searchsorted(a_values, points, side="right") / a_values.size
    b_cdf = np.searchsorted(b_values, points, side="right") / b_values.size
    return float(np.max(np.abs(a_cdf - b_cdf)))


def compare_h5(
    sample_a: str | Path,
    sample_b: str | Path,
    output_dir: str | Path,
    *,
    label_a: str = "sample A",
    label_b: str = "sample B",
    variables: list[tuple[str, str]] | None = None,
) -> None:
    """Write normalised A/B comparison histograms into ``output_dir``.

    Parameters
    ----------
    sample_a : str | Path
        First Salt-shaped H5 file.
    sample_b : str | Path
        Second Salt-shaped H5 file.
    output_dir : str | Path
        Destination for the PNGs and the summary table; created if absent.
    label_a : str, optional
        Legend label for ``sample_a``.
    label_b : str, optional
        Legend label for ``sample_b``.
    variables : list[tuple[str, str]] | None, optional
        ``(dataset, variable)`` pairs to plot. Defaults to the built-in set.
    """
    # Imported here so the Agg backend is selected before pyplot is first
    # touched, and so the CLI's other subcommands do not pay for matplotlib.
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    with h5py.File(sample_a, "r") as handle_a, h5py.File(sample_b, "r") as handle_b:
        variables = variables or _variables(handle_a, handle_b)
        for dataset, variable in variables:
            a_values = _values(handle_a, dataset, variable)
            b_values = _values(handle_b, dataset, variable)
            png = output_dir / f"{dataset}.{variable}.png"

            fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
            finite = [values for values in (a_values, b_values) if values.size]
            if finite:
                combined = np.concatenate(finite)
                low = float(np.quantile(combined, 0.005))
                high = float(np.quantile(combined, 0.995))
                if not np.isfinite(low) or not np.isfinite(high) or low == high:
                    low = float(np.min(combined))
                    high = float(np.max(combined))
                if low == high:
                    low -= 0.5
                    high += 0.5
                bins = np.linspace(low, high, 50)
                for values, label, color in (
                    (a_values, label_a, "#3366cc"),
                    (b_values, label_b, "#dc3912"),
                ):
                    if values.size:
                        weights = np.ones(values.size) / values.size
                        ax.hist(
                            values,
                            bins=bins.tolist(),
                            weights=weights,
                            histtype="step",
                            linewidth=1.8,
                            label=f"{label} (n={values.size})",
                            color=color,
                        )
            ax.set_title(f"{dataset}.{variable}")
            ax.set_xlabel(variable)
            ax.set_ylabel("normalized entries")
            ax.legend(loc="best")
            ax.grid(alpha=0.25)
            fig.savefig(png, dpi=140)
            plt.close(fig)

            a_summary = _summary(a_values)
            b_summary = _summary(b_values)
            rows.append({
                "dataset": dataset,
                "variable": variable,
                "entries_a": int(a_values.size),
                "entries_b": int(b_values.size),
                "ks_distance": _ks(a_values, b_values),
                **{f"a_{key}": value for key, value in a_summary.items()},
                **{f"b_{key}": value for key, value in b_summary.items()},
                "histogram": str(png),
            })

    if rows:
        with (output_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    with (output_dir / "summary.md").open("w") as handle:
        handle.write("# H5 Comparison\n\n")
        handle.write("Histograms are normalized independently for each sample.\n\n")
        handle.write("| dataset | variable | entries A | entries B | KS |\n")
        handle.write("|---|---|---:|---:|---:|\n")
        for row in rows:
            handle.write(
                f"| `{row['dataset']}` | `{row['variable']}` | "
                f"{row['entries_a']} | {row['entries_b']} | {row['ks_distance']:.3g} |\n"
            )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fcc-ftag-compare`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Make normalized comparison histograms for two H5 files."
    )
    parser.add_argument("sample_a")
    parser.add_argument("sample_b")
    parser.add_argument("-o", "--output-dir", default="plots/compare")
    parser.add_argument("--label-a", default="sample A")
    parser.add_argument("--label-b", default="sample B")
    parser.add_argument(
        "--variable",
        action="append",
        help="Restrict to one dataset.variable; may be given multiple times.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``fcc-ftag-compare``.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command line, defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    compare_h5(
        args.sample_a,
        args.sample_b,
        args.output_dir,
        label_a=args.label_a,
        label_b=args.label_b,
        variables=_parse_variables(args.variable),
    )
    print(f"wrote comparison plots to {Path(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
