"""Concatenate multiple Salt-shaped H5 files into one, streaming row-wise."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from fcc_ftag_converter.report import ReportAccumulator, write_conversion_report

_COPY_CHUNK = 100_000
# Datasets the converter only resizes to keep the leading dimension aligned with
# `jets`; they hold nothing but fill values, so they are grown rather than copied.
# Everything else is real payload — listing the fill-only names instead of the
# payload names keeps streams the EDM4hep path does not know about (`pfcand` on
# the higgs_decay path) from being silently dropped.
_FILL_ONLY = ("calo",)
# Streams the report accumulator understands; other payload datasets are merged
# but do not contribute to the summary statistics.
_ACCUMULATED = ("jets", "tracks_ghost", "flows", "electrons", "muons")


def _dataset_specs(handle: h5py.File) -> dict[str, dict]:
    specs: dict[str, dict] = {}
    for name in handle:
        obj = handle[name]
        if not isinstance(obj, h5py.Dataset):
            continue
        specs[name] = {
            "dtype": obj.dtype,
            "ndim": obj.ndim,
            "shape_tail": tuple(obj.shape[1:]),
            "fillvalue": obj.fillvalue,
            "compression": obj.compression,
            "compression_opts": obj.compression_opts,
        }
    return specs


def _verify_compatible(
    reference_specs: dict[str, dict], path: Path, current_specs: dict[str, dict]
) -> None:
    missing = set(reference_specs) - set(current_specs)
    extra = set(current_specs) - set(reference_specs)
    if missing or extra:
        raise ValueError(
            f"{path} has incompatible datasets (missing: {sorted(missing)}, extra: {sorted(extra)})"
        )
    for name, ref in reference_specs.items():
        cur = current_specs[name]
        if ref["dtype"] != cur["dtype"]:
            raise ValueError(f"{path}: dtype mismatch for {name!r}")
        if ref["shape_tail"] != cur["shape_tail"]:
            raise ValueError(f"{path}: tail-shape mismatch for {name!r}")


def merge_h5_files(
    inputs: Sequence[str | Path],
    output: str | Path,
    *,
    copy_chunk: int = _COPY_CHUNK,
    accumulator: ReportAccumulator | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Concatenate Salt-shaped H5 inputs into ``output``.

    All inputs must share dataset names, dtypes, and tail shapes. Compression
    settings are inherited from the first input. If an ``accumulator`` is
    given it is updated chunk-by-chunk during the copy so a merged summary
    report can be written without re-reading the output.

    Parameters
    ----------
    inputs : Sequence[str | Path]
        Salt-shaped H5 files to concatenate, in order.
    output : str | Path
        Destination path; parent directories are created.
    copy_chunk : int, optional
        Rows copied per read/write step.
    accumulator : ReportAccumulator | None, optional
        Updated during the copy when a merged report is wanted.
    verbose : bool, optional
        Print one line per merged input.

    Returns
    -------
    dict[str, Any]
        ``{"n_inputs", "n_jets", "sources"}`` for the merged output.

    Raises
    ------
    ValueError
        If ``inputs`` is empty or the inputs disagree on dataset layout.
    """
    paths = [Path(p) for p in inputs]
    if not paths:
        raise ValueError("no inputs given")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(paths[0], "r") as first:
        ref_specs = _dataset_specs(first)
        attrs = dict(first.attrs)

    if not ref_specs:
        raise ValueError(f"{paths[0]} has no top-level datasets")

    payload_specs = {name: spec for name, spec in ref_specs.items() if name not in _FILL_ONLY}
    placeholder_names = sorted(set(ref_specs) - set(payload_specs))

    diagnostics_per_file: list[dict[str, int]] = []
    sources: list[str] = []

    with h5py.File(output, "w") as out_handle:
        for key, value in attrs.items():
            out_handle.attrs[key] = value
        out_handle.attrs["producer"] = "fcc-ftag-merge"
        out_handle.attrs["merged_inputs"] = [str(p) for p in paths]

        for name, spec in ref_specs.items():
            shape = (0, *spec["shape_tail"])
            maxshape = (None, *spec["shape_tail"])
            chunks_arg = (2048, *spec["shape_tail"]) if spec["shape_tail"] else (2048,)
            out_handle.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                chunks=chunks_arg,
                dtype=spec["dtype"],
                fillvalue=spec["fillvalue"],
                compression=spec["compression"],
                compression_opts=spec["compression_opts"],
            )

        total_jets = 0
        for input_path in paths:
            with h5py.File(input_path, "r") as src:
                current_specs = _dataset_specs(src)
                _verify_compatible(ref_specs, input_path, current_specs)
                sources.append(str(input_path))
                file_diag = src.attrs.get("diagnostics", None)
                if file_diag is not None:
                    diagnostics_per_file.append(dict(file_diag))

                n_src = int(src["jets"].shape[0]) if "jets" in src else 0
                if n_src == 0:
                    continue
                total_jets += n_src

                acc_buffers: dict[str, list[np.ndarray]] = {}

                for name in payload_specs:
                    src_ds = src[name]
                    out_ds = out_handle[name]
                    start_total = int(out_ds.shape[0])
                    new_size = start_total + n_src
                    out_ds.resize((new_size, *out_ds.shape[1:]))
                    for i in range(0, n_src, copy_chunk):
                        j = min(i + copy_chunk, n_src)
                        block = src_ds[i:j]
                        out_ds[start_total + i : start_total + j] = block
                        if accumulator is not None and name in _ACCUMULATED:
                            acc_buffers.setdefault(name, []).append(block)

                for name in placeholder_names:
                    out_ds = out_handle[name]
                    new_size = int(out_ds.shape[0]) + n_src
                    out_ds.resize((new_size, *out_ds.shape[1:]))

                if accumulator is not None:
                    # higgs_decay outputs carry only `jets` + `pfcand`, so the
                    # ATLAS-shaped streams are absent; pass empty stand-ins for
                    # the ones this file does not have.
                    empty = np.empty(0)
                    stacked = {
                        name: np.concatenate(blocks, axis=0) for name, blocks in acc_buffers.items()
                    }
                    accumulator.update(*(stacked.get(name, empty) for name in _ACCUMULATED))

            if verbose:
                print(f"merged {input_path} → {output} (total jets: {total_jets})")

    return {"n_inputs": len(paths), "n_jets": total_jets, "sources": sources}


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fcc-ftag-merge`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="fcc-ftag-merge",
        description="Concatenate Salt-shaped H5 files into one.",
    )
    parser.add_argument("inputs", nargs="+", help="Input H5 files (output of fcc-ftag-convert).")
    parser.add_argument("-o", "--output", required=True, help="Merged output H5 path.")
    parser.add_argument("--report", action="store_true", help="Write a merged summary report.")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``fcc-ftag-merge``.

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
    accumulator = ReportAccumulator() if args.report else None
    result = merge_h5_files(
        args.inputs,
        args.output,
        accumulator=accumulator,
        verbose=not args.quiet,
    )
    print(f"wrote {args.output} ({result['n_jets']} jets from {result['n_inputs']} files)")
    if accumulator is not None:
        write_conversion_report(
            Path(args.output),
            accumulator,
            inputs=result["sources"],
            max_events=None,
            missing_collections=set(),
            lepton_match_mode="(merged)",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
