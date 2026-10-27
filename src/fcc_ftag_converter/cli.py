"""CLI entry point and streaming orchestrator."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fcc_ftag_converter.reader import build_rows_from_chunk
from fcc_ftag_converter.report import ReportAccumulator, write_conversion_report
from fcc_ftag_converter.schema import (
    FCC_TRAINING_DATASETS,
    GLOBAL_DTYPE,
    SaltH5Writer,
    _load_reference_schema,
    default_schema_path,
)
from fcc_ftag_converter.streaming import iter_event_chunks

DEFAULT_REFERENCE = str(default_schema_path())
DEFAULT_OUTPUT = "outputs/converted.h5"
DEFAULT_CHUNK_SIZE = 10_000
_FILLED = ("global", "jets", "tracks_ghost", "tracks_ghost_rich_ip", "flows", "electrons", "muons")


def convert_fcc_edm4hep_to_h5(
    inputs: list[str],
    output: str | Path,
    *,
    reference: str | Path,
    max_events: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_tracks: int = 75,
    max_flows: int = 100,
    max_electrons: int = 10,
    max_muons: int = 5,
    magnetic_field_t: float = 2.0,
    truth_cone_radius: float = 1.5,
    hadron_pt_cut_gev: float = 5.0,
    lepton_match_mode: str = "constituent",
    lepton_match_dr: float = 0.4,
    write_report: bool = True,
    quiet: bool = False,
    compression: str = "gzip",
    compression_opts: int = 7,
) -> None:
    """Stream EDM4hep events from ROOT into a Salt-shaped HDF5 file.

    Events are read in chunks of ``chunk_size`` via ``uproot.iterate`` and
    written incrementally to the output H5, so memory stays roughly O(chunk).

    Parameters
    ----------
    inputs : list[str]
        Input EDM4hep ROOT files, local paths or ``root://`` / ``https://`` URLs.
    output : str | Path
        Destination H5 path; parent directories are created.
    reference : str | Path
        Schema source: a ``.json`` snapshot or an existing Salt-shaped H5.
    max_events : int | None, optional
        Stop after this many events across all inputs.
    chunk_size : int, optional
        Events read per uproot step.
    max_tracks : int, optional
        Track slots per jet.
    max_flows : int, optional
        Flow slots per jet.
    max_electrons : int, optional
        Electron slots per jet.
    max_muons : int, optional
        Muon slots per jet.
    magnetic_field_t : float, optional
        Solenoid field in tesla, used for the q/p conversion.
    truth_cone_radius : float, optional
        Cone radius for the HadronConeExcl truth labelling.
    hadron_pt_cut_gev : float, optional
        Minimum truth-hadron pT considered for labelling.
    lepton_match_mode : str, optional
        ``"constituent"`` or ``"nearest-jet"``.
    lepton_match_dr : float, optional
        Cone used by the ``"nearest-jet"`` mode.
    write_report : bool, optional
        Write the JSON summary report next to the output.
    quiet : bool, optional
        Suppress per-chunk progress output.
    compression : str, optional
        h5py dataset compression filter.
    compression_opts : int, optional
        h5py dataset compression level.

    Raises
    ------
    RuntimeError
        If the inputs cannot be opened or carry no ``events`` tree.
    """
    output = Path(output)
    ref_schema = _load_reference_schema(
        reference,
        tuple(spec.name for spec in FCC_TRAINING_DATASETS if spec.name != "global"),
    )
    accumulator = ReportAccumulator()
    missing_collections: set[str] = set()
    has_electrons = has_muons = True

    chunk_index = 0
    with SaltH5Writer(
        output,
        reference,
        datasets=FCC_TRAINING_DATASETS,
        dtype_overrides={"global": GLOBAL_DTYPE},
        compression=compression,
        compression_opts=compression_opts,
    ) as writer:
        for chunk_index, (chunk_arrays, missing, source_path, entry_start) in enumerate(
            iter_event_chunks(inputs, max_events=max_events, chunk_size=chunk_size), start=1
        ):
            if chunk_index == 1:
                missing_collections = set(missing)
                has_electrons = "Electron#0" not in missing_collections
                has_muons = "Muon#0" not in missing_collections

            (
                global_rows,
                jet_rows,
                track_rows,
                rich_track_rows,
                flow_rows,
                electron_rows,
                muon_rows,
                chunk_diagnostics,
            ) = build_rows_from_chunk(
                chunk_arrays,
                ref_schema=ref_schema,
                has_electrons=has_electrons,
                has_muons=has_muons,
                source_path=source_path,
                entry_start=entry_start,
                max_tracks=max_tracks,
                max_flows=max_flows,
                max_electrons=max_electrons,
                max_muons=max_muons,
                magnetic_field_t=magnetic_field_t,
                truth_cone_radius=truth_cone_radius,
                hadron_pt_cut_gev=hadron_pt_cut_gev,
                lepton_match_mode=lepton_match_mode,
                lepton_match_dr=lepton_match_dr,
            )
            n_jets = len(jet_rows)

            payloads = {
                "global": global_rows,
                "jets": jet_rows,
                "tracks_ghost": track_rows,
                "tracks_ghost_rich_ip": rich_track_rows,
                "flows": flow_rows,
                "electrons": electron_rows,
                "muons": muon_rows,
            }
            for spec in FCC_TRAINING_DATASETS:
                if spec.name in _FILLED:
                    writer.append(spec.name, payloads[spec.name])
                else:
                    writer.extend_with_fill(spec.name, n_jets)

            accumulator.update(
                jet_rows,
                track_rows,
                flow_rows,
                electron_rows,
                muon_rows,
                diagnostics=chunk_diagnostics,
            )

            if not quiet:
                print(f"chunk {chunk_index}: +{n_jets} jets (total {accumulator.n_jets})")

    if chunk_index == 0:
        raise RuntimeError("no events were read from the input(s)")

    if write_report:
        write_conversion_report(
            output,
            accumulator,
            inputs=inputs,
            max_events=max_events,
            missing_collections=missing_collections,
            lepton_match_mode=lepton_match_mode,
        )


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fcc-ftag-convert`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="fcc-ftag-convert",
        description="Convert FCC-ee EDM4hep ROOT files into a Salt-shaped HDF5 file.",
    )
    parser.add_argument(
        "inputs", nargs="+", help="Input EDM4hep ROOT file(s), local path or EOS/root URL"
    )
    parser.add_argument(
        "-o", "--output", default=DEFAULT_OUTPUT, help=f"Output H5 path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Salt schema source: configs/salt_schema.json (default) or a Salt reference H5.",
    )
    parser.add_argument("--max-events", type=int, help="Optional cap on number of events to read")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Number of events read+written per pass (default: {DEFAULT_CHUNK_SIZE}). "
        "Smaller = less memory, larger = less overhead.",
    )
    parser.add_argument(
        "--max-tracks",
        type=int,
        default=75,
        help="Max ghost-track slots per jet (covers ≥99.9%% of FCC R=1.5 jets).",
    )
    parser.add_argument(
        "--max-flows",
        type=int,
        default=100,
        help="Max flow-constituent slots per jet (covers ≥99.8%% of FCC R=1.5 jets).",
    )
    parser.add_argument("--max-electrons", type=int, default=10)
    parser.add_argument("--max-muons", type=int, default=5)
    parser.add_argument("--magnetic-field-t", type=float, default=2.0)
    parser.add_argument(
        "--truth-cone-radius",
        type=float,
        default=1.5,
        help="ΔR cone for heavy-hadron cone matching. Default 1.5 matches the "
        "FCC R=1.5 ee-anti-kT jet size. For ATLAS-style R=0.4 jets use 0.3-0.4.",
    )
    parser.add_argument(
        "--hadron-pt-cut-gev",
        type=float,
        default=5.0,
        help="Minimum hadron pT (GeV) for cone labelling (ATLAS default: 5.0).",
    )
    parser.add_argument(
        "--lepton-match-mode",
        choices=["constituent", "nearest-jet"],
        default="constituent",
        help="How to assign electrons/muons to jets (default: constituent).",
    )
    parser.add_argument(
        "--lepton-match-dr",
        type=float,
        default=0.4,
        help="Maximum deltaR for nearest-jet lepton matching (default: 0.4).",
    )
    parser.add_argument("--no-report", action="store_true", help="Skip writing the summary report.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-chunk progress output.")
    parser.add_argument("--compression", default="gzip")
    parser.add_argument("--compression-opts", type=int, default=7)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel worker subprocesses for file-level sharding (default: 1). "
        "Useful when you pass many input ROOT files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``fcc-ftag-convert``.

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
    if args.workers > 1 and len(args.inputs) > 1:
        # Lazy: only the multi-worker path needs the process pool machinery.
        from fcc_ftag_converter.parallel import parallel_convert  # noqa: PLC0415

        effective_workers = min(args.workers, len(args.inputs))
        parallel_convert(
            args.inputs,
            args.output,
            n_workers=effective_workers,
            reference=args.reference,
            max_events=args.max_events,
            chunk_size=args.chunk_size,
            max_tracks=args.max_tracks,
            max_flows=args.max_flows,
            max_electrons=args.max_electrons,
            max_muons=args.max_muons,
            magnetic_field_t=args.magnetic_field_t,
            truth_cone_radius=args.truth_cone_radius,
            hadron_pt_cut_gev=args.hadron_pt_cut_gev,
            lepton_match_mode=args.lepton_match_mode,
            lepton_match_dr=args.lepton_match_dr,
            compression=args.compression,
            compression_opts=args.compression_opts,
            write_report=not args.no_report,
            quiet=args.quiet,
        )
        print(f"wrote {Path(args.output)}")
        return 0

    if args.workers > 1:
        print(
            f"warning: --workers={args.workers} but only {len(args.inputs)} input(s); "
            "falling back to single-process conversion.",
            file=sys.stderr,
        )

    convert_fcc_edm4hep_to_h5(
        args.inputs,
        args.output,
        reference=args.reference,
        max_events=args.max_events,
        chunk_size=args.chunk_size,
        max_tracks=args.max_tracks,
        max_flows=args.max_flows,
        max_electrons=args.max_electrons,
        max_muons=args.max_muons,
        magnetic_field_t=args.magnetic_field_t,
        truth_cone_radius=args.truth_cone_radius,
        hadron_pt_cut_gev=args.hadron_pt_cut_gev,
        lepton_match_mode=args.lepton_match_mode,
        lepton_match_dr=args.lepton_match_dr,
        write_report=not args.no_report,
        quiet=args.quiet,
        compression=args.compression,
        compression_opts=args.compression_opts,
    )
    print(f"wrote {Path(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
