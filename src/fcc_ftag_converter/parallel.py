"""Parallel conversion: shard input files across worker subprocesses + merge."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from fcc_ftag_converter.merge import merge_h5_files
from fcc_ftag_converter.report import ReportAccumulator, write_conversion_report


def _split_evenly(items: list[str], n_shards: int) -> list[list[str]]:
    """Round-robin so any single huge file doesn't pin one worker for the whole run.

    Parameters
    ----------
    items : list[str]
        Input paths to distribute.
    n_shards : int
        Requested number of shards.

    Returns
    -------
    list[list[str]]
        Non-empty shards only, so fewer inputs than shards is not an error.
    """
    shards: list[list[str]] = [[] for _ in range(n_shards)]
    for i, item in enumerate(items):
        shards[i % n_shards].append(item)
    return [s for s in shards if s]


def _run_worker(args: list[str]) -> tuple[int, str]:
    """Run one fcc-ftag-convert subprocess.

    Parameters
    ----------
    args : list[str]
        Command-line arguments for ``fcc_ftag_converter.cli``.

    Returns
    -------
    tuple[int, str]
        The subprocess return code and its last stdout line.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "fcc_ftag_converter.cli", *args],
        capture_output=True,
        text=True,
        check=False,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode != 0:
        sys.stderr.write(f"worker failed (rc={proc.returncode}):\n{proc.stderr}\n")
    return proc.returncode, tail


def parallel_convert(
    inputs: list[str],
    output: str | Path,
    *,
    n_workers: int,
    reference: str,
    max_events: int | None = None,
    chunk_size: int = 10_000,
    max_tracks: int = 50,
    max_flows: int = 50,
    max_electrons: int = 10,
    max_muons: int = 5,
    magnetic_field_t: float = 2.0,
    truth_cone_radius: float = 0.4,
    hadron_pt_cut_gev: float = 5.0,
    lepton_match_mode: str = "constituent",
    lepton_match_dr: float = 0.4,
    compression: str = "gzip",
    compression_opts: int = 7,
    write_report: bool = True,
    quiet: bool = False,
) -> None:
    """Shard ``inputs`` over ``n_workers`` subprocesses, then merge.

    Each worker is a fresh ``python -m fcc_ftag_converter.cli`` invocation that
    handles a subset of input files and writes to a temporary H5. Once all
    workers finish, the temp H5s are concatenated via :func:`merge_h5_files`
    and the temp dir is removed.

    File-level sharding only — if you have a single huge ROOT file this falls
    back to a single worker. Use uproot's entry-range slicing if you need
    intra-file parallelism.

    Parameters
    ----------
    inputs : list[str]
        Input EDM4hep ROOT files, local paths or ``root://`` / ``https://`` URLs.
    output : str | Path
        Destination H5 path; parent directories are created.
    n_workers : int
        Number of convert subprocesses to dispatch.
    reference : str
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
    compression : str, optional
        h5py dataset compression filter.
    compression_opts : int, optional
        h5py dataset compression level.
    write_report : bool, optional
        Write the JSON summary report next to the merged output.
    quiet : bool, optional
        Suppress per-worker progress output.

    Raises
    ------
    RuntimeError
        If any worker subprocess exits non-zero.
    """
    output = Path(output)
    shards = _split_evenly(inputs, n_workers)
    if len(shards) <= 1:
        raise RuntimeError(
            "parallel_convert called with only one effective shard; "
            "use convert_fcc_edm4hep_to_h5 directly instead."
        )

    tmpdir = Path(tempfile.mkdtemp(prefix="fcc_ftag_parallel_"))
    worker_outputs: list[Path] = []
    try:
        worker_args: list[list[str]] = []
        for idx, shard in enumerate(shards):
            shard_output = tmpdir / f"worker_{idx:03d}.h5"
            worker_outputs.append(shard_output)
            shard_max_events = None
            if max_events is not None:
                shard_max_events = max_events // len(shards) + (
                    1 if idx < max_events % len(shards) else 0
                )
            args = [
                *shard,
                "--output",
                str(shard_output),
                "--reference",
                reference,
                "--chunk-size",
                str(chunk_size),
                "--max-tracks",
                str(max_tracks),
                "--max-flows",
                str(max_flows),
                "--max-electrons",
                str(max_electrons),
                "--max-muons",
                str(max_muons),
                "--magnetic-field-t",
                str(magnetic_field_t),
                "--truth-cone-radius",
                str(truth_cone_radius),
                "--hadron-pt-cut-gev",
                str(hadron_pt_cut_gev),
                "--lepton-match-mode",
                lepton_match_mode,
                "--lepton-match-dr",
                str(lepton_match_dr),
                "--compression",
                compression,
                "--compression-opts",
                str(compression_opts),
                "--quiet",
            ]
            if shard_max_events is not None:
                args += ["--max-events", str(shard_max_events)]
            worker_args.append(args)

        if not quiet:
            n_files = sum(len(s) for s in shards)
            print(f"dispatching {len(worker_args)} workers, {n_files} input files total")

        with ProcessPoolExecutor(max_workers=len(worker_args)) as pool:
            futures = {pool.submit(_run_worker, args): i for i, args in enumerate(worker_args)}
            failed = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                rc, tail = fut.result()
                if rc != 0:
                    failed += 1
                if not quiet:
                    print(f"worker {idx:03d}: rc={rc} {tail}")
            if failed:
                raise RuntimeError(f"{failed}/{len(worker_args)} workers failed")

        accumulator = ReportAccumulator() if write_report else None
        merge_h5_files(
            worker_outputs,
            output,
            accumulator=accumulator,
            verbose=not quiet,
        )

        worker_missing: set[str] = set()
        if accumulator is not None:
            for worker_h5 in worker_outputs:
                json_path = worker_h5.parent / f"{worker_h5.stem}.summary.json"
                if not json_path.exists():
                    continue
                with open(json_path) as fh:
                    data = json.load(fh)
                for key, value in data.get("diagnostics", {}).items():
                    accumulator.diagnostics[key] = accumulator.diagnostics.get(key, 0) + int(value)
                worker_missing.update(data.get("missing_collections", []))
            write_conversion_report(
                output,
                accumulator,
                inputs=inputs,
                max_events=max_events,
                missing_collections=worker_missing,
                lepton_match_mode=f"{lepton_match_mode} (parallel, {len(worker_args)} workers)",
            )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
