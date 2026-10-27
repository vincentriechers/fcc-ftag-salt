"""Chunked event streaming from EDM4hep ROOT files via uproot."""

from __future__ import annotations

import socket
import ssl
import warnings
from collections.abc import Iterator
from http.client import HTTPSConnection
from typing import Any

_REQUIRED_COLUMNS = [
    "Jet/Jet.energy",
    "Jet/Jet.mass",
    "Jet/Jet.momentum.x",
    "Jet/Jet.momentum.y",
    "Jet/Jet.momentum.z",
    "Jet/Jet.particles_begin",
    "Jet/Jet.particles_end",
    "Jet#2/Jet#2.index",
    "Jet#2/Jet#2.collectionID",
    "ReconstructedParticles/ReconstructedParticles.momentum.x",
    "ReconstructedParticles/ReconstructedParticles.momentum.y",
    "ReconstructedParticles/ReconstructedParticles.momentum.z",
    "ReconstructedParticles/ReconstructedParticles.energy",
    "ReconstructedParticles/ReconstructedParticles.charge",
    "ReconstructedParticles/ReconstructedParticles.tracks_begin",
    "ReconstructedParticles/ReconstructedParticles.tracks_end",
    "ReconstructedParticles#1/ReconstructedParticles#1.index",
    "ReconstructedParticles#1/ReconstructedParticles#1.collectionID",
    "EFlowTrack/EFlowTrack.trackStates_begin",
    "EFlowTrack/EFlowTrack.trackStates_end",
    "EFlowTrack_1/EFlowTrack_1.D0",
    "EFlowTrack_1/EFlowTrack_1.phi",
    "EFlowTrack_1/EFlowTrack_1.Z0",
    "EFlowTrack_1/EFlowTrack_1.tanLambda",
    "EFlowTrack_1/EFlowTrack_1.omega",
    "EFlowTrack_1/EFlowTrack_1.location",
    "EFlowTrack_1/EFlowTrack_1.covMatrix[21]",
    "MCRecoAssociations#0/MCRecoAssociations#0.index",
    "MCRecoAssociations#0/MCRecoAssociations#0.collectionID",
    "MCRecoAssociations#1/MCRecoAssociations#1.index",
    "MCRecoAssociations#1/MCRecoAssociations#1.collectionID",
    "Particle/Particle.PDG",
    "Particle/Particle.momentum.x",
    "Particle/Particle.momentum.y",
    "Particle/Particle.momentum.z",
    "Particle/Particle.vertex.x",
    "Particle/Particle.vertex.y",
    "Particle/Particle.vertex.z",
    "Particle/Particle.parents_begin",
    "Particle/Particle.parents_end",
    "Particle#0/Particle#0.index",
    "Particle#0/Particle#0.collectionID",
]

_OPTIONAL_COLUMN_GROUPS: dict[str, list[str]] = {
    "Electron#0": [
        "Electron#0/Electron#0.index",
        "Electron#0/Electron#0.collectionID",
    ],
    "Muon#0": [
        "Muon#0/Muon#0.index",
        "Muon#0/Muon#0.collectionID",
    ],
}


def _normalise_uproot_path(input_path: str) -> str:
    if input_path.startswith("root://eospublic.cern.ch//"):
        return "https://eospublic.cern.ch/" + input_path.split("//", 2)[2]
    return input_path


def _patch_uproot_https_verification() -> None:
    """Skip certificate verification for uproot's own HTTPS connections.

    EOS hosts present a CERN Grid CA chain that is not in the standard trust
    store, so the default context rejects them. uproot builds its connections
    without an ``ssl_context`` argument, so the only hook is its
    ``make_connection`` factory — patching that instead of
    ``ssl._create_default_https_context`` keeps verification intact for every
    other https client in the process.
    """
    # Lazy: only https inputs need uproot's http backend, and importing it
    # pulls in the whole source stack at package import time otherwise.
    import uproot.source.http as uproot_http  # noqa: PLC0415  # type: ignore[import-not-found]

    if getattr(uproot_http, "_fcc_unverified_https", False):
        return

    original = uproot_http.make_connection
    # Deliberate, and scoped to uproot alone — see the docstring above.
    context = ssl._create_unverified_context()  # noqa: S323, SLF001

    def make_connection(parsed_url: Any, timeout: float | None):
        if parsed_url.scheme != "https":
            return original(parsed_url, timeout)
        if timeout is None and socket.getdefaulttimeout() is None:
            timeout = 30
        return HTTPSConnection(
            parsed_url.hostname, parsed_url.port, timeout=timeout, context=context
        )

    uproot_http.make_connection = make_connection
    uproot_http._fcc_unverified_https = True  # noqa: SLF001  our own idempotency flag


def _open_options(path: str) -> dict[str, Any]:
    if not str(path).startswith(("http://", "https://")):
        return {}
    from uproot.source.http import (  # noqa: PLC0415  # type: ignore[import-not-found]
        MultithreadedHTTPSource,
    )

    _patch_uproot_https_verification()
    return {"handler": MultithreadedHTTPSource, "timeout": 60}


def _check_optional_groups(
    tree: Any, optional_groups: dict[str, list[str]]
) -> tuple[list[str], set[str]]:
    available: list[str] = []
    missing: set[str] = set()
    for group_name, cols in optional_groups.items():
        try:
            tree.arrays(cols, entry_stop=1, library="ak")
            available.extend(cols)
        except Exception:  # noqa: BLE001  a probe: any failure means "not present"
            missing.add(group_name)
    return available, missing


def iter_event_chunks(
    inputs: list[str],
    *,
    max_events: int | None,
    chunk_size: int,
    required_columns: list[str] | None = None,
    optional_column_groups: dict[str, list[str]] | None = None,
) -> Iterator[tuple[Any, set[str], str, int]]:
    """Yield ``(chunk_array, missing_groups, source_path, entry_start)`` per step.

    ``source_path`` is the input file the chunk came from and ``entry_start`` is
    the 0-based entry index of the chunk's first event within that file. Together
    they uniquely identify every physics event, which the reader hashes into a
    stable global ``eventNumber`` (parallel- and merge-safe).

    ``missing_groups`` is determined once from the first input and is stable
    across all yielded chunks; if any optional EDM4hep collection is missing
    a single warning is emitted (no spam per chunk).

    Parameters
    ----------
    inputs : list[str]
        Input ROOT files, local paths or ``root://`` / ``https://`` URLs.
    max_events : int | None
        Stop after this many events across all inputs; None reads everything.
    chunk_size : int
        Events per yielded chunk.
    required_columns : list[str] | None, optional
        EDM4hep branches that must be present. Defaults to the converter's set.
    optional_column_groups : dict[str, list[str]] | None, optional
        Branch groups that may be absent; missing ones are reported, not fatal.

    Yields
    ------
    tuple[Any, set[str], str, int]
        ``(chunk_array, missing_groups, source_path, entry_start)``.

    Raises
    ------
    RuntimeError
        If uproot is unavailable or an input cannot be opened.
    ValueError
        If ``inputs`` is empty.
    """
    try:
        import uproot  # noqa: PLC0415  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "uproot is required for FCC EDM4hep conversion. "
            "Install it with `pip install uproot` or activate a Key4hep env."
        ) from exc

    if not inputs:
        raise ValueError("no inputs given")

    required = list(required_columns) if required_columns is not None else list(_REQUIRED_COLUMNS)
    optional = (
        dict(optional_column_groups)
        if optional_column_groups is not None
        else dict(_OPTIONAL_COLUMN_GROUPS)
    )

    first = _normalise_uproot_path(inputs[0])
    try:
        first_tree = uproot.open(first, **_open_options(first))["events"]
    except Exception as exc:
        raise RuntimeError(f"failed to open {inputs[0]!r}: {exc}") from exc
    available_optional, missing = _check_optional_groups(first_tree, optional)
    for name in sorted(missing):
        warnings.warn(
            f"Optional EDM4hep collection {name!r} not found in {inputs[0]!r}; "
            "writing schema-compatible placeholders.",
            stacklevel=2,
        )
    all_columns = required + available_optional

    remaining = max_events
    for input_path in inputs:
        if remaining is not None and remaining <= 0:
            break
        path = _normalise_uproot_path(input_path)
        try:
            handle = uproot.open(path, **_open_options(path))
        except Exception as exc:
            raise RuntimeError(f"failed to open {input_path!r}: {exc}") from exc
        tree = handle["events"]
        entry_stop = remaining if remaining is not None else None
        file_entry = 0
        for chunk in tree.iterate(
            all_columns,
            step_size=chunk_size,
            entry_stop=entry_stop,
            library="ak",
        ):
            n = len(chunk)
            emitted = chunk
            if remaining is not None and n > remaining:
                emitted = chunk[:remaining]
                n = remaining
            yield emitted, missing, input_path, file_entry
            file_entry += n
            if remaining is not None:
                remaining -= n
                if remaining <= 0:
                    break
