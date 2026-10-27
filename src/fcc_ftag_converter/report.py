"""Streaming conversion-report accumulator and writer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

_APPROXIMATE_VARIABLES = (
    "tracks_ghost.deta: uses ReconstructedParticle direction, not perigee track eta",
    "lifetime significances: value/uncertainty only; no PV/sign convention",
    "electrons.ftag_ptVarCone30OverPt: charged-particle cone isolation approximation",
    "electrons.ftag_deltaPOverP: from reco four-vector, not track-cluster comparison",
    "electrons.ftag_energyOverP: from reco four-vector",
    "lepton ptfrac/ptrel/dr: relative to FCC jet axis",
    "ptFromTruthDressedWZJet: reco jet pt placeholder",
)

_PLACEHOLDER_VARIABLES = (
    "numberOfPixelHits, numberOfSCTHits: no subdetector hit summary in Delphes",
    "muon quality/spectrometer/energy-loss fields: Muon#0 only gives reco particle refs",
    "electron PID/shower-shape: ParticleIDs uninformative, cluster shapes empty",
    "calo: entirely placeholder",
)


class _Reservoir:
    """Reservoir sampling for approximate streaming quantiles.

    Implements Vitter's Algorithm R in a vectorised form. Median/p95 are
    estimated from a fixed-size random sample (default 100 000) regardless of
    how many events are streamed, so memory is O(k) instead of O(N).
    """

    def __init__(self, k: int = 100_000, seed: int = 0) -> None:
        self.k = int(k)
        self.values: list[float] = []
        self.n_seen = 0
        self.rng = np.random.default_rng(seed)

    def update(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float64).ravel()
        samples = samples[np.isfinite(samples) & (samples != 0)]
        if samples.size == 0:
            return

        if len(self.values) < self.k:
            take = min(samples.size, self.k - len(self.values))
            self.values.extend(samples[:take].tolist())
            self.n_seen += take
            samples = samples[take:]
            if samples.size == 0:
                return

        n_new = samples.size
        seen_at = self.n_seen + np.arange(1, n_new + 1)
        positions = self.rng.integers(0, seen_at)
        keep = positions < self.k
        if keep.any():
            kept_idx = np.where(keep)[0]
            kept_pos = positions[keep]
            for i, p in zip(kept_idx, kept_pos, strict=True):
                self.values[int(p)] = float(samples[int(i)])
        self.n_seen += n_new

    def median(self) -> float | None:
        if not self.values:
            return None
        return float(np.median(self.values))

    def p95(self) -> float | None:
        if not self.values:
            return None
        return float(np.percentile(self.values, 95))

    def stats(self) -> dict[str, float] | None:
        median, p95 = self.median(), self.p95()
        if median is None or p95 is None:
            return None
        return {"median": median, "p95": p95}


class ReportAccumulator:
    """Running statistics + counters accumulated chunk-by-chunk."""

    def __init__(self, reservoir_size: int = 100_000) -> None:
        self.n_jets = 0
        self.valid_counts = {"tracks_ghost": 0, "flows": 0, "electrons": 0, "muons": 0, "calo": 0}
        self.diagnostics = {
            "n_events_total": 0,
            "n_events_empty": 0,
            "n_jets_total": 0,
            "n_jets_no_tracks": 0,
            "n_jets_no_flows": 0,
            "n_jets_bad_kinematics": 0,
        }
        # Jet-level pt is small (one value per jet), keep full reservoir too
        self.jet_pt = _Reservoir(reservoir_size)
        self.reservoirs: dict[str, _Reservoir] = {
            "tracks_ghost.qOverP": _Reservoir(reservoir_size),
            "flows.pt": _Reservoir(reservoir_size),
            "electrons.pt": _Reservoir(reservoir_size),
            "muons.pt": _Reservoir(reservoir_size),
        }

    @staticmethod
    def _valid_mask(rows: np.ndarray) -> np.ndarray:
        if "valid" in (rows.dtype.names or ()):
            return rows["valid"].astype(bool)
        return np.ones(rows.shape, dtype=bool)

    def update(
        self,
        jet_rows: np.ndarray,
        track_rows: np.ndarray,
        flow_rows: np.ndarray,
        electron_rows: np.ndarray,
        muon_rows: np.ndarray,
        *,
        diagnostics: dict[str, int] | None = None,
    ) -> None:
        """Fold one chunk of rows into the running summary.

        Parameters
        ----------
        jet_rows : np.ndarray
            Per-jet rows; their count drives ``n_jets``.
        track_rows : np.ndarray
            Per-jet track rows, or an empty stand-in.
        flow_rows : np.ndarray
            Per-jet flow rows, or an empty stand-in.
        electron_rows : np.ndarray
            Per-jet electron rows, or an empty stand-in.
        muon_rows : np.ndarray
            Per-jet muon rows, or an empty stand-in.
        diagnostics : dict[str, int] | None, optional
            Per-chunk counters to add to the running totals.
        """
        self.n_jets += len(jet_rows)
        if diagnostics is not None:
            for key, value in diagnostics.items():
                self.diagnostics[key] = self.diagnostics.get(key, 0) + int(value)
        if "pt_btagJes" in (jet_rows.dtype.names or ()) and len(jet_rows) > 0:
            self.jet_pt.update(jet_rows["pt_btagJes"])

        for rows, count_key, stats_key, field in (
            (track_rows, "tracks_ghost", "tracks_ghost.qOverP", "qOverP"),
            (flow_rows, "flows", "flows.pt", "pt"),
            (electron_rows, "electrons", "electrons.pt", "pt"),
            (muon_rows, "muons", "muons.pt", "pt"),
        ):
            if rows.ndim < 2:
                continue
            mask = self._valid_mask(rows)
            self.valid_counts[count_key] += int(np.sum(mask))
            if field in (rows.dtype.names or ()):
                self.reservoirs[stats_key].update(rows[field][mask])

    def finalize(self) -> dict[str, Any]:
        """Collapse the accumulated state into the report payload.

        Returns
        -------
        dict[str, Any]
            Jet count, per-stream valid counts, median/p95 stats and diagnostics.
        """
        stats: dict[str, dict[str, float]] = {}
        if (s := self.jet_pt.stats()) is not None:
            stats["jets.pt_btagJes"] = s
        for key, reservoir in self.reservoirs.items():
            if (s := reservoir.stats()) is not None:
                stats[key] = s
        return {
            "n_jets": self.n_jets,
            "valid_counts": dict(self.valid_counts),
            "stats_median_p95": stats,
            "diagnostics": dict(self.diagnostics),
        }


def write_conversion_report(
    output_path: Path,
    accumulator: ReportAccumulator,
    *,
    inputs: list[str],
    max_events: int | None,
    missing_collections: set[str],
    lepton_match_mode: str,
) -> None:
    """Write the JSON conversion report next to ``output_path``.

    Parameters
    ----------
    output_path : Path
        The H5 the report describes; the report gets the same stem.
    accumulator : ReportAccumulator
        Accumulated per-chunk statistics.
    inputs : list[str]
        Input paths that produced the output.
    max_events : int | None
        The ``--max-events`` value in force, for provenance.
    missing_collections : set[str]
        Optional EDM4hep collections that were absent from the inputs.
    lepton_match_mode : str
        The lepton-matching mode used.
    """
    summary = accumulator.finalize()
    data: dict[str, Any] = {
        "inputs": inputs,
        "output": str(output_path),
        "max_events": max_events,
        "lepton_match_mode": lepton_match_mode,
        "units_note": "momentum/energy in MeV; qOverP in MeV^-1",
        "missing_collections": sorted(missing_collections),
        "approximate_variables": list(_APPROXIMATE_VARIABLES),
        "placeholder_variables": list(_PLACEHOLDER_VARIABLES),
        **summary,
    }

    stem = output_path.stem
    json_path = output_path.parent / f"{stem}.summary.json"
    md_path = output_path.parent / f"{stem}.summary.md"

    with open(json_path, "w") as fh:
        json.dump(data, fh, indent=2)

    lines = [
        "# Conversion Summary",
        "",
        f"**Output:** `{output_path}`  ",
        f"**Max events:** {max_events if max_events is not None else 'all'}  ",
        f"**Jets written:** {summary['n_jets']}  ",
        f"**Lepton match mode:** `{lepton_match_mode}`  ",
        "",
        "## Valid object counts",
        "",
        "| Dataset | Valid entries |",
        "|---------|--------------|",
    ]
    for ds, count in data["valid_counts"].items():
        lines.append(f"| {ds} | {count} |")

    diag = summary.get("diagnostics", {})
    if diag:
        lines += [
            "",
            "## Conversion diagnostics",
            "",
            "| Counter | Value |",
            "|---------|------:|",
        ]
        for key in (
            "n_events_total",
            "n_events_empty",
            "n_jets_total",
            "n_jets_no_tracks",
            "n_jets_no_flows",
            "n_jets_bad_kinematics",
        ):
            if key in diag:
                lines.append(f"| `{key}` | {diag[key]} |")

    lines += [
        "",
        "## Variable statistics (reservoir-sampled)",
        "",
        "> Momentum and energy are written in **MeV**; `qOverP` in **MeV⁻¹**.",
        "",
        "| Variable | Median | p95 |",
        "|----------|--------|-----|",
    ]
    for var, s in summary["stats_median_p95"].items():
        lines.append(f"| `{var}` | {s['median']:.4g} | {s['p95']:.4g} |")

    if missing_collections:
        lines += ["", "## Missing collections (placeholders written)", ""]
        for col in sorted(missing_collections):
            lines.append(f"- `{col}`")

    lines += ["", "## Approximate variables", ""]
    for v in _APPROXIMATE_VARIABLES:
        lines.append(f"- {v}")

    lines += ["", "## Placeholder variables (unavailable in FCC Delphes sample)", ""]
    for v in _PLACEHOLDER_VARIABLES:
        lines.append(f"- {v}")

    lines += ["", "## Inputs", ""]
    for inp in inputs:
        lines.append(f"- `{inp}`")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
