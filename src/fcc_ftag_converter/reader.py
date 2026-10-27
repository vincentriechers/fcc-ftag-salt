"""Per-chunk row builder: takes an awkward Array of events, returns Salt rows."""

from __future__ import annotations

import hashlib
from operator import itemgetter
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import awkward as ak
import numpy as np
import numpy.typing as npt

from fcc_ftag_converter.physics import (
    _charged_pt_cone_over_pt,
    _ensure_finite_training_defaults,
    _eta_from_components,
    _gev_to_mev,
    _lepton_track_info,
    _nearest_jet_lepton_match,
    _ptrel,
    _q_over_p_from_omega,
    _select_track_state,
    _signed_lepton_id,
    _track_uncertainties,
    _wrap_delta_phi,
)
from fcc_ftag_converter.schema import (
    ELECTRON_TARGET_FIELDS,
    FLOW_TARGET_FIELDS,
    TRACK_TARGET_FIELDS,
    _global_rows_from_jets,
    empty_rows_from_reference,
)
from fcc_ftag_converter.truth import (
    _assign_track_vertex_indices,
    _build_children_map,
    _child_heavy_flavour_pdg,
    _cone_exclusive_truth_labels,
    _label_from_ancestors,
    _lepton_decay_label_for_hadron,
    _lepton_id_from_mc,
    _origin_label_from_ancestors,
    _track_type_label_from_mc,
)


def _stable_event_number(source_path: str, global_entry: int) -> int:
    """Deterministic 63-bit eventNumber from (source file, entry index).

    Parallel- and merge-safe: every physics event hashes to the same value no
    matter which worker or output part processes it, all jets in one event share
    it (so train/val/test splits on ``eventNumber % N`` never leak across the
    split), and the digest is uniform mod N. Uses the file basename so the id is
    stable across staging/scratch path moves.

    Parameters
    ----------
    source_path : str
        Path of the input ROOT file; only its basename is hashed.
    global_entry : int
        Entry index within that file.

    Returns
    -------
    int
        A 63-bit event number.
    """
    key = f"{Path(source_path).name}:{int(global_entry)}".encode()
    digest = hashlib.blake2b(key, digest_size=8).digest()
    return int.from_bytes(digest, "big") & 0x7FFF_FFFF_FFFF_FFFF


def _jet_fold_hash(event_number: int, local_jet_index: int) -> int:
    """Per-jet uint32 fold hash, stable for a given (event, jet) pair.

    Parameters
    ----------
    event_number : int
        Event number from :func:`_stable_event_number`.
    local_jet_index : int
        Index of the jet within its event.

    Returns
    -------
    int
        A uint32-range fold hash.
    """
    key = f"{event_number}:{local_jet_index}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=4).digest(), "big")


def _fields(rows: np.ndarray) -> tuple[str, ...]:
    """Field names of a structured array, empty tuple for an unstructured one.

    Parameters
    ----------
    rows : np.ndarray
        The array to inspect.

    Returns
    -------
    tuple[str, ...]
        ``rows.dtype.names`` with the ``None`` case flattened away, so callers
        can test membership without narrowing first.
    """
    return rows.dtype.names or ()


def _ak_np(arrays: Any, key: str, event_index: int, dtype: npt.DTypeLike) -> np.ndarray:
    return np.asarray(ak.to_numpy(arrays[key][event_index]), dtype=dtype)


def _extract_event_arrays(
    arrays: Any, event_index: int, has_electrons: bool, has_muons: bool
) -> SimpleNamespace:
    """Pull per-event awkward fields into a SimpleNamespace of numpy arrays.

    Parameters
    ----------
    arrays : Any
        Awkward Array of the chunk, keyed by EDM4hep branch name.
    event_index : int
        Index of the event within the chunk.
    has_electrons : bool
        Whether the input carries an Electron collection.
    has_muons : bool
        Whether the input carries a Muon collection.

    Returns
    -------
    SimpleNamespace
        One attribute per EDM4hep field used by the row builder.
    """
    ev = SimpleNamespace()

    ev.jet_px = _ak_np(arrays, "Jet/Jet.momentum.x", event_index, np.float32)
    ev.jet_py = _ak_np(arrays, "Jet/Jet.momentum.y", event_index, np.float32)
    ev.jet_pz = _ak_np(arrays, "Jet/Jet.momentum.z", event_index, np.float32)
    ev.jet_energy = _ak_np(arrays, "Jet/Jet.energy", event_index, np.float32)
    ev.jet_mass = _ak_np(arrays, "Jet/Jet.mass", event_index, np.float32)
    ev.jet_pt = np.hypot(ev.jet_px, ev.jet_py).astype(np.float32)
    ev.jet_eta = _eta_from_components(ev.jet_px, ev.jet_py, ev.jet_pz)
    ev.jet_phi = np.arctan2(ev.jet_py, ev.jet_px).astype(np.float32)
    ev.jet_particle_begin = _ak_np(arrays, "Jet/Jet.particles_begin", event_index, np.int64)
    ev.jet_particle_end = _ak_np(arrays, "Jet/Jet.particles_end", event_index, np.int64)
    ev.jet_reco_index = _ak_np(arrays, "Jet#2/Jet#2.index", event_index, np.int64)
    ev.jet_reco_collection = _ak_np(arrays, "Jet#2/Jet#2.collectionID", event_index, np.int64)

    ev.reco_px = _ak_np(
        arrays, "ReconstructedParticles/ReconstructedParticles.momentum.x", event_index, np.float32
    )
    ev.reco_py = _ak_np(
        arrays, "ReconstructedParticles/ReconstructedParticles.momentum.y", event_index, np.float32
    )
    ev.reco_pz = _ak_np(
        arrays, "ReconstructedParticles/ReconstructedParticles.momentum.z", event_index, np.float32
    )
    ev.reco_energy = _ak_np(
        arrays, "ReconstructedParticles/ReconstructedParticles.energy", event_index, np.float32
    )
    ev.reco_charge = _ak_np(
        arrays, "ReconstructedParticles/ReconstructedParticles.charge", event_index, np.float32
    )
    ev.reco_pt_all = np.hypot(ev.reco_px, ev.reco_py).astype(np.float32)
    ev.reco_eta_all = np.zeros_like(ev.reco_pt_all)
    np.divide(ev.reco_pz, ev.reco_pt_all, out=ev.reco_eta_all, where=ev.reco_pt_all > 0)
    ev.reco_eta_all = np.arcsinh(ev.reco_eta_all).astype(np.float32)
    ev.reco_phi_all = np.arctan2(ev.reco_py, ev.reco_px).astype(np.float32)
    ev.reco_tracks_begin = _ak_np(
        arrays, "ReconstructedParticles/ReconstructedParticles.tracks_begin", event_index, np.int64
    )
    ev.reco_tracks_end = _ak_np(
        arrays, "ReconstructedParticles/ReconstructedParticles.tracks_end", event_index, np.int64
    )
    ev.reco_track_index = _ak_np(
        arrays, "ReconstructedParticles#1/ReconstructedParticles#1.index", event_index, np.int64
    )
    ev.reco_track_collection = _ak_np(
        arrays,
        "ReconstructedParticles#1/ReconstructedParticles#1.collectionID",
        event_index,
        np.int64,
    )

    if has_electrons:
        e_idx = _ak_np(arrays, "Electron#0/Electron#0.index", event_index, np.int64)
        e_col = _ak_np(arrays, "Electron#0/Electron#0.collectionID", event_index, np.int64)
        ev.electron_reco_indices = {
            int(i) for i, c in zip(e_idx, e_col, strict=False) if int(c) == 15
        }
    else:
        ev.electron_reco_indices = set()

    if has_muons:
        m_idx = _ak_np(arrays, "Muon#0/Muon#0.index", event_index, np.int64)
        m_col = _ak_np(arrays, "Muon#0/Muon#0.collectionID", event_index, np.int64)
        ev.muon_reco_indices = {int(i) for i, c in zip(m_idx, m_col, strict=False) if int(c) == 15}
    else:
        ev.muon_reco_indices = set()

    ev.track_states_begin = _ak_np(
        arrays, "EFlowTrack/EFlowTrack.trackStates_begin", event_index, np.int64
    )
    ev.track_states_end = _ak_np(
        arrays, "EFlowTrack/EFlowTrack.trackStates_end", event_index, np.int64
    )
    ev.state_d0 = _ak_np(arrays, "EFlowTrack_1/EFlowTrack_1.D0", event_index, np.float32)
    ev.state_phi = _ak_np(arrays, "EFlowTrack_1/EFlowTrack_1.phi", event_index, np.float32)
    ev.state_z0 = _ak_np(arrays, "EFlowTrack_1/EFlowTrack_1.Z0", event_index, np.float32)
    ev.state_tan_lambda = _ak_np(
        arrays, "EFlowTrack_1/EFlowTrack_1.tanLambda", event_index, np.float32
    )
    ev.state_omega = _ak_np(arrays, "EFlowTrack_1/EFlowTrack_1.omega", event_index, np.float32)
    ev.state_location = _ak_np(arrays, "EFlowTrack_1/EFlowTrack_1.location", event_index, np.int32)
    ev.state_covariance = _ak_np(
        arrays, "EFlowTrack_1/EFlowTrack_1.covMatrix[21]", event_index, np.float32
    )

    a0_idx = _ak_np(
        arrays, "MCRecoAssociations#0/MCRecoAssociations#0.index", event_index, np.int64
    )
    a0_col = _ak_np(
        arrays, "MCRecoAssociations#0/MCRecoAssociations#0.collectionID", event_index, np.int64
    )
    a1_idx = _ak_np(
        arrays, "MCRecoAssociations#1/MCRecoAssociations#1.index", event_index, np.int64
    )
    a1_col = _ak_np(
        arrays, "MCRecoAssociations#1/MCRecoAssociations#1.collectionID", event_index, np.int64
    )
    reco_to_mc: dict[int, list[int]] = {}
    for a0, c0, a1, c1 in zip(a0_idx, a0_col, a1_idx, a1_col, strict=False):
        if c0 == 15 and c1 == 8:
            reco_to_mc.setdefault(int(a0), []).append(int(a1))
        elif c0 == 8 and c1 == 15:
            reco_to_mc.setdefault(int(a1), []).append(int(a0))
    ev.reco_to_mc = reco_to_mc

    ev.pdg_ids = _ak_np(arrays, "Particle/Particle.PDG", event_index, np.int64)
    ev.mc_px = _ak_np(arrays, "Particle/Particle.momentum.x", event_index, np.float32)
    ev.mc_py = _ak_np(arrays, "Particle/Particle.momentum.y", event_index, np.float32)
    ev.mc_pz = _ak_np(arrays, "Particle/Particle.momentum.z", event_index, np.float32)
    ev.mc_vx = _ak_np(arrays, "Particle/Particle.vertex.x", event_index, np.float32)
    ev.mc_vy = _ak_np(arrays, "Particle/Particle.vertex.y", event_index, np.float32)
    ev.mc_vz = _ak_np(arrays, "Particle/Particle.vertex.z", event_index, np.float32)
    ev.parent_begin = _ak_np(arrays, "Particle/Particle.parents_begin", event_index, np.int64)
    ev.parent_end = _ak_np(arrays, "Particle/Particle.parents_end", event_index, np.int64)
    ev.parent_index = _ak_np(arrays, "Particle#0/Particle#0.index", event_index, np.int64)

    return ev


def _track_kwargs(ev: SimpleNamespace) -> dict[str, Any]:
    return {
        "reco_tracks_begin": ev.reco_tracks_begin,
        "reco_tracks_end": ev.reco_tracks_end,
        "reco_track_index": ev.reco_track_index,
        "reco_track_collection": ev.reco_track_collection,
        "track_states_begin": ev.track_states_begin,
        "track_states_end": ev.track_states_end,
        "state_d0": ev.state_d0,
        "state_z0": ev.state_z0,
        "state_tan_lambda": ev.state_tan_lambda,
        "state_omega": ev.state_omega,
        "state_location": ev.state_location,
        "state_covariance": ev.state_covariance,
    }


def _build_lepton_dict(
    reco_index: int,
    target_j: int,
    ev: SimpleNamespace,
    *,
    magnetic_field_t: float,
    is_electron: bool,
) -> dict:
    reco_pt = float(ev.reco_pt_all[reco_index])
    reco_eta = float(ev.reco_eta_all[reco_index])
    reco_phi = float(ev.reco_phi_all[reco_index])
    jet_pt_j = float(ev.jet_pt[target_j])
    deta = reco_eta - float(ev.jet_eta[target_j])
    dphi = float(_wrap_delta_phi(reco_phi - float(ev.jet_phi[target_j])))
    dr = float(np.hypot(deta, dphi))
    ptfrac = reco_pt / jet_pt_j if jet_pt_j > 0 else 0.0
    ptrel = float(
        _gev_to_mev(
            _ptrel(
                float(ev.reco_px[reco_index]),
                float(ev.reco_py[reco_index]),
                float(ev.reco_pz[reco_index]),
                float(ev.jet_px[target_j]),
                float(ev.jet_py[target_j]),
                float(ev.jet_pz[target_j]),
            )
        )
    )
    tinfo = _lepton_track_info(reco_index, magnetic_field_t=magnetic_field_t, **_track_kwargs(ev))

    if not is_electron:
        return {
            "reco_index": reco_index,
            "pt": float(_gev_to_mev(reco_pt)),
            "eta": reco_eta,
            "phi": reco_phi,
            "ptfrac": ptfrac,
            "ptrel": ptrel,
            "dr": dr,
            "qOverP": tinfo["qOverP"],
        }

    reco_e = float(ev.reco_energy[reco_index])
    return {
        "reco_index": reco_index,
        "pt": float(_gev_to_mev(reco_pt)),
        "eta": reco_eta,
        "phi": reco_phi,
        "abs_eta": abs(reco_eta),
        "ftag_et": float(_gev_to_mev(reco_e)),
        "ptfrac": ptfrac,
        "ptrel": ptrel,
        "dr": dr,
        "qOverP": tinfo["qOverP"],
        "d0RelativeToBeamspot": tinfo["d0"],
        "d0RelativeToBeamspotSignificance": tinfo["d0_significance"],
        "ftag_ptVarCone30OverPt": _charged_pt_cone_over_pt(
            reco_index, ev.reco_pt_all, ev.reco_eta_all, ev.reco_phi_all, ev.reco_charge
        ),
        # In Delphes/EDM4hep ReconstructedParticles, energy is reconstructed as
        # |p| for massless tracks, so E/p ≡ 1 and (p-E)/p ≡ 0. These are not
        # physical at FCC reco level — leave as NaN so the model can mask them.
        "ftag_deltaPOverP": float("nan"),
        "ftag_energyOverP": float("nan"),
    }


def _build_nearest_jet_lepton_lists(
    ev: SimpleNamespace, n_jets: int, lepton_match_dr: float, magnetic_field_t: float
) -> tuple[dict[int, list[dict]], dict[int, list[dict]]]:
    electrons_by_jet: dict[int, list[dict]] = {j: [] for j in range(n_jets)}
    muons_by_jet: dict[int, list[dict]] = {j: [] for j in range(n_jets)}
    for reco_index, target_j in _nearest_jet_lepton_match(
        ev.electron_reco_indices,
        ev.reco_eta_all,
        ev.reco_phi_all,
        ev.jet_eta,
        ev.jet_phi,
        lepton_match_dr,
    ).items():
        electrons_by_jet[target_j].append(
            _build_lepton_dict(
                reco_index, target_j, ev, magnetic_field_t=magnetic_field_t, is_electron=True
            )
        )
    for reco_index, target_j in _nearest_jet_lepton_match(
        ev.muon_reco_indices,
        ev.reco_eta_all,
        ev.reco_phi_all,
        ev.jet_eta,
        ev.jet_phi,
        lepton_match_dr,
    ).items():
        muons_by_jet[target_j].append(
            _build_lepton_dict(
                reco_index, target_j, ev, magnetic_field_t=magnetic_field_t, is_electron=False
            )
        )
    return electrons_by_jet, muons_by_jet


def _fill_track_slot(
    track_rows: np.ndarray,
    rich_track_rows: np.ndarray,
    global_jet_index: int,
    slot: int,
    track: dict,
) -> None:
    for field in (
        "pt",
        "deta",
        "dphi",
        "d0",
        "z0SinTheta",
        "qOverP",
        "qOverPUncertainty",
        "d0Uncertainty",
        "z0SinThetaUncertainty",
        "phiUncertainty",
        "thetaUncertainty",
        "ftagTruthTypeLabel",
        "ftagTruthOriginLabel",
        "leptonID",
    ):
        if field in _fields(track_rows):
            track_rows[field][global_jet_index, slot] = track[field]
        if field in _fields(rich_track_rows):
            rich_track_rows[field][global_jet_index, slot] = track[field]
    # Significance with uncertainty floor (1 μm) + clip (±1000) to suppress
    # blow-ups from degenerate Delphes covariance matrices.
    min_unc_mm = 1e-3
    max_sig = 1000.0
    if "lifetimeSignedD0Significance" in _fields(track_rows):
        d0_unc = max(float(track.get("d0Uncertainty", 0.0)), min_unc_mm)
        sig = float(np.clip(track["d0"] / d0_unc, -max_sig, max_sig))
        track_rows["lifetimeSignedD0Significance"][global_jet_index, slot] = sig
        rich_track_rows["lifetimeSignedD0Significance"][global_jet_index, slot] = sig
    if "lifetimeSignedZ0SinThetaSignificance" in _fields(track_rows):
        z0_unc = max(float(track.get("z0SinThetaUncertainty", 0.0)), min_unc_mm)
        sig = float(np.clip(track["z0SinTheta"] / z0_unc, -max_sig, max_sig))
        track_rows["lifetimeSignedZ0SinThetaSignificance"][global_jet_index, slot] = sig
        rich_track_rows["lifetimeSignedZ0SinThetaSignificance"][global_jet_index, slot] = sig
    if "valid" in _fields(track_rows):
        _ensure_finite_training_defaults(track_rows, (global_jet_index, slot), TRACK_TARGET_FIELDS)
        track_rows["valid"][global_jet_index, slot] = True
    if "valid" in _fields(rich_track_rows):
        _ensure_finite_training_defaults(
            rich_track_rows, (global_jet_index, slot), TRACK_TARGET_FIELDS
        )
        rich_track_rows["valid"][global_jet_index, slot] = True


def build_rows_from_chunk(
    chunk_arrays: Any,
    *,
    ref_schema: dict,
    has_electrons: bool,
    has_muons: bool,
    source_path: str = "",
    entry_start: int = 0,
    max_tracks: int = 50,
    max_flows: int = 50,
    max_electrons: int = 10,
    max_muons: int = 5,
    magnetic_field_t: float = 2.0,
    truth_cone_radius: float = 0.4,
    hadron_pt_cut_gev: float = 5.0,
    lepton_match_mode: str = "constituent",
    lepton_match_dr: float = 0.4,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, int],
]:
    """Build Salt-shaped row buffers for one chunk of EDM4hep events.

    Parameters
    ----------
    chunk_arrays : Any
        Awkward Array of the chunk, keyed by EDM4hep branch name.
    ref_schema : dict
        Pre-loaded schema from :func:`_load_reference_schema`.
    has_electrons : bool
        Whether the input carries an Electron collection.
    has_muons : bool
        Whether the input carries a Muon collection.
    source_path : str, optional
        Input file path, hashed into the event numbers.
    entry_start : int, optional
        Index of the chunk's first event within the file.
    max_tracks : int, optional
        Track slots per jet.
    max_flows : int, optional
        Flow slots per jet.
    max_electrons : int, optional
        Electron slots per jet.
    max_muons : int, optional
        Muon slots per jet.
    magnetic_field_t : float, optional
        Solenoid field in tesla.
    truth_cone_radius : float, optional
        Cone radius for the HadronConeExcl truth labelling.
    hadron_pt_cut_gev : float, optional
        Minimum truth-hadron pT considered for labelling.
    lepton_match_mode : str, optional
        ``"constituent"`` or ``"nearest-jet"``.
    lepton_match_dr : float, optional
        Cone used by the ``"nearest-jet"`` mode.

    Returns
    -------
    tuple
        ``(global, jets, tracks, rich_tracks, flows, electrons, muons,
        diagnostics)`` where ``diagnostics`` carries per-chunk counters for
        empty events, jets without tracks/flows, and jets with non-finite
        kinematics.
    """
    jet_counts = np.asarray(ak.num(chunk_arrays["Jet/Jet.energy"], axis=1), dtype=np.int64)
    n_total = int(jet_counts.sum())
    jet_rows = empty_rows_from_reference(ref_schema, "jets", n_total)
    track_rows = empty_rows_from_reference(ref_schema, "tracks_ghost", n_total)
    rich_track_rows = empty_rows_from_reference(ref_schema, "tracks_ghost_rich_ip", n_total)
    flow_rows = empty_rows_from_reference(ref_schema, "flows", n_total)
    electron_rows = empty_rows_from_reference(ref_schema, "electrons", n_total)
    muon_rows = empty_rows_from_reference(ref_schema, "muons", n_total)
    track_slots = min(max_tracks, track_rows.shape[1] if track_rows.ndim == 2 else 0)
    flow_slots = min(max_flows, flow_rows.shape[1] if flow_rows.ndim == 2 else 0)
    electron_slots = min(max_electrons, electron_rows.shape[1] if electron_rows.ndim == 2 else 0)
    muon_slots = min(max_muons, muon_rows.shape[1] if muon_rows.ndim == 2 else 0)

    diagnostics = {
        "n_events_total": len(jet_counts),
        "n_events_empty": int(np.sum(jet_counts == 0)),
        "n_jets_total": n_total,
        "n_jets_no_tracks": 0,
        "n_jets_no_flows": 0,
        "n_jets_bad_kinematics": 0,
    }

    offset = 0
    for event_index, count in enumerate(jet_counts):
        if count == 0:
            continue
        stop = offset + int(count)
        ev = _extract_event_arrays(chunk_arrays, event_index, has_electrons, has_muons)
        jet_pt_mev = _gev_to_mev(ev.jet_pt).astype(np.float32)

        for field, value in (
            ("pt", jet_pt_mev),
            ("pt_btagJes", jet_pt_mev),
            ("energy", _gev_to_mev(ev.jet_energy).astype(np.float32)),
            ("mass", _gev_to_mev(ev.jet_mass).astype(np.float32)),
            ("eta", ev.jet_eta),
            ("eta_btagJes", ev.jet_eta),
            ("absEta_btagJes", np.abs(ev.jet_eta)),
            ("phi", ev.jet_phi),
        ):
            if field in _fields(jet_rows):
                jet_rows[field][offset:stop] = value

        event_number = _stable_event_number(source_path, entry_start + event_index)
        if "eventNumber" in _fields(jet_rows):
            jet_rows["eventNumber"][offset:stop] = event_number

        for field in ("n_tracks_ghost", "n_tracks_ghost_rich_ip", "n_flow_calo_cal_topo_clusters"):
            if field in _fields(jet_rows):
                jet_rows[field][offset:stop] = 0
        for field in (
            "HadronConeExclTruthLabelID",
            "HadronGhostTruthLabelID",
            "PartonTruthLabelID",
            "HadronConeExclExtendedTruthLabelID",
            "HadronGhostExtendedTruthLabelID",
        ):
            if field in _fields(jet_rows):
                jet_rows[field][offset:stop] = -1

        label_cache: dict[int, int] = {}
        origin_cache: dict[int, int] = {}
        children_map = _build_children_map(
            ev.parent_begin,
            ev.parent_end,
            ev.parent_index,
            len(ev.pdg_ids),
        )
        cone_labels, cone_ext_labels, cone_label_pdg_ids, cone_match_mc = (
            _cone_exclusive_truth_labels(
                ev.jet_eta,
                ev.jet_phi,
                pdg_ids=ev.pdg_ids,
                mc_px=ev.mc_px,
                mc_py=ev.mc_py,
                mc_pz=ev.mc_pz,
                parent_begin=ev.parent_begin,
                parent_end=ev.parent_end,
                parent_index=ev.parent_index,
                children_map=children_map,
                radius=truth_cone_radius,
                hadron_pt_min_gev=hadron_pt_cut_gev,
            )
        )
        event_track_mcs: list[tuple[tuple[int, int], int]] = []

        if lepton_match_mode == "nearest-jet":
            nj_electrons_by_jet, nj_muons_by_jet = _build_nearest_jet_lepton_lists(
                ev,
                int(count),
                lepton_match_dr,
                magnetic_field_t,
            )

        for local_jet_index in range(int(count)):
            global_jet_index = offset + local_jet_index
            fold_hash = _jet_fold_hash(event_number, local_jet_index)
            for fold_field in ("jetFoldHash", "jetFoldHash_noHits"):
                if fold_field in _fields(jet_rows):
                    jet_rows[fold_field][global_jet_index] = fold_hash
            constituents: list[dict] = []
            flows: list[dict] = []
            electrons: list[dict] = []
            muons: list[dict] = []

            for relation_index in range(
                int(ev.jet_particle_begin[local_jet_index]),
                int(ev.jet_particle_end[local_jet_index]),
            ):
                if relation_index < 0 or relation_index >= len(ev.jet_reco_index):
                    continue
                if ev.jet_reco_collection[relation_index] != 15:
                    continue
                reco_index = int(ev.jet_reco_index[relation_index])
                if reco_index < 0 or reco_index >= len(ev.reco_tracks_begin):
                    continue

                reco_mc_ids = ev.reco_to_mc.get(reco_index, [])
                reco_label = 0
                for mc_index in reco_mc_ids:
                    reco_label = max(
                        reco_label,
                        _label_from_ancestors(
                            mc_index,
                            ev.pdg_ids,
                            ev.parent_begin,
                            ev.parent_end,
                            ev.parent_index,
                            label_cache,
                        ),
                    )
                # ATLAS ftagTruthOriginLabel: take the first valid MC association.
                if reco_mc_ids:
                    origin_label = _origin_label_from_ancestors(
                        reco_mc_ids[0],
                        ev.pdg_ids,
                        ev.parent_begin,
                        ev.parent_end,
                        ev.parent_index,
                        origin_cache,
                    )
                else:
                    origin_label = 7  # OtherSecondary (no MC link)

                reco_pt = float(ev.reco_pt_all[reco_index])
                if reco_pt <= 0:
                    continue
                reco_eta = float(ev.reco_eta_all[reco_index])
                reco_phi = float(ev.reco_phi_all[reco_index])
                reco_deta = reco_eta - float(ev.jet_eta[local_jet_index])
                reco_dphi = float(_wrap_delta_phi(reco_phi - ev.jet_phi[local_jet_index]))
                flows.append({
                    "reco_index": reco_index,
                    "pt": float(_gev_to_mev(reco_pt)),
                    "energy": float(_gev_to_mev(float(ev.reco_energy[reco_index]))),
                    "deta": reco_deta,
                    "dphi": reco_dphi,
                    "isCharged": int(abs(float(ev.reco_charge[reco_index])) > 0.0),
                })

                is_electron = reco_index in ev.electron_reco_indices
                is_muon = reco_index in ev.muon_reco_indices

                electron = None
                muon = None
                if lepton_match_mode == "constituent":
                    if is_electron:
                        electron = _build_lepton_dict(
                            reco_index,
                            local_jet_index,
                            ev,
                            magnetic_field_t=magnetic_field_t,
                            is_electron=True,
                        )
                    if is_muon:
                        muon = _build_lepton_dict(
                            reco_index,
                            local_jet_index,
                            ev,
                            magnetic_field_t=magnetic_field_t,
                            is_electron=False,
                        )

                track_type_label = _track_type_label_from_mc(
                    reco_mc_ids,
                    ev.pdg_ids,
                    ev.parent_begin,
                    ev.parent_end,
                    ev.parent_index,
                )
                if is_electron:
                    lepton_id = _signed_lepton_id(11, float(ev.reco_charge[reco_index]))
                elif is_muon:
                    lepton_id = _signed_lepton_id(13, float(ev.reco_charge[reco_index]))
                else:
                    lepton_id = _lepton_id_from_mc(reco_mc_ids, ev.pdg_ids)

                for track_relation in range(
                    int(ev.reco_tracks_begin[reco_index]), int(ev.reco_tracks_end[reco_index])
                ):
                    if track_relation < 0 or track_relation >= len(ev.reco_track_index):
                        continue
                    if ev.reco_track_collection[track_relation] != 6:
                        continue
                    track_index = int(ev.reco_track_index[track_relation])
                    if track_index < 0 or track_index >= len(ev.track_states_begin):
                        continue
                    state_index = _select_track_state(
                        int(ev.track_states_begin[track_index]),
                        int(ev.track_states_end[track_index]),
                        ev.state_location,
                    )
                    if state_index < 0 or state_index >= len(ev.state_d0):
                        continue
                    tan_lambda = float(ev.state_tan_lambda[state_index])
                    sin_theta = 1.0 / np.sqrt(1.0 + tan_lambda * tan_lambda)
                    z0_sin_theta = float(ev.state_z0[state_index] * sin_theta)
                    omega = float(ev.state_omega[state_index])
                    q_over_p = _q_over_p_from_omega(omega, tan_lambda, magnetic_field_t)
                    covariance = (
                        ev.state_covariance[state_index]
                        if state_index < len(ev.state_covariance)
                        else np.zeros(21, dtype=np.float32)
                    )
                    uncertainties = _track_uncertainties(
                        np.asarray(covariance, dtype=np.float32),
                        z0=float(ev.state_z0[state_index]),
                        omega=omega,
                        tan_lambda=tan_lambda,
                        magnetic_field_t=magnetic_field_t,
                    )
                    constituents.append({
                        "track_index": track_index,
                        "pt": float(_gev_to_mev(reco_pt)),
                        "deta": reco_eta - float(ev.jet_eta[local_jet_index]),
                        "dphi": float(
                            _wrap_delta_phi(ev.state_phi[state_index] - ev.jet_phi[local_jet_index])
                        ),
                        "d0": float(ev.state_d0[state_index]),
                        "z0SinTheta": z0_sin_theta,
                        "qOverP": q_over_p,
                        "truth_label": reco_label,
                        "ftagTruthTypeLabel": track_type_label,
                        "ftagTruthOriginLabel": origin_label,
                        "leptonID": lepton_id,
                        "_mc_index": reco_mc_ids[0] if reco_mc_ids else -1,
                        **uncertainties,
                    })
                if electron is not None:
                    electrons.append(electron)
                if muon is not None:
                    muons.append(muon)

            if lepton_match_mode == "nearest-jet":
                electrons = list(nj_electrons_by_jet.get(local_jet_index, []))
                muons = list(nj_muons_by_jet.get(local_jet_index, []))

            unique_constituents = {t["track_index"]: t for t in constituents}
            tracks = sorted(unique_constituents.values(), key=itemgetter("pt"), reverse=True)[
                :track_slots
            ]
            unique_flows = {f["reco_index"]: f for f in flows}
            flows = sorted(unique_flows.values(), key=itemgetter("pt"), reverse=True)[:flow_slots]
            unique_electrons = {e["reco_index"]: e for e in electrons}
            electrons = sorted(unique_electrons.values(), key=itemgetter("pt"), reverse=True)[
                :electron_slots
            ]
            unique_muons = {m["reco_index"]: m for m in muons}
            muons = sorted(unique_muons.values(), key=itemgetter("pt"), reverse=True)[:muon_slots]

            if "n_tracks_ghost" in _fields(jet_rows):
                jet_rows["n_tracks_ghost"][global_jet_index] = len(tracks)
            if "n_tracks_ghost_rich_ip" in _fields(jet_rows):
                jet_rows["n_tracks_ghost_rich_ip"][global_jet_index] = len(tracks)
            if "n_flow_calo_cal_topo_clusters" in _fields(jet_rows):
                jet_rows["n_flow_calo_cal_topo_clusters"][global_jet_index] = len(flows)
            if len(tracks) == 0:
                diagnostics["n_jets_no_tracks"] += 1
            if len(flows) == 0:
                diagnostics["n_jets_no_flows"] += 1
            if not (
                np.isfinite(jet_pt_mev[local_jet_index])
                and np.isfinite(ev.jet_eta[local_jet_index])
                and np.isfinite(ev.jet_phi[local_jet_index])
            ):
                diagnostics["n_jets_bad_kinematics"] += 1

            # ATLAS spec: n_b/n_c/n_bc/n_hf use ftagTruthOriginLabel
            #   3 = FromB, 4 = FromBC (c-from-b), 5 = FromC (direct c, no b)
            #   n_hf = n_b + n_bc + n_c (all heavy-flavour origin tracks).
            track_label_counts = {
                "n_b_tracks_ghost": sum(1 for t in tracks if t.get("ftagTruthOriginLabel") == 3),
                "n_c_tracks_ghost": sum(1 for t in tracks if t.get("ftagTruthOriginLabel") == 5),
                "n_bc_tracks_ghost": sum(1 for t in tracks if t.get("ftagTruthOriginLabel") == 4),
                "n_hf_tracks_ghost": sum(
                    1 for t in tracks if t.get("ftagTruthOriginLabel") in {3, 4, 5}
                ),
            }
            for field, value in track_label_counts.items():
                if field in _fields(jet_rows):
                    jet_rows[field][global_jet_index] = value

            cone_label = int(cone_labels[local_jet_index])
            cone_ext = int(cone_ext_labels[local_jet_index])
            if "HadronConeExclTruthLabelID" in _fields(jet_rows):
                jet_rows["HadronConeExclTruthLabelID"][global_jet_index] = cone_label
            if "HadronConeExclExtendedTruthLabelID" in _fields(jet_rows):
                jet_rows["HadronConeExclExtendedTruthLabelID"][global_jet_index] = cone_ext
            if "ptFromTruthDressedWZJet" in _fields(jet_rows):
                jet_rows["ptFromTruthDressedWZJet"][global_jet_index] = jet_pt_mev[local_jet_index]
            cone_pdg = int(cone_label_pdg_ids[local_jet_index])
            if "HadronConeExclTruthLabelPdgId" in _fields(jet_rows):
                jet_rows["HadronConeExclTruthLabelPdgId"][global_jet_index] = cone_pdg
            # ATLAS' ghost-matched PdgId comes from re-clustering with ghost
            # particles, which we don't perform. For single-heavy-hadron jets
            # the value coincides with the cone match, so we alias it. This
            # populates the Salt `jets_bccharge` task input with real data
            # instead of an all-zero placeholder.
            if "HadronGhostInitialTruthLabelPdgId" in _fields(jet_rows):
                jet_rows["HadronGhostInitialTruthLabelPdgId"][global_jet_index] = cone_pdg
            if "HadronConeExclTruthLabelChildPdgId" in _fields(jet_rows):
                jet_rows["HadronConeExclTruthLabelChildPdgId"][global_jet_index] = (
                    _child_heavy_flavour_pdg(
                        int(cone_match_mc[local_jet_index]),
                        ev.pdg_ids,
                        children_map,
                    )
                )
            if "LeptonDecayLabel" in _fields(jet_rows):
                jet_rows["LeptonDecayLabel"][global_jet_index] = _lepton_decay_label_for_hadron(
                    int(cone_match_mc[local_jet_index]),
                    ev.pdg_ids,
                    children_map,
                )
            # HadronGhost*/PartonTruthLabelID require true ghost-association in
            # the jet clustering, which the FCC Delphes sample doesn't expose.
            # Leave them at their schema fillvalue (-1) instead of writing a
            # misleading approximation derived from constituent ancestry.

            for slot, track in enumerate(tracks):
                _fill_track_slot(track_rows, rich_track_rows, global_jet_index, slot, track)
                event_track_mcs.append(((global_jet_index, slot), int(track.get("_mc_index", -1))))

            for slot, flow in enumerate(flows):
                for field in ("pt", "energy", "deta", "dphi", "isCharged"):
                    if field in _fields(flow_rows):
                        flow_rows[field][global_jet_index, slot] = flow[field]
                if "valid" in _fields(flow_rows):
                    _ensure_finite_training_defaults(
                        flow_rows, (global_jet_index, slot), FLOW_TARGET_FIELDS
                    )
                    flow_rows["valid"][global_jet_index, slot] = True

            for slot, electron in enumerate(electrons):
                for field in (
                    "pt",
                    "eta",
                    "phi",
                    "abs_eta",
                    "ftag_et",
                    "ptfrac",
                    "ptrel",
                    "dr",
                    "qOverP",
                    "d0RelativeToBeamspot",
                    "d0RelativeToBeamspotSignificance",
                    "ftag_ptVarCone30OverPt",
                    "ftag_deltaPOverP",
                    "ftag_energyOverP",
                ):
                    if field in _fields(electron_rows):
                        electron_rows[field][global_jet_index, slot] = electron[field]
                if "valid" in _fields(electron_rows):
                    _ensure_finite_training_defaults(
                        electron_rows,
                        (global_jet_index, slot),
                        ELECTRON_TARGET_FIELDS,
                    )
                    electron_rows["valid"][global_jet_index, slot] = True

            for slot, muon in enumerate(muons):
                for field in ("pt", "eta", "phi", "ptfrac", "ptrel", "dr", "qOverP"):
                    if field in _fields(muon_rows):
                        muon_rows[field][global_jet_index, slot] = muon[field]
                if "valid" in _fields(muon_rows):
                    muon_rows["valid"][global_jet_index, slot] = True

        if event_track_mcs and "ftagTruthVertexIndex" in _fields(track_rows):
            vtx_map = _assign_track_vertex_indices(
                event_track_mcs,
                ev.mc_vx,
                ev.mc_vy,
                ev.mc_vz,
            )
            for (gj, slot), vidx in vtx_map.items():
                track_rows["ftagTruthVertexIndex"][gj, slot] = vidx
                if "ftagTruthVertexIndex" in _fields(rich_track_rows):
                    rich_track_rows["ftagTruthVertexIndex"][gj, slot] = vidx

        offset = stop

    global_rows = _global_rows_from_jets(jet_rows)
    return (
        global_rows,
        jet_rows,
        track_rows,
        rich_track_rows,
        flow_rows,
        electron_rows,
        muon_rows,
        diagnostics,
    )
