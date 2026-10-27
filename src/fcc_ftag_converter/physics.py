"""Kinematics, track parameter conversion, and lepton-jet matching helpers."""

from __future__ import annotations

from typing import TypeVar

import numpy as np

# Unit conversions are shape-preserving: a float in, a float out.
_Numeric = TypeVar("_Numeric", float, np.ndarray)

EDM4HEP_TRACK_AT_OTHER = 0
EDM4HEP_TRACK_AT_IP = 1
GEV_TO_MEV = 1000.0
INV_GEV_TO_INV_MEV = 1.0 / GEV_TO_MEV
TRACK_OMEGA_TO_Q_OVER_PT = 0.000299792458  # GeV / (T mm)


def _wrap_delta_phi(phi: float | np.ndarray) -> float | np.ndarray:
    return (phi + np.pi) % (2.0 * np.pi) - np.pi


def _eta_from_components(px: np.ndarray, py: np.ndarray, pz: np.ndarray) -> np.ndarray:
    pt = np.hypot(px, py).astype(np.float32)
    eta = np.zeros_like(pt)
    np.divide(pz, pt, out=eta, where=pt > 0)
    return np.arcsinh(eta).astype(np.float32)


def _eta_phi_from_momentum(
    px: np.ndarray, py: np.ndarray, pz: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    pt = np.hypot(px, py)
    eta = np.zeros_like(pt, dtype=np.float32)
    np.divide(pz, pt, out=eta, where=pt > 0)
    return np.arcsinh(eta).astype(np.float32), np.arctan2(py, px).astype(np.float32)


def _ptrel(
    px: float,
    py: float,
    pz: float,
    jet_px: float,
    jet_py: float,
    jet_pz: float,
) -> float:
    jet_norm = float(np.sqrt(jet_px * jet_px + jet_py * jet_py + jet_pz * jet_pz))
    if jet_norm <= 0.0:
        return 0.0
    cross = np.cross(
        np.asarray([px, py, pz], dtype=np.float64),
        np.asarray([jet_px, jet_py, jet_pz], dtype=np.float64),
    )
    return float(np.linalg.norm(cross) / jet_norm)


def _cov_index(i: int, j: int) -> int:
    if i < j:
        i, j = j, i
    return i * (i + 1) // 2 + j


def _safe_sqrt(value: float) -> float:
    if not np.isfinite(value) or value <= 0.0:
        return 0.0
    return float(np.sqrt(value))


def _gev_to_mev(value: _Numeric) -> _Numeric:
    return value * GEV_TO_MEV


def _inverse_gev_to_inverse_mev(value: _Numeric) -> _Numeric:
    return value * INV_GEV_TO_INV_MEV


def _signed_lepton_id(abs_pdg_id: int, charge: float) -> int:
    if charge > 0.0:
        return -abs_pdg_id
    return abs_pdg_id


def _ensure_finite_training_defaults(
    array: np.ndarray, index: tuple[int, int] | int, fields: tuple[str, ...]
) -> None:
    names = array.dtype.names or ()
    for field in fields:
        if field not in names:
            continue
        # `names` being non-empty is what makes `.fields` non-None.
        dtype = array.dtype.fields[field][0]  # type: ignore[index]
        if np.issubdtype(dtype, np.floating) and not np.isfinite(array[field][index]):
            array[field][index] = 0.0


def _charged_pt_cone_over_pt(
    reco_index: int,
    reco_pt: np.ndarray,
    reco_eta: np.ndarray,
    reco_phi: np.ndarray,
    reco_charge: np.ndarray,
    *,
    radius: float = 0.3,
) -> float:
    if reco_index < 0 or reco_index >= len(reco_pt) or reco_pt[reco_index] <= 0.0:
        return 0.0
    deta = reco_eta - reco_eta[reco_index]
    dphi = _wrap_delta_phi(reco_phi - reco_phi[reco_index])
    cone = (
        (np.hypot(deta, dphi) < radius)
        & (np.arange(len(reco_pt)) != reco_index)
        & (np.abs(reco_charge) > 0.0)
    )
    return float(np.sum(reco_pt[cone]) / reco_pt[reco_index])


def _select_track_state(begin: int, end: int, locations: np.ndarray) -> int:
    """Prefer an EDM4hep AtIP state, then AtOther, then the first valid state.

    Parameters
    ----------
    begin : int
        Start of this track's states in ``locations``.
    end : int
        End of this track's states in ``locations``.
    locations : np.ndarray
        EDM4hep location code per track state.

    Returns
    -------
    int
        Index of the chosen state, -1 when the slice is empty.
    """
    if begin < 0 or end <= begin:
        return -1
    end = min(end, len(locations))
    if begin >= end:
        return -1
    for wanted in (EDM4HEP_TRACK_AT_IP, EDM4HEP_TRACK_AT_OTHER):
        for state_index in range(begin, end):
            if int(locations[state_index]) == wanted:
                return state_index
    return begin


def _q_over_p_from_omega(omega: float, tan_lambda: float, magnetic_field_t: float) -> float:
    """Convert EDM4hep helix curvature to signed q/p in MeV^-1.

    EDM4hep/LCIO-style omega is signed curvature in 1/mm. The FCC Delphes
    sample momenta are in GeV. Salt's ATLAS-like training surface stores
    momenta in MeV, so q/p is written in inverse MeV.

    Parameters
    ----------
    omega : float
        Signed curvature in 1/mm.
    tan_lambda : float
        Track dip angle.
    magnetic_field_t : float
        Solenoid field in tesla.

    Returns
    -------
    float
        Signed q/p in MeV^-1, 0.0 when the field is not positive.
    """
    if magnetic_field_t <= 0.0:
        return 0.0
    sin_theta = 1.0 / np.sqrt(1.0 + tan_lambda * tan_lambda)
    q_over_p_gev = omega * sin_theta / (TRACK_OMEGA_TO_Q_OVER_PT * magnetic_field_t)
    return float(_inverse_gev_to_inverse_mev(q_over_p_gev))


def _track_uncertainties(
    covariance: np.ndarray,
    *,
    z0: float,
    omega: float,
    tan_lambda: float,
    magnetic_field_t: float,
) -> dict[str, float]:
    """Propagate EDM4hep TrackState covariance to Salt track uncertainties.

    Parameters
    ----------
    covariance : np.ndarray
        Packed lower-triangular 5x5 covariance of the track state.
    z0 : float
        Longitudinal impact parameter in mm.
    omega : float
        Signed curvature in 1/mm.
    tan_lambda : float
        Track dip angle.
    magnetic_field_t : float
        Solenoid field in tesla.

    Returns
    -------
    dict[str, float]
        Salt uncertainty fields, empty when the covariance is not 21 elements.
    """
    if len(covariance) < 21:
        return {}

    var_d0 = float(covariance[_cov_index(0, 0)])
    var_phi = float(covariance[_cov_index(1, 1)])
    var_omega = float(covariance[_cov_index(2, 2)])
    var_z0 = float(covariance[_cov_index(3, 3)])
    var_tan = float(covariance[_cov_index(4, 4)])
    cov_omega_tan = float(covariance[_cov_index(4, 2)])
    cov_z0_tan = float(covariance[_cov_index(4, 3)])

    one_plus_tan2 = 1.0 + tan_lambda * tan_lambda
    sin_theta = 1.0 / np.sqrt(one_plus_tan2)
    d_sin_d_tan = -tan_lambda / (one_plus_tan2**1.5)

    z0_sin_d_z0 = sin_theta
    z0_sin_d_tan = z0 * d_sin_d_tan
    var_z0_sin = (
        z0_sin_d_z0 * z0_sin_d_z0 * var_z0
        + z0_sin_d_tan * z0_sin_d_tan * var_tan
        + 2.0 * z0_sin_d_z0 * z0_sin_d_tan * cov_z0_tan
    )

    var_q_over_p = 0.0
    if magnetic_field_t > 0.0:
        scale = 1.0 / (TRACK_OMEGA_TO_Q_OVER_PT * magnetic_field_t)
        q_over_p_d_omega = sin_theta * scale
        q_over_p_d_tan = omega * d_sin_d_tan * scale
        var_q_over_p = (
            q_over_p_d_omega * q_over_p_d_omega * var_omega
            + q_over_p_d_tan * q_over_p_d_tan * var_tan
            + 2.0 * q_over_p_d_omega * q_over_p_d_tan * cov_omega_tan
        )

    return {
        "d0Uncertainty": _safe_sqrt(var_d0),
        "z0SinThetaUncertainty": _safe_sqrt(var_z0_sin),
        "phiUncertainty": _safe_sqrt(var_phi),
        "thetaUncertainty": _safe_sqrt(var_tan) / one_plus_tan2,
        "qOverPUncertainty": float(_inverse_gev_to_inverse_mev(_safe_sqrt(var_q_over_p))),
    }


def _lepton_track_info(
    reco_index: int,
    reco_tracks_begin: np.ndarray,
    reco_tracks_end: np.ndarray,
    reco_track_index: np.ndarray,
    reco_track_collection: np.ndarray,
    track_states_begin: np.ndarray,
    track_states_end: np.ndarray,
    state_d0: np.ndarray,
    state_z0: np.ndarray,
    state_tan_lambda: np.ndarray,
    state_omega: np.ndarray,
    state_location: np.ndarray,
    state_covariance: np.ndarray,
    magnetic_field_t: float,
) -> dict[str, float]:
    """Return {qOverP, d0, d0_significance} for the linked track, or zeros.

    Parameters
    ----------
    reco_index : int
        Index of the reconstructed lepton.
    reco_tracks_begin : np.ndarray
        Start of each particle's slice into the track relation.
    reco_tracks_end : np.ndarray
        End of each particle's slice into the track relation.
    reco_track_index : np.ndarray
        Flat track relation: target index.
    reco_track_collection : np.ndarray
        Flat track relation: source collection id.
    track_states_begin : np.ndarray
        Start of each track's slice into the state arrays.
    track_states_end : np.ndarray
        End of each track's slice into the state arrays.
    state_d0 : np.ndarray
        Transverse impact parameter per track state.
    state_z0 : np.ndarray
        Longitudinal impact parameter per track state.
    state_tan_lambda : np.ndarray
        Dip angle per track state.
    state_omega : np.ndarray
        Signed curvature per track state.
    state_location : np.ndarray
        EDM4hep location code per track state.
    state_covariance : np.ndarray
        Packed covariance per track state.
    magnetic_field_t : float
        Solenoid field in tesla.

    Returns
    -------
    dict[str, float]
        The three fields above; all zero when no track is linked.
    """
    for track_rel in range(int(reco_tracks_begin[reco_index]), int(reco_tracks_end[reco_index])):
        if track_rel < 0 or track_rel >= len(reco_track_index):
            continue
        if reco_track_collection[track_rel] != 6:
            continue
        track_index = int(reco_track_index[track_rel])
        if track_index < 0 or track_index >= len(track_states_begin):
            continue
        state_index = _select_track_state(
            int(track_states_begin[track_index]),
            int(track_states_end[track_index]),
            state_location,
        )
        if state_index < 0 or state_index >= len(state_d0):
            continue
        tan_lambda = float(state_tan_lambda[state_index])
        omega = float(state_omega[state_index])
        q_over_p = _q_over_p_from_omega(omega, tan_lambda, magnetic_field_t)
        cov = (
            state_covariance[state_index]
            if state_index < len(state_covariance)
            else np.zeros(21, dtype=np.float32)
        )
        unc = _track_uncertainties(
            np.asarray(cov, dtype=np.float32),
            z0=float(state_z0[state_index]),
            omega=omega,
            tan_lambda=tan_lambda,
            magnetic_field_t=magnetic_field_t,
        )
        d0_unc = unc.get("d0Uncertainty", 0.0)
        return {
            "qOverP": q_over_p,
            "d0": float(state_d0[state_index]),
            "d0_significance": float(state_d0[state_index]) / d0_unc if d0_unc > 0.0 else 0.0,
        }
    return {"qOverP": 0.0, "d0": 0.0, "d0_significance": 0.0}


def _nearest_jet_lepton_match(
    lepton_reco_indices: set[int],
    reco_eta_all: np.ndarray,
    reco_phi_all: np.ndarray,
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
    max_dr: float,
) -> dict[int, int]:
    """Return {reco_index: local_jet_index} for leptons matched within max_dr.

    Parameters
    ----------
    lepton_reco_indices : set[int]
        Candidate lepton indices into the reco arrays.
    reco_eta_all : np.ndarray
        Pseudorapidity of every reconstructed particle.
    reco_phi_all : np.ndarray
        Azimuth of every reconstructed particle.
    jet_eta : np.ndarray
        Jet pseudorapidities for this event.
    jet_phi : np.ndarray
        Jet azimuths for this event.
    max_dr : float
        Largest accepted lepton-jet separation.

    Returns
    -------
    dict[int, int]
        Matched leptons only; unmatched ones are absent.
    """
    result: dict[int, int] = {}
    if len(jet_eta) == 0:
        return result
    for reco_index in lepton_reco_indices:
        if reco_index < 0 or reco_index >= len(reco_eta_all):
            continue
        deta = jet_eta - reco_eta_all[reco_index]
        dphi = _wrap_delta_phi(jet_phi - reco_phi_all[reco_index])
        dr = np.hypot(deta, dphi)
        nearest = int(np.argmin(dr))
        if float(dr[nearest]) < max_dr:
            result[reco_index] = nearest
    return result
