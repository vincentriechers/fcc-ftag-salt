"""ATLAS HadronConeExcl jet labelling and its hadron-descendant helpers."""

from __future__ import annotations

from operator import itemgetter

import numpy as np

from fcc_ftag_converter.physics import _eta_phi_from_momentum, _wrap_delta_phi
from fcc_ftag_converter.truth.origin import _label_from_ancestors
from fcc_ftag_converter.truth.pdg import _heavy_flavour_from_pdg, _is_heavy_hadron_pdg


def _build_children_map(
    parent_begin: np.ndarray,
    parent_end: np.ndarray,
    parent_index: np.ndarray,
    n_particles: int,
) -> dict[int, list[int]]:
    """Invert the parent relation into a children map (one O(N) scan per event).

    Parameters
    ----------
    parent_begin : np.ndarray
        Start of each particle's slice into ``parent_index``.
    parent_end : np.ndarray
        End of each particle's slice into ``parent_index``.
    parent_index : np.ndarray
        Flat parent-link array.
    n_particles : int
        Number of truth particles in the event.

    Returns
    -------
    dict[int, list[int]]
        Child indices keyed by parent index. Parents with no children are absent.
    """
    children: dict[int, list[int]] = {}
    for child_idx in range(n_particles):
        b = int(parent_begin[child_idx])
        e = int(parent_end[child_idx])
        for r in range(b, e):
            if 0 <= r < len(parent_index):
                p = int(parent_index[r])
                if 0 <= p < n_particles:
                    children.setdefault(p, []).append(child_idx)
    return children


def _lepton_decay_label_for_hadron(
    hadron_mc_index: int,
    pdg_ids: np.ndarray,
    children_map: dict[int, list[int]],
) -> int:
    """Walk descendants of one hadron/tau and encode ATLAS LeptonDecayLabel digits.

    ATLAS digits: 1 = electron, 2 = muon, 3 = tau (one digit per lepton flavour
    that appears in the decay chain). We do NOT distinguish "in both decays"
    (b-leg AND c-leg) -- that requires separating the b-decay from the c-leg,
    which is not robustly reconstructible from EDM4hep ancestry alone. So we
    produce 0/1/2/3 and the simple combinations (12/13/23/123); the
    "both-decay" extended codes (11/22/33/...) are out of scope here.

    Parameters
    ----------
    hadron_mc_index : int
        Index of the hadron or tau whose decay chain is walked.
    pdg_ids : np.ndarray
        PDG id per truth particle.
    children_map : dict[int, list[int]]
        Children map from :func:`_build_children_map`.

    Returns
    -------
    int
        The concatenated lepton-flavour digits, 0 when the chain is hadronic.
    """
    if hadron_mc_index < 0 or hadron_mc_index >= len(pdg_ids):
        return 0
    visited: set[int] = set()
    stack: list[int] = list(children_map.get(hadron_mc_index, []))
    has_e = has_mu = has_tau = False
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        pdg = abs(int(pdg_ids[idx]))
        if pdg == 11:
            has_e = True
        elif pdg == 13:
            has_mu = True
        elif pdg == 15:
            has_tau = True
        stack.extend(children_map.get(idx, []))

    label = 0
    if has_e:
        label = label * 10 + 1
    if has_mu:
        label = label * 10 + 2
    if has_tau:
        label = label * 10 + 3
    return label


def _is_weakly_decaying_heavy_hadron(
    mc_index: int,
    flavour: int,
    pdg_ids: np.ndarray,
    children_map: dict[int, list[int]],
) -> bool:
    """A heavy hadron is "weakly decaying" iff it has no daughter of the same
    heavy-flavour family (b or c). Excited states like B*->B gamma and cascades
    like B**->B pi are therefore filtered out, matching the ATLAS spec.

    Parameters
    ----------
    mc_index : int
        Index of the candidate hadron.
    flavour : int
        Its heavy flavour, 4 or 5.
    pdg_ids : np.ndarray
        PDG id per truth particle.
    children_map : dict[int, list[int]]
        Children map from :func:`_build_children_map`.

    Returns
    -------
    bool
        True when no daughter carries the same heavy flavour.
    """
    for child_idx in children_map.get(mc_index, ()):
        if not 0 <= child_idx < len(pdg_ids):
            continue
        if _heavy_flavour_from_pdg(int(pdg_ids[child_idx])) == flavour:
            return False
    return True


def _child_heavy_flavour_pdg(
    mc_index: int,
    pdg_ids: np.ndarray,
    children_map: dict[int, list[int]],
) -> int:
    """Return the PDG id of the dominant heavy-flavour daughter of ``mc_index``.

    For a B hadron this is typically the c-hadron from the b decay (used by
    ATLAS in ``HadronConeExclTruthLabelChildPdgId``). For a tau or c-hadron
    parent there is usually no further heavy-flavour daughter, so 0 is returned.
    Picks the daughter whose own flavour is the lightest heavy flavour
    (c preferred over b when both somehow appear).

    Parameters
    ----------
    mc_index : int
        Index of the parent particle.
    pdg_ids : np.ndarray
        PDG id per truth particle.
    children_map : dict[int, list[int]]
        Children map from :func:`_build_children_map`.

    Returns
    -------
    int
        Signed PDG id of the chosen daughter, 0 when there is none.
    """
    if mc_index < 0 or mc_index >= len(pdg_ids):
        return 0
    best_pdg = 0
    best_priority = -1  # prefer c (4) over b (5) since b->c
    for child_idx in children_map.get(mc_index, ()):
        if 0 <= child_idx < len(pdg_ids):
            cpdg = int(pdg_ids[child_idx])
            if _is_heavy_hadron_pdg(cpdg):
                fla = _heavy_flavour_from_pdg(cpdg)
                # We want c (=4) preferred; if no c, fall back to b
                priority = 2 if fla == 4 else 1
                if priority > best_priority:
                    best_priority = priority
                    best_pdg = cpdg
    return best_pdg


def _cone_exclusive_truth_labels(
    jet_eta: np.ndarray,
    jet_phi: np.ndarray,
    *,
    pdg_ids: np.ndarray,
    mc_px: np.ndarray,
    mc_py: np.ndarray,
    mc_pz: np.ndarray,
    parent_begin: np.ndarray,
    parent_end: np.ndarray,
    parent_index: np.ndarray,
    children_map: dict[int, list[int]],
    radius: float,
    hadron_pt_min_gev: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ATLAS-style truth labelling: returns ``(label, extended, pdg_id, match_mc_idx)`` per jet.

    Implements the full ATLAS HadronConeExcl scheme:
      - matches b- and c-hadrons AND tau-leptons with pT > ``hadron_pt_min_gev``
      - each candidate is assigned to its nearest jet within ``radius``
      - c-hadrons whose ancestry contains a b-hadron are dropped (handled as
        the c-leg of a b decay, not an independent c-jet)

    Output codes:

    ===========  =======================  ==================================
    label        extended_label           meaning
    ===========  =======================  ==================================
    0            0                        light jet
    5            5                        single b
    5            54                       b + extra c-hadron
    5            55                       two b-hadrons
    4            4                        single c
    4            44                       two c-hadrons
    15           15                       single tau
    15           1515                     two taus
    ===========  =======================  ==================================

    Parameters
    ----------
    jet_eta : np.ndarray
        Jet pseudorapidities, one entry per jet.
    jet_phi : np.ndarray
        Jet azimuths, one entry per jet.
    pdg_ids : np.ndarray
        PDG id per truth particle.
    mc_px : np.ndarray
        Truth-particle momentum x-component in GeV.
    mc_py : np.ndarray
        Truth-particle momentum y-component in GeV.
    mc_pz : np.ndarray
        Truth-particle momentum z-component in GeV.
    parent_begin : np.ndarray
        Start of each particle's slice into ``parent_index``.
    parent_end : np.ndarray
        End of each particle's slice into ``parent_index``.
    parent_index : np.ndarray
        Flat parent-link array.
    children_map : dict[int, list[int]]
        Children map from :func:`_build_children_map`.
    radius : float
        Cone radius used for the jet-hadron match.
    hadron_pt_min_gev : float, optional
        Minimum truth-particle pT considered for matching.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Per-jet ``(label, extended_label, pdg_id, matched_truth_index)``. The
        last is -1 where no candidate matched.
    """
    n_jets = len(jet_eta)
    labels = np.zeros(n_jets, dtype=np.int32)
    extended = np.zeros(n_jets, dtype=np.int32)
    pdg_labels = np.zeros(n_jets, dtype=np.int32)
    match_mc_indices = np.full(n_jets, -1, dtype=np.int64)
    if n_jets == 0 or len(pdg_ids) == 0:
        return labels, extended, pdg_labels, match_mc_indices

    mc_eta, mc_phi = _eta_phi_from_momentum(mc_px, mc_py, mc_pz)
    mc_pt = np.hypot(mc_px, mc_py)
    label_cache: dict[int, int] = {}
    matches: list[list[tuple[int, int, float, int]]] = [[] for _ in range(n_jets)]

    for mc_index, pdg in enumerate(pdg_ids):
        apid = abs(int(pdg))
        if apid == 15:
            flavour = 15
        elif _is_heavy_hadron_pdg(int(pdg)):
            flavour = _heavy_flavour_from_pdg(int(pdg))
            if not _is_weakly_decaying_heavy_hadron(mc_index, flavour, pdg_ids, children_map):
                continue
            ancestry = _label_from_ancestors(
                mc_index,
                pdg_ids,
                parent_begin,
                parent_end,
                parent_index,
                label_cache,
            )
            if flavour == 4 and ancestry == 5:
                continue
        else:
            continue
        if float(mc_pt[mc_index]) < hadron_pt_min_gev:
            continue

        deta = jet_eta - float(mc_eta[mc_index])
        dphi = _wrap_delta_phi(jet_phi - float(mc_phi[mc_index]))
        dr = np.hypot(deta, dphi)
        nearest = int(np.argmin(dr))
        nearest_dr = float(dr[nearest])
        if nearest_dr > radius:
            continue
        matches[nearest].append((flavour, int(pdg), nearest_dr, int(mc_index)))

    for j in range(n_jets):
        bucket = matches[j]
        if not bucket:
            continue
        bucket.sort(key=itemgetter(2))  # closest first
        n_b = sum(1 for m in bucket if m[0] == 5)
        n_c = sum(1 for m in bucket if m[0] == 4)
        n_tau = sum(1 for m in bucket if m[0] == 15)

        if n_b:
            labels[j] = 5
            if n_b >= 2:
                extended[j] = 55
            elif n_c >= 1:
                extended[j] = 54
            else:
                extended[j] = 5
            for m in bucket:
                if m[0] == 5:
                    pdg_labels[j] = m[1]
                    match_mc_indices[j] = m[3]
                    break
        elif n_c:
            labels[j] = 4
            extended[j] = 44 if n_c >= 2 else 4
            for m in bucket:
                if m[0] == 4:
                    pdg_labels[j] = m[1]
                    match_mc_indices[j] = m[3]
                    break
        elif n_tau:
            labels[j] = 15
            if n_tau >= 2:
                # ATLAS sub-classification: if exactly one of the matched taus
                # decays leptonically (->e or ->mu), encode 151511 / 151513.
                # Both-hadronic stays 1515. Both-leptonic stays 1515 (no spec).
                tau_decay_modes = []
                for m in bucket:
                    if m[0] != 15:
                        continue
                    mode = 0  # hadronic
                    for child in children_map.get(int(m[3]), ()):
                        if 0 <= child < len(pdg_ids):
                            apc = abs(int(pdg_ids[child]))
                            if apc == 11:
                                mode = 11
                                break
                            if apc == 13:
                                mode = 13
                                break
                    tau_decay_modes.append(mode)
                tau_decay_modes.sort()
                if tau_decay_modes == [0, 11]:
                    extended[j] = 151511
                elif tau_decay_modes == [0, 13]:
                    extended[j] = 151513
                else:
                    extended[j] = 1515
            else:
                extended[j] = 15
            for m in bucket:
                if m[0] == 15:
                    pdg_labels[j] = m[1]
                    match_mc_indices[j] = m[3]
                    break

    return labels, extended, pdg_labels, match_mc_indices
