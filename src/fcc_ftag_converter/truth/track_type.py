"""Per-track truth type label (ftagTruthTypeLabel) and lepton id."""

from __future__ import annotations

import numpy as np


def _track_type_label_from_mc(  # noqa: PLR0911  one return per ATLAS label
    mc_indices: list[int],
    pdg_ids: np.ndarray,
    parent_begin: np.ndarray | None = None,
    parent_end: np.ndarray | None = None,
    parent_index: np.ndarray | None = None,
) -> int:
    """Return ATLAS ftagTruthTypeLabel for the first matched truth particle.

    ATLAS convention (FTAG docs): for labels > 1 the SIGN of the value matches
    the electric-charge sign of the truth particle, NOT the PDG-id sign.
    - pi+/pi-  -> +-2   (PDG sign == charge sign)
    - K+/K-    -> +-3
    - Lambda   -> 4     (neutral -> no sign)
    - e-/e+    -> -+5    (PDG sign opposite to charge sign for leptons)
    - mu-/mu+  -> -+6
    - gamma    -> 7
    - other    -> 1
    - no MC    -> 0

    Parameters
    ----------
    mc_indices : list[int]
        Candidate truth-particle indices for this track, best match first.
    pdg_ids : np.ndarray
        PDG id per truth particle.
    parent_begin : np.ndarray | None, optional
        Start of each particle's slice into ``parent_index``.
    parent_end : np.ndarray | None, optional
        End of each particle's slice into ``parent_index``.
    parent_index : np.ndarray | None, optional
        Flat parent-link array, used for the Lambda-daughter fallback.

    Returns
    -------
    int
        The signed ftagTruthTypeLabel described above.
    """
    for mc_index in mc_indices:
        if mc_index < 0 or mc_index >= len(pdg_ids):
            continue
        pdg_id = int(pdg_ids[mc_index])
        apid = abs(pdg_id)
        if apid == 211:
            return 2 if pdg_id > 0 else -2
        if apid == 321:
            return 3 if pdg_id > 0 else -3
        if apid == 3122:
            return 4
        if apid == 11:
            return -5 if pdg_id > 0 else 5
        if apid == 13:
            return -6 if pdg_id > 0 else 6
        if apid == 22:
            return 7
        # ATLAS-style fallback: Lambdas decay before being tracked, so the truth
        # particle linked to the track is the proton/pion daughter. Check the
        # immediate parent -- if it's a Lambda, label as 4 (Lambda).
        if (
            apid in {2212, 211}
            and parent_begin is not None
            and parent_end is not None
            and parent_index is not None
        ):
            for r in range(int(parent_begin[mc_index]), int(parent_end[mc_index])):
                if 0 <= r < len(parent_index):
                    p = int(parent_index[r])
                    if 0 <= p < len(pdg_ids) and abs(int(pdg_ids[p])) == 3122:
                        return 4
    return 1 if mc_indices else 0


def _lepton_id_from_mc(mc_indices: list[int], pdg_ids: np.ndarray) -> int:
    for mc_index in mc_indices:
        if mc_index < 0 or mc_index >= len(pdg_ids):
            continue
        pdg_id = int(pdg_ids[mc_index])
        if abs(pdg_id) in {11, 13}:
            return pdg_id
    return 0
