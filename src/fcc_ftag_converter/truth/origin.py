"""MC parent-ancestry heavy-flavour labels (jet HadronConeExcl + track origin)."""

from __future__ import annotations

import numpy as np

from fcc_ftag_converter.truth.pdg import _heavy_flavour_from_pdg


def _label_from_ancestors(
    mc_index: int,
    pdg_ids: np.ndarray,
    parent_begin: np.ndarray,
    parent_end: np.ndarray,
    parent_index: np.ndarray,
    cache: dict[int, int],
) -> int:
    if mc_index in cache:
        return cache[mc_index]
    if mc_index < 0 or mc_index >= len(pdg_ids):
        return 0

    label = _heavy_flavour_from_pdg(int(pdg_ids[mc_index]))
    for relation_index in range(int(parent_begin[mc_index]), int(parent_end[mc_index])):
        if relation_index < 0 or relation_index >= len(parent_index):
            continue
        label = max(
            label,
            _label_from_ancestors(
                int(parent_index[relation_index]),
                pdg_ids,
                parent_begin,
                parent_end,
                parent_index,
                cache,
            ),
        )
        if label == 5:
            break

    cache[mc_index] = label
    return label


def _origin_label_from_ancestors(
    mc_index: int,
    pdg_ids: np.ndarray,
    parent_begin: np.ndarray,
    parent_end: np.ndarray,
    parent_index: np.ndarray,
    cache: dict[int, int],
) -> int:
    """Return ATLAS ftagTruthOriginLabel (0..7) for an MC particle.

    Walks the PARENTS of ``mc_index`` (not the particle itself) and looks for
    b-hadrons, c-hadrons and tau-leptons in the ancestry. Maps to:

    ===  =================  ============================================
    val  category           condition
    ===  =================  ============================================
    2    Primary            no heavy-flavour or tau ancestor found
    3    FromB              b-hadron ancestor, no c-hadron in between
    4    FromBC             both a b- and a c-hadron in the ancestry
    5    FromC              c-hadron ancestor, no b-hadron above
    6    FromTau            tau-lepton ancestor only
    7    OtherSecondary     no MC association at all
    ===  =================  ============================================

    Pileup (0) and Fake (1) are ATLAS-specific. Pileup doesn't exist in FCC-ee;
    Fake requires a truth-match-probability cut that EDM4hep doesn't expose,
    so it is collapsed into 7 (OtherSecondary).

    Parameters
    ----------
    mc_index : int
        Index of the particle whose ancestry is walked.
    pdg_ids : np.ndarray
        PDG id per truth particle.
    parent_begin : np.ndarray
        Start of each particle's slice into ``parent_index``.
    parent_end : np.ndarray
        End of each particle's slice into ``parent_index``.
    parent_index : np.ndarray
        Flat parent-link array.
    cache : dict[int, int]
        Memo of already-resolved indices, updated in place.

    Returns
    -------
    int
        One of the origin codes in the table above.
    """
    if mc_index < 0 or mc_index >= len(pdg_ids):
        return 7
    if mc_index in cache:
        return cache[mc_index]

    visited: set[int] = set()
    stack: list[int] = []
    for r in range(int(parent_begin[mc_index]), int(parent_end[mc_index])):
        if 0 <= r < len(parent_index):
            p = int(parent_index[r])
            if 0 <= p < len(pdg_ids):
                stack.append(p)

    has_b = has_c = has_tau = False
    while stack:
        idx = stack.pop()
        if idx in visited:
            continue
        visited.add(idx)
        pdg = abs(int(pdg_ids[idx]))
        if pdg == 15:
            has_tau = True
        if pdg >= 100:
            fla = _heavy_flavour_from_pdg(pdg)
            if fla == 5:
                has_b = True
            elif fla == 4:
                has_c = True
        for r in range(int(parent_begin[idx]), int(parent_end[idx])):
            if 0 <= r < len(parent_index):
                p = int(parent_index[r])
                if 0 <= p < len(pdg_ids):
                    stack.append(p)

    if has_b and has_c:
        label = 4  # FromBC
    elif has_b:
        label = 3  # FromB
    elif has_c:
        label = 5  # FromC
    elif has_tau:
        label = 6  # FromTau
    else:
        label = 2  # Primary
    cache[mc_index] = label
    return label
