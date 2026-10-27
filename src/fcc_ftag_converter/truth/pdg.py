"""PDG-id helpers: heavy-flavour classification of a single particle."""

from __future__ import annotations


def _heavy_flavour_from_pdg(pdg_id: int) -> int:
    """Return Salt-style heavy-flavour label priority for one PDG id.

    Parameters
    ----------
    pdg_id : int
        PDG id of the particle.

    Returns
    -------
    int
        5 for beauty, 4 for charm, 0 otherwise. Beauty wins over charm.
    """
    apid = abs(int(pdg_id))
    if apid == 5:
        return 5
    if apid == 4:
        return 4
    if apid < 100:
        return 0

    digits = {
        (apid // 1000) % 10,
        (apid // 100) % 10,
        (apid // 10) % 10,
    }
    if 5 in digits:
        return 5
    if 4 in digits:
        return 4
    return 0


def _is_heavy_hadron_pdg(pdg_id: int) -> bool:
    return abs(int(pdg_id)) >= 100 and _heavy_flavour_from_pdg(pdg_id) in {4, 5}
