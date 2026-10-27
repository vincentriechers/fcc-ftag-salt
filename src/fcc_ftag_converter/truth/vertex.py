"""Truth vertex indices from clustering MC production vertices."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TypeVar

import numpy as np

# The caller picks what identifies a track: the EDM4hep converter keys on a
# ``(jet, slot)`` pair, the unit tests on a plain int.
Uid = TypeVar("Uid", bound=Hashable)


def _assign_track_vertex_indices(
    track_mc_indices: list[tuple[Uid, int]],
    mc_vx: np.ndarray,
    mc_vy: np.ndarray,
    mc_vz: np.ndarray,
    *,
    merge_radius_mm: float = 0.1,
) -> dict[Uid, int]:
    """Cluster MC production vertices and map each track to a vertex index.

    ATLAS convention: 0 = PV (the most populous cluster, normally the hard IR),
    1, 2, ... = SVs in arbitrary order, -2 = no truth link.

    Parameters
    ----------
    track_mc_indices : list[tuple[Uid, int]]
        ``(track_uid, mc_index)`` pairs; a negative ``mc_index`` means no link.
    mc_vx : np.ndarray
        MC production-vertex x-coordinate in mm.
    mc_vy : np.ndarray
        MC production-vertex y-coordinate in mm.
    mc_vz : np.ndarray
        MC production-vertex z-coordinate in mm.
    merge_radius_mm : float, optional
        Vertices closer than this are treated as one.

    Returns
    -------
    dict[Uid, int]
        ``{track_uid: vertex_index}`` using the convention above.
    """
    if not track_mc_indices:
        return {}

    unique_mc = sorted({m for _, m in track_mc_indices if m >= 0 and m < len(mc_vx)})
    if not unique_mc:
        return {uid: -2 for uid, _ in track_mc_indices}

    clusters_xyz: list[np.ndarray] = []
    clusters_members: list[list[int]] = []
    mc_to_cluster: dict[int, int] = {}
    radius_sq = merge_radius_mm * merge_radius_mm

    for mc_idx in unique_mc:
        xyz = np.asarray([mc_vx[mc_idx], mc_vy[mc_idx], mc_vz[mc_idx]], dtype=np.float32)
        assigned_cluster = -1
        for ci, cx in enumerate(clusters_xyz):
            d = xyz - cx
            if float(d @ d) < radius_sq:
                assigned_cluster = ci
                break
        if assigned_cluster < 0:
            assigned_cluster = len(clusters_xyz)
            clusters_xyz.append(xyz)
            clusters_members.append([])
        clusters_members[assigned_cluster].append(mc_idx)
        mc_to_cluster[mc_idx] = assigned_cluster

    # Largest cluster becomes the PV (index 0); rest are SVs (1, 2, ...).
    biggest = max(range(len(clusters_members)), key=lambda ci: len(clusters_members[ci]))
    new_index = {biggest: 0}
    next_idx = 1
    for ci in range(len(clusters_xyz)):
        if ci == biggest:
            continue
        new_index[ci] = next_idx
        next_idx += 1

    result: dict[Uid, int] = {}
    for uid, mc_idx in track_mc_indices:
        if mc_idx < 0 or mc_idx not in mc_to_cluster:
            result[uid] = -2
        else:
            result[uid] = new_index[mc_to_cluster[mc_idx]]
    return result
