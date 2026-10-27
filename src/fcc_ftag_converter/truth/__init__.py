"""MC-truth helpers: heavy-flavour ancestry, track-type/origin/vertex, jet cone labels.

Split into focused submodules (pdg, origin, vertex, track_type, cone); this package
re-exports the same names the rest of the converter has always imported.
"""

from __future__ import annotations

from fcc_ftag_converter.truth.cone import (
    _build_children_map,
    _child_heavy_flavour_pdg,
    _cone_exclusive_truth_labels,
    _is_weakly_decaying_heavy_hadron,
    _lepton_decay_label_for_hadron,
)
from fcc_ftag_converter.truth.origin import (
    _label_from_ancestors,
    _origin_label_from_ancestors,
)
from fcc_ftag_converter.truth.pdg import _heavy_flavour_from_pdg, _is_heavy_hadron_pdg
from fcc_ftag_converter.truth.track_type import _lepton_id_from_mc, _track_type_label_from_mc
from fcc_ftag_converter.truth.vertex import _assign_track_vertex_indices

__all__ = [
    "_assign_track_vertex_indices",
    "_build_children_map",
    "_child_heavy_flavour_pdg",
    "_cone_exclusive_truth_labels",
    "_heavy_flavour_from_pdg",
    "_is_heavy_hadron_pdg",
    "_is_weakly_decaying_heavy_hadron",
    "_label_from_ancestors",
    "_lepton_decay_label_for_hadron",
    "_lepton_id_from_mc",
    "_origin_label_from_ancestors",
    "_track_type_label_from_mc",
]
