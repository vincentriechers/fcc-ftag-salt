"""Convert FCC-ee *higgs_decay* flat jet-flavour ntuples into Salt-shaped HDF5.

These are the central ``pre_summer2026`` samples
(``/eos/experiment/fcc/ee/jet_flavour_tagging/.../chunks/stage2_<decay>_<n>.root``).

Unlike the EDM4hep path (``cli.py`` + ``reader.py`` + ``physics.py``), the
higgs_decay ntuples are **already at feature level**: every ``pfcand_*`` variable is
precomputed. So there is no physics to redo here — this module is a thin
repackager that

  * builds a single ``flavour_label`` integer from the per-jet ghost hadron and
    parton labels, falling back to the 7-way one-hot
    (``recojet_is{G,U,D,S,C,B,TAU}``) on ntuples produced before those existed,
  * pads the jagged ``pfcand_*`` arrays into a fixed ``(n_jets, max_pfcand)``
    block with a boolean ``valid`` mask,
  * writes a Salt-native ``jets`` + ``pfcand`` H5 that UPP / Salt can consume.

The target schema is FCC-native: the ``pfcand`` group fields *are* the 35
features, not ATLAS ``tracks_ghost`` names. See ``configs/upp/variables-fcc-higgs_decay.yaml``
and ``configs/salt/gn3.yaml`` for the matching downstream configs.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import awkward as ak
import numpy as np
import uproot

from fcc_ftag_converter.schema import DatasetSpec, SaltH5Writer

# ----------------------------------------------------------------------------
# Schema definition
# ----------------------------------------------------------------------------

#: The 35 per-constituent input features, in the reference ntuple order. These become the
#: float32 fields of the ``pfcand`` group (plus a ``valid`` mask).
PFCAND_FEATURES: tuple[str, ...] = (
    "pfcand_erel_log",
    "pfcand_thetarel",
    "pfcand_phirel",
    "pfcand_dptdpt",
    "pfcand_detadeta",
    "pfcand_dphidphi",
    "pfcand_dxydxy",
    "pfcand_dzdz",
    "pfcand_dxydz",
    "pfcand_dphidxy",
    "pfcand_dlambdadz",
    "pfcand_dxyc",
    "pfcand_dxyctgtheta",
    "pfcand_phic",
    "pfcand_phidz",
    "pfcand_phictgtheta",
    "pfcand_cdz",
    "pfcand_cctgtheta",
    "pfcand_mtof",
    "pfcand_dndx",
    "pfcand_charge",
    "pfcand_isMu",
    "pfcand_isEl",
    "pfcand_isChargedHad",
    "pfcand_isGamma",
    "pfcand_isNeutralHad",
    "pfcand_dxy",
    "pfcand_dz",
    "pfcand_btagSip2dVal",
    "pfcand_btagSip2dSig",
    "pfcand_btagSip3dVal",
    "pfcand_btagSip3dSig",
    "pfcand_btagJetDistVal",
    "pfcand_btagJetDistSig",
    "pfcand_type",
)

#: Constituent kinematics for salt's ParT-style pairwise (edge) features, which
#: need jet-relative CARTESIAN angles plus a momentum-like scale
#: (salt/utils/edge_features.py: dR, kt, z). The reference features above offer
#: neither: pfcand_thetarel / pfcand_phirel are POLAR coordinates about the jet
#: axis (opening angle, azimuth around it), over which sqrt(dtheta^2 + dphi^2)
#: is not an angular distance, and pfcand_erel_log is log10(E / E_jet). The
#: angles are derived from the stage2 theta / phi branches the way jet_eta is;
#: the energy is copied, and carries no information the 35 do not already have
#: (pfcand_e == jet_e * 10**pfcand_erel_log, bit for bit).
#:
#: The energy, NOT a momentum, deliberately: Delphes hands every track the
#: GENERATOR mass, so E**2 - p**2 is the true particle mass to four digits
#: (measured on the 260918 stage2: 0.1396 for truth pions, 0.4936 for truth
#: kaons, 0.9383 for protons, 100% of constituents). With the energy already
#: recoverable from erel_log and the jet energy, any momentum magnitude --
#: pt, p, q/p, curvature -- would turn into truth-level hadron identification.
#: Never add one.
PFCAND_KINEMATICS: tuple[str, ...] = (
    "pfcand_e",  # copied
    "pfcand_deta",  # eta(theta) - jet_eta
    "pfcand_dphi",  # phi - jet_phi, wrapped to [-pi, pi)
)

#: Read for :data:`PFCAND_KINEMATICS`. In the stage2 of FCCAnalyses#524 and
#: later.
PFCAND_KINEMATIC_BRANCHES: tuple[str, ...] = ("pfcand_e", "pfcand_theta", "pfcand_phi")

#: Lifetime features where stage2 (current FCCAnalyses master) fills failed
#: track fits with covariance-derived sentinels (|x| ~ 1e12-1e15) instead of
#: the -9 used for neutrals (and used throughout the reference ParT ntuples).
#: Remapped to -9 at conversion: the huge values would otherwise dominate the
#: downstream mean/std normalisation and numerically erase the real values.
IP_SENTINEL_FEATURES: tuple[str, ...] = (
    "pfcand_dxy",
    "pfcand_dz",
    "pfcand_btagSip2dVal",
    "pfcand_btagSip2dSig",
    "pfcand_btagSip3dVal",
    "pfcand_btagSip3dSig",
    "pfcand_btagJetDistVal",
    "pfcand_btagJetDistSig",
)
# Genuine tails top out ~3e6 (Sip significances); above that only the no-track
# sentinels (~1e12+) and rare failed-geometry values (~1e7-1e9) occur.
IP_SENTINEL_ABOVE = 1e7
IP_FILL = np.float32(-9.0)

#: Per-constituent truth LABELS (GN3 aux-task targets), written as int32 and
#: only when present in the input ntuple. Produced by the FCCAnalyses
#: JetTruthLabelHelper (HEP-FCC/FCCAnalyses#549-554). OPTIONAL by design so this
#: converter still runs on older ntuples that lack them -- older files simply
#: omit these H5 fields. The three legacy names at the end come from the
#: pre-ghost weaver and are renamed on write, see :data:`LEGACY_H5_NAMES`.
PFCAND_LABELS: tuple[str, ...] = (
    "pfcand_truthOriginLabel",  # 2..7, -1 no MC link        -> salt track_origin
    "pfcand_truthTypeLabel",  # signed ATLAS type encoding   -> salt track_type
    "pfcand_truthSourceLabel",  # secondary-vertex source    (carried, no head yet)
    "pfcand_truthVertexIndex",  # 0 = MC PV, -2 no MC link   -> salt track_vertexing
    "pfcand_truthPdgId",  # matched MC PDG, 0 unmatched      (carried, no head yet)
    "pfcand_truthOrigin",  # legacy -> pfcand_truthOriginLabel
    "pfcand_truthVertex",  # legacy -> pfcand_truthVertexIndex
    "pfcand_truthPID",  # legacy -> pfcand_truthPdgId
)

#: OPTIONAL jet-level truth, all of the FCCAnalyses ghost-association and
#: truth-jet branches. Detected per input like :data:`PFCAND_LABELS`, so the
#: converter still runs on older ntuples; the two legacy names at the end are
#: renamed on write.
JET_TRUTH_SCALARS: tuple[str, ...] = (
    "jet_HadronGhostTruthLabelID",  # 0 light, 4 c, 5 b, 15 tau
    "jet_HadronGhostExtendedTruthLabelID",  # adds 44 cc, 54 bc, 55 bb, 1515 tautau
    "jet_HadronGhostTruthLabelPdgId",  # PDG of the labelling hadron, 0 for light
    "jet_HadronGhostTruthLabelPt",  # pT of that hadron, NaN for light
    "jet_HadronGhostInitialTruthLabelID",  # same, production-flavour hadron
    "jet_HadronGhostInitialExtendedTruthLabelID",
    "jet_HadronGhostInitialTruthLabelPdgId",  # -> salt jets_bccharge
    "jet_GhostBHadronsFinalCount",
    "jet_GhostCHadronsFinalCount",
    "jet_PartonTruthLabelID",  # -1 none, 1 d, 2 u, 3 s, 4 c, 5 b, 21 gluon
    "jet_PartonTruthLabelPt",
    "jet_PartonTruthLabelDR",
    "jet_truthJetPt",  # matched truth jet, NaN if unmatched
    "jet_truthJetEnergy",
    "jet_truthJetDR",
    "jet_truthJetPurity",  # reco jet energy fraction inside its truth jet
    "jet_truthJetCompleteness",  # truth jet energy fraction covered by the reco jet
    "jet_hadronPdgId",  # legacy -> jet_HadronGhostInitialTruthLabelPdgId
    "jet_truthPt",  # legacy -> jet_truthJetPt
)

#: Jet truth branches written as int32 rather than float32. stage2 stores every
#: branch as float, so the integer ones have to be named rather than guessed:
#: salt's label_map lookups (jets_bccharge) need exact ints.
JET_TRUTH_INTS: frozenset[str] = frozenset({
    "jet_HadronGhostTruthLabelID",
    "jet_HadronGhostExtendedTruthLabelID",
    "jet_HadronGhostTruthLabelPdgId",
    "jet_HadronGhostInitialTruthLabelID",
    "jet_HadronGhostInitialExtendedTruthLabelID",
    "jet_HadronGhostInitialTruthLabelPdgId",
    "jet_GhostBHadronsFinalCount",
    "jet_GhostCHadronsFinalCount",
    "jet_PartonTruthLabelID",
})

#: H5 output names for branches the pre-ghost weaver spelled differently (ROOT
#: branch name -> H5 field name), so one set of downstream configs works on
#: ntuples of either vintage. This renames only: the legacy branches keep their
#: own ENCODING (no-MC-link was 7 for the origin and -1 for the vertex, and the
#: vertex index was not anchored to the MC primary vertex), which the new names
#: define differently. Old ntuples therefore stay self-consistent but are not
#: interchangeable with new ones inside a single training sample.
LEGACY_H5_NAMES = {
    "pfcand_truthOrigin": "pfcand_truthOriginLabel",
    "pfcand_truthVertex": "pfcand_truthVertexIndex",
    "pfcand_truthPID": "pfcand_truthPdgId",
    "jet_hadronPdgId": "jet_HadronGhostInitialTruthLabelPdgId",
    "jet_truthPt": "jet_truthJetPt",
}


def _h5_label(name: str) -> str:
    return LEGACY_H5_NAMES.get(name, name)


#: Fill value for label slots with no constituent (padding beyond n_pfcand).
#: Also what ``pfcand_truthOriginLabel`` uses for "no MC link", which is
#: deliberate: salt pins ``ignore_index = -1`` on every classification loss and
#: VertexingTask drops any pair with a negative label, so padding and unmatched
#: constituents are excluded from the aux-task losses in exactly the same way.
LABEL_FILL = -1

#: Per-constituent track-type label. Read straight from the input when the
#: FCCAnalyses JetTruthLabelHelper produced it; on older ntuples that carry only
#: a matched PDG it is derived here instead, under the same field name.
PFCAND_TYPE_LABEL = "pfcand_truthTypeLabel"


def _pfcand_type_from_pid(pid: ak.Array) -> ak.Array:
    """Map matched-MC PDG -> ATLAS ftagTruthTypeLabel encoding (sign-aware).

    Legacy path only: newer ntuples carry ``pfcand_truthTypeLabel`` from the
    FCCAnalyses JetTruthLabelHelper and this is not called. pi=+/-2, K=+/-3,
    Lambda=4, e=-/+5, mu=-/+6, photon=7, everything else=1 (Other). Mirrors
    ``fcc_ftag_converter.truth._track_type_label_from_mc``; the salt config's
    ``label_map`` folds these into the 6 physics classes.

    Parameters
    ----------
    pid : ak.Array
        Matched-MC PDG id per constituent.

    Returns
    -------
    ak.Array
        The signed track-type label, same layout as ``pid``.
    """
    apid = abs(pid)
    t = ak.full_like(pid, 1)  # Other
    t = ak.where(apid == 211, ak.where(pid > 0, 2, -2), t)
    t = ak.where(apid == 321, ak.where(pid > 0, 3, -3), t)
    t = ak.where(apid == 3122, 4, t)
    t = ak.where(apid == 11, ak.where(pid > 0, -5, 5), t)
    t = ak.where(apid == 13, ak.where(pid > 0, -6, 6), t)
    return ak.where(apid == 22, 7, t)


#: Jet-level scalar branches copied straight through.
JET_SCALARS: tuple[str, ...] = (
    "jet_p",
    "jet_e",
    "jet_mass",
    "jet_phi",
    "jet_theta",
    "jet_nconst",
    "jet_nmu",
    "jet_nel",
    "jet_nchad",
    "jet_ngamma",
    "jet_nnhad",
    "jet_npfcand",
)

#: The 7 one-hot truth branches. Index in this tuple == ``flavour_label`` value
#: and == the Salt ``class_names`` order; keep both configs in sync with this.
ONEHOT_LABELS: tuple[str, ...] = (
    "recojet_isU",  # 0  ujets
    "recojet_isD",  # 1  djets
    "recojet_isS",  # 2  sjets
    "recojet_isC",  # 3  cjets
    "recojet_isB",  # 4  bjets
    "recojet_isG",  # 5  gjets
    "recojet_isTAU",  # 6  taujets
)
FLAVOUR_NAMES: tuple[str, ...] = (
    "ujets",
    "djets",
    "sjets",
    "cjets",
    "bjets",
    "gjets",
    "taujets",
)

#: The two branches ``flavour_label`` is built from when they are available.
GHOST_FLAVOUR_BRANCHES: tuple[str, ...] = (
    "jet_HadronGhostTruthLabelID",
    "jet_PartonTruthLabelID",
)

#: Ghost hadron label -> ``flavour_label``. Light (0) is not here: it needs the
#: parton label to tell u, d, s and gluon apart.
HADRON_FLAVOUR_LABELS = {5: 4, 4: 3, 15: 6}

#: Parton label -> ``flavour_label``, for jets the hadron label calls light.
#: PDG has 1 = d and 2 = u, in that order. atlas-ftag-tools has those two the
#: other way round in the `partonic` category of its ftag/flavours.yaml; this
#: follows PDG, so do not "fix" it against that file.
LIGHT_PARTON_LABELS = {2: 0, 1: 1, 3: 2, 21: 5}

DEFAULT_TREE = "tree"
# 75 matches the PF-candidate cap used in the reference FCC trainings (per
# upstream). The higgs_decay chunks observed so far peak at ~80 constituents (p95~48),
# so only a small tail is clipped; matching the cap keeps results comparable.
DEFAULT_MAX_PFCAND = 75
DEFAULT_CHUNK_SIZE = 20_000


def _jet_truth_dtype(name: str) -> str:
    """Return the H5 dtype string for one jet truth branch.

    Parameters
    ----------
    name : str
        ROOT branch name, legacy spelling accepted.

    Returns
    -------
    str
        ``"i4"`` for the label / PDG / count branches, ``"f4"`` for the rest.
    """
    return "i4" if _h5_label(name) in JET_TRUTH_INTS else "f4"


def _jets_dtype(truth_scalars: Sequence[str] = ()) -> np.dtype:
    fields = [("flavour_label", "i4")]
    # one-hot kept too: it is the file-name labelling, so comparing it against
    # flavour_label is what quantifies how wrong that labelling was.
    fields += [(name, "i4") for name in ONEHOT_LABELS]
    fields += [(name, "f4") for name in JET_SCALARS]
    fields += [(_h5_label(name), _jet_truth_dtype(name)) for name in truth_scalars]
    # derived kinematics for UPP region cuts (FCC jets are given as p, theta).
    fields += [("jet_pt", "f4"), ("jet_eta", "f4"), ("n_pfcand", "i4")]
    # Global running index used as the UPP split key (train/val/test on
    # ``eventNumber % 20``; see split-fcc-higgs_decay.yaml). Named ``eventNumber`` to
    # that name is in UPP's class_dict ignore
    # list, so this high-cardinality field is loaded for the split cut but NOT
    # fed into np.unique/combine_class_dict (which is O(n^2) and otherwise hangs
    # the normalisation stage for hours). higgs_decay ntuples have no real event id,
    # so this is a per-jet counter, not a physics event number.
    fields += [("eventNumber", "i8")]
    return np.dtype(fields)


def _pfcand_dtype(label_names: Sequence[str] = ()) -> np.dtype:
    fields = [("valid", "?")]
    fields += [(name, "f4") for name in (*PFCAND_FEATURES, *PFCAND_KINEMATICS)]
    fields += [(_h5_label(name), "i4") for name in label_names]
    return np.dtype(fields)


def _detect_present(inputs: Sequence[str], tree: str, names: Sequence[str]) -> tuple[str, ...]:
    """Return the subset of ``names`` that exist as branches in the input tree.

    Parameters
    ----------
    inputs : Sequence[str]
        Input files; only the first is opened.
    tree : str
        Tree name inside the file.
    names : Sequence[str]
        Candidate branch names, in the order to preserve.

    Returns
    -------
    tuple[str, ...]
        Those of ``names`` present in the tree.
    """
    with uproot.open(inputs[0]) as f:
        keys = set(f[tree].keys())
    return tuple(name for name in names if name in keys)


def _drop_legacy_duplicates(names: Sequence[str]) -> tuple[str, ...]:
    """Keep one branch per H5 field name, preferring the first spelling seen.

    :data:`PFCAND_LABELS` and :data:`JET_TRUTH_SCALARS` list the current names
    before the legacy ones, so a file carrying both is read under the current
    spelling and the legacy alias is dropped rather than colliding in the dtype.

    Parameters
    ----------
    names : Sequence[str]
        Branch names present in the input, in preference order.

    Returns
    -------
    tuple[str, ...]
        Those names whose H5 field is not already claimed by an earlier one.
    """
    seen: set[str] = set()
    kept = []
    for name in names:
        field = _h5_label(name)
        if field not in seen:
            seen.add(field)
            kept.append(name)
    return tuple(kept)


def _detect_pfcand_labels(inputs: Sequence[str], tree: str) -> tuple[str, ...]:
    """Return the subset of :data:`PFCAND_LABELS` present in the input tree.

    Parameters
    ----------
    inputs : Sequence[str]
        Input files; only the first is opened.
    tree : str
        Tree name inside the file.

    Returns
    -------
    tuple[str, ...]
        The labels available for this input, legacy aliases resolved.
    """
    return _drop_legacy_duplicates(_detect_present(inputs, tree, PFCAND_LABELS))


def has_ghost_labels(names: Sequence[str]) -> bool:
    """Say whether ``names`` carries everything :func:`_flavour_label_from_ghost` needs.

    Parameters
    ----------
    names : Sequence[str]
        Branch or field names available.

    Returns
    -------
    bool
        True when both of :data:`GHOST_FLAVOUR_BRANCHES` are there.
    """
    return all(name in names for name in GHOST_FLAVOUR_BRANCHES)


def _detect_jet_truth(inputs: Sequence[str], tree: str) -> tuple[str, ...]:
    """Return the subset of :data:`JET_TRUTH_SCALARS` present in the input tree.

    Parameters
    ----------
    inputs : Sequence[str]
        Input files; only the first is opened.
    tree : str
        Tree name inside the file.

    Returns
    -------
    tuple[str, ...]
        The jet truth branches available for this input, legacy aliases resolved.
    """
    return _drop_legacy_duplicates(_detect_present(inputs, tree, JET_TRUTH_SCALARS))


JETS_DTYPE = _jets_dtype()
PFCAND_DTYPE = _pfcand_dtype()


# ----------------------------------------------------------------------------
# Per-chunk row building
# ----------------------------------------------------------------------------


def _padded(arrays: ak.Array, name: str, max_pfcand: int, fill: float) -> np.ndarray:
    """Return one jagged constituent branch as a ``(n_jets, max_pfcand)`` block.

    Parameters
    ----------
    arrays : ak.Array
        One chunk of the ntuple.
    name : str
        Branch to take.
    max_pfcand : int
        Constituent slots per jet; longer jets are clipped.
    fill : float
        Value for the padding slots.

    Returns
    -------
    np.ndarray
        The padded block, dtype as the branch; the caller casts.
    """
    padded = ak.pad_none(arrays[name], max_pfcand, clip=True)
    return ak.to_numpy(ak.fill_none(padded, fill))


def _eta(theta: np.ndarray) -> np.ndarray:
    """Pseudorapidity from the polar angle.

    Parameters
    ----------
    theta : np.ndarray
        Polar angle.

    Returns
    -------
    np.ndarray
        ``-ln tan(theta / 2)``, with ``theta`` clamped off the exact poles.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        return -np.log(np.tan(np.clip(theta, 1e-6, np.pi - 1e-6) / 2.0))


def _flavour_label_from_onehot(arrays: ak.Array) -> np.ndarray:
    """Collapse the 7 one-hot truth branches into one int index (0..6).

    FCCAnalyses stage2 sets these from the input FILE NAME, so every jet in an
    ``Hbb`` file comes out a b-jet. Only used on ntuples without the ghost
    labels; :func:`_flavour_label_from_ghost` is the real thing.

    Parameters
    ----------
    arrays : ak.Array
        Chunk carrying the :data:`ONEHOT_LABELS` branches.

    Returns
    -------
    np.ndarray
        Per-jet label index, or -1 where the one-hot row is not exactly one hot
        so callers can detect and exclude malformed truth.
    """
    onehot = np.stack(
        [ak.to_numpy(arrays[name]).astype(np.int32) for name in ONEHOT_LABELS],
        axis=1,
    )
    row_sum = onehot.sum(axis=1)
    label = np.argmax(onehot, axis=1).astype(np.int32)
    label[row_sum != 1] = -1
    return label


def _flavour_label_from_ghost(arrays: ak.Array) -> np.ndarray:
    """Build the per-jet flavour from the ghost hadron and parton labels.

    The hadron label settles b, c and tau on its own. It cannot separate u, d,
    s and gluon, which all come out as 0 (light), so those four are taken from
    the parton label instead. A jet that is neither heavy-flavour nor a light
    jet with a light-quark or gluon parton gets -1 and is excluded downstream.

    Parameters
    ----------
    arrays : ak.Array
        Chunk carrying :data:`GHOST_FLAVOUR_BRANCHES`.

    Returns
    -------
    np.ndarray
        Per-jet label index in :data:`ONEHOT_LABELS` order, or -1.
    """
    # stage2 writes every branch as float32. Comparing in float space rather
    # than casting keeps a NaN from becoming INT_MIN: it simply matches nothing
    # and the jet falls through to -1.
    had = ak.to_numpy(arrays["jet_HadronGhostTruthLabelID"]).astype(np.float64)
    parton = ak.to_numpy(arrays["jet_PartonTruthLabelID"]).astype(np.float64)

    label = np.full(had.shape, -1, dtype=np.int32)
    for hadron_id, index in HADRON_FLAVOUR_LABELS.items():
        label[had == hadron_id] = index
    light = had == 0
    for parton_id, index in LIGHT_PARTON_LABELS.items():
        label[light & (parton == parton_id)] = index
    return label


def _flavour_label(arrays: ak.Array) -> np.ndarray:
    """Build the per-jet flavour, from the ghost labels where they exist.

    Parameters
    ----------
    arrays : ak.Array
        One chunk of the ntuple.

    Returns
    -------
    np.ndarray
        Per-jet label index in :data:`ONEHOT_LABELS` order, or -1.
    """
    if has_ghost_labels(arrays.fields):
        return _flavour_label_from_ghost(arrays)
    return _flavour_label_from_onehot(arrays)


def build_chunk_rows(
    arrays: ak.Array,
    *,
    max_pfcand: int,
    id_offset: int = 0,
    label_names: Sequence[str] = (),
    jet_truth_names: Sequence[str] = (),
) -> tuple[np.ndarray, np.ndarray, int]:
    """Build ``(jets_rows, pfcand_rows, n_truncated)`` for one chunk.

    ``jets_rows`` has shape ``(n_jets,)`` and dtype :func:`_jets_dtype`
    ``(jet_truth_names)``; ``pfcand_rows`` has shape ``(n_jets, max_pfcand)``
    and dtype :func:`_pfcand_dtype` ``(label_names)``. Constituents beyond
    ``max_pfcand`` are dropped and counted into ``n_truncated``. ``id_offset``
    seeds the global ``eventNumber``. ``label_names`` are the optional int
    truth labels (subset of :data:`PFCAND_LABELS`) to carry through; padding
    slots get :data:`LABEL_FILL`. ``jet_truth_names`` are the optional jet-level
    truth targets (subset of :data:`JET_TRUTH_SCALARS`).

    Parameters
    ----------
    arrays : ak.Array
        One chunk of the higgs_decay ntuple.
    max_pfcand : int
        Constituent slots per jet.
    id_offset : int, optional
        Seed for the global ``eventNumber``.
    label_names : Sequence[str], optional
        Optional per-constituent truth labels to carry through.
    jet_truth_names : Sequence[str], optional
        Optional jet-level truth targets to carry through.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        ``(jets_rows, pfcand_rows, n_truncated)``.
    """
    n_jets = len(arrays)

    jets = np.zeros(n_jets, dtype=_jets_dtype(jet_truth_names))
    jets["eventNumber"] = id_offset + np.arange(n_jets, dtype=np.int64)
    jets["flavour_label"] = _flavour_label(arrays)
    for name in ONEHOT_LABELS:
        jets[name] = ak.to_numpy(arrays[name]).astype(np.int32)
    for name in JET_SCALARS:
        jets[name] = ak.to_numpy(arrays[name]).astype(np.float32)
    for name in jet_truth_names:
        # stage2 stores everything as float; cast back per-field (PDG -> int).
        field = _h5_label(name)
        values = ak.to_numpy(arrays[name])
        if jets.dtype[field].kind == "i":
            # a NaN (unmatched truth jet, light-flavour hadron pT) would cast to
            # INT_MIN and blow up the salt label_map lookups, so send it to the
            # branch's own "nothing here" value instead.
            fill = 0 if field.endswith("PdgId") else LABEL_FILL
            values = np.where(np.isfinite(values), values, fill)
        jets[field] = values.astype(jets.dtype[field])

    theta = jets["jet_theta"].astype(np.float64)
    jet_eta = _eta(theta)
    jets["jet_pt"] = (jets["jet_p"].astype(np.float64) * np.sin(theta)).astype(np.float32)
    jets["jet_eta"] = jet_eta.astype(np.float32)

    counts = ak.to_numpy(ak.num(arrays[PFCAND_FEATURES[0]])).astype(np.int64)
    jets["n_pfcand"] = np.minimum(counts, max_pfcand).astype(np.int32)
    n_truncated = int(np.sum(np.maximum(counts - max_pfcand, 0)))

    pfcand = np.zeros((n_jets, max_pfcand), dtype=_pfcand_dtype(label_names))
    col = np.arange(max_pfcand)[None, :]
    pfcand["valid"] = col < jets["n_pfcand"][:, None]

    for name in PFCAND_FEATURES:
        block = _padded(arrays, name, max_pfcand, 0.0).astype(np.float32)
        if name in IP_SENTINEL_FEATURES:
            block = np.where(np.abs(block) > IP_SENTINEL_ABOVE, IP_FILL, block)
        pfcand[name] = block

    # Kinematics for the edge features (PFCAND_KINEMATICS); the angles are
    # jet-relative so that they do not depend on where the jet sits in the
    # detector. Padding slots are zeroed explicitly: a padded theta of 0 would
    # otherwise come out as eta = 13.8.
    valid = pfcand["valid"]
    eta = _eta(_padded(arrays, "pfcand_theta", max_pfcand, 0.0).astype(np.float64))
    phi = _padded(arrays, "pfcand_phi", max_pfcand, 0.0).astype(np.float64)
    dphi = (phi - jets["jet_phi"].astype(np.float64)[:, None] + np.pi) % (2 * np.pi) - np.pi
    pfcand["pfcand_e"] = _padded(arrays, "pfcand_e", max_pfcand, 0.0).astype(np.float32)
    for name, block in (("pfcand_deta", eta - jet_eta[:, None]), ("pfcand_dphi", dphi)):
        pfcand[name] = np.where(valid, block, 0.0).astype(np.float32)

    for name in label_names:
        # int truth labels: pad missing slots with LABEL_FILL (not 0, which is a
        # valid label value). The valid mask still marks the real constituents.
        pfcand[_h5_label(name)] = _padded(arrays, name, max_pfcand, LABEL_FILL).astype(np.int32)

    return jets, pfcand, n_truncated


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------


def convert_fcc_higgs_decay_to_h5(
    inputs: list[str],
    output: str | Path,
    *,
    tree: str = DEFAULT_TREE,
    max_pfcand: int = DEFAULT_MAX_PFCAND,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    max_jets: int | None = None,
    quiet: bool = False,
    compression: str = "gzip",
    compression_opts: int = 7,
) -> dict:
    """Stream higgs_decay ntuples into a Salt-native ``jets`` + ``pfcand`` H5.

    Parameters
    ----------
    inputs : list[str]
        File paths (local, xrdcp'd, or ``root://`` if ``fsspec-xrootd`` is
        installed).
    output : str | Path
        Destination H5 path.
    tree : str, optional
        Tree name inside the inputs.
    max_pfcand : int, optional
        Constituent slots per jet.
    chunk_size : int, optional
        Jets read per uproot step.
    max_jets : int | None, optional
        Stop after this many jets.
    quiet : bool, optional
        Suppress per-chunk progress output.
    compression : str, optional
        h5py dataset compression filter.
    compression_opts : int, optional
        h5py dataset compression level.

    Returns
    -------
    dict
        Summary of the conversion (counts and output path).

    Raises
    ------
    ValueError
        If the inputs lack the constituent-kinematics branches the pair
        features are derived from.
    """
    read_labels = _detect_pfcand_labels(inputs, tree)
    # On legacy ntuples the type label is DERIVED from the matched PDG rather
    # than read from the tree, under the same H5 field name either way.
    write_labels = read_labels
    derive_type = PFCAND_TYPE_LABEL not in read_labels and "pfcand_truthPID" in read_labels
    if derive_type:
        write_labels = (*read_labels, PFCAND_TYPE_LABEL)
    jet_truth = _detect_jet_truth(inputs, tree)
    present = _detect_present(inputs, tree, PFCAND_KINEMATIC_BRANCHES)
    if missing := sorted(set(PFCAND_KINEMATIC_BRANCHES) - set(present)):
        raise ValueError(
            f"{inputs[0]} lacks {missing}: the constituent kinematics "
            f"{PFCAND_KINEMATICS} need the stage2 of FCCAnalyses#524 or later."
        )
    branches = (
        list(PFCAND_FEATURES)
        + list(PFCAND_KINEMATIC_BRANCHES)
        + list(JET_SCALARS)
        + list(ONEHOT_LABELS)
        + list(read_labels)
        + list(jet_truth)
    )
    sources = dict.fromkeys(inputs, tree)

    datasets = (DatasetSpec("jets", None), DatasetSpec("pfcand", max_pfcand))
    dtype_overrides = {"jets": _jets_dtype(jet_truth), "pfcand": _pfcand_dtype(write_labels)}

    ghost_labelled = has_ghost_labels(jet_truth)
    n_written = 0
    n_truncated = 0
    label_counts = np.zeros(len(FLAVOUR_NAMES), dtype=np.int64)
    n_bad_label = 0
    n_relabelled = 0

    def _log(msg: str) -> None:
        if not quiet:
            print(msg, file=sys.stderr)

    with SaltH5Writer(
        output,
        reference="<fcc-higgs_decay-native>",
        datasets=datasets,
        dtype_overrides=dtype_overrides,
        compression=compression,
        compression_opts=compression_opts,
    ) as writer:
        for arrays in uproot.iterate(
            sources, expressions=branches, step_size=chunk_size, library="ak"
        ):
            if max_jets is not None and n_written >= max_jets:
                break
            chunk = arrays
            if max_jets is not None and n_written + len(chunk) > max_jets:
                chunk = chunk[: max_jets - n_written]

            if derive_type:
                chunk[PFCAND_TYPE_LABEL] = _pfcand_type_from_pid(chunk["pfcand_truthPID"])
            jets, pfcand, trunc = build_chunk_rows(
                chunk,
                max_pfcand=max_pfcand,
                id_offset=n_written,
                label_names=write_labels,
                jet_truth_names=jet_truth,
            )
            writer.append("jets", jets)
            writer.append("pfcand", pfcand)

            n_written += len(jets)
            n_truncated += trunc
            n_bad_label += int(np.sum(jets["flavour_label"] < 0))
            if ghost_labelled:
                # How wrong the file-name labelling was, for the record.
                onehot = _flavour_label_from_onehot(chunk)
                n_relabelled += int(
                    np.sum((jets["flavour_label"] >= 0) & (jets["flavour_label"] != onehot))
                )
            good = jets["flavour_label"] >= 0
            label_counts += np.bincount(jets["flavour_label"][good], minlength=len(FLAVOUR_NAMES))
            _log(f"  wrote {n_written:,} jets")

    flavour_counts = {FLAVOUR_NAMES[i]: int(label_counts[i]) for i in range(len(FLAVOUR_NAMES))}
    summary = {
        "output": str(output),
        "n_jets": n_written,
        "n_truncated_constituents": n_truncated,
        "n_bad_label": n_bad_label,
        "flavour_label_source": "ghost" if ghost_labelled else "onehot",
        "n_relabelled_vs_onehot": n_relabelled if ghost_labelled else None,
        "max_pfcand": max_pfcand,
        "pfcand_labels": [_h5_label(name) for name in write_labels],
        "jet_truth_scalars": [_h5_label(name) for name in jet_truth],
        "flavour_counts": flavour_counts,
    }
    if not quiet:
        _log("")
        _log(f"Done: {n_written:,} jets -> {output}")
        _log(f"  pfcand truth labels: {summary['pfcand_labels'] or 'none in input'}")
        _log(f"  jet truth scalars:   {summary['jet_truth_scalars'] or 'none in input'}")
        _log(f"  flavour_label from:  {summary['flavour_label_source']}")
        for name, count in flavour_counts.items():
            _log(f"  {name:9s} {count:,}")
        if n_truncated:
            _log(f"  truncated constituents (> {max_pfcand}): {n_truncated:,}")
        if ghost_labelled:
            frac = 100 * n_relabelled / n_written if n_written else 0.0
            _log(f"  disagrees with the file-name label: {n_relabelled:,} ({frac:.2f}%)")
            _log(f"  no flavour resolved (excluded):     {n_bad_label:,}")
        elif n_bad_label:
            _log(f"  WARNING malformed one-hot truth rows: {n_bad_label:,}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Build the ``fcc-ftag-higgs-decay`` argument parser.

    Returns
    -------
    argparse.ArgumentParser
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        description="Convert FCC-ee higgs_decay flat ntuples to Salt-native H5.",
    )
    parser.add_argument("inputs", nargs="+", help="higgs_decay ROOT files (or globs).")
    parser.add_argument("-o", "--output", required=True, help="Output H5 path.")
    parser.add_argument("--tree", default=DEFAULT_TREE, help="Tree name.")
    parser.add_argument(
        "--max-pfcand",
        type=int,
        default=DEFAULT_MAX_PFCAND,
        help="Fixed constituent dimension (jagged arrays padded/clipped to this).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Jets per read/write chunk.",
    )
    parser.add_argument(
        "--max-jets",
        type=int,
        default=None,
        help="Stop after this many jets.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``fcc-ftag-higgs-decay``.

    Parameters
    ----------
    argv : list[str] | None, optional
        Command line, defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Process exit status.
    """
    args = build_parser().parse_args(argv)
    convert_fcc_higgs_decay_to_h5(
        args.inputs,
        args.output,
        tree=args.tree,
        max_pfcand=args.max_pfcand,
        chunk_size=args.chunk_size,
        max_jets=args.max_jets,
        quiet=args.quiet,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
