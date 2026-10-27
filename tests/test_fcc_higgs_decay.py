import awkward as ak
import numpy as np

from fcc_ftag_converter.fcc_higgs_decay import (
    JET_SCALARS,
    LABEL_FILL,
    ONEHOT_LABELS,
    PFCAND_FEATURES,
    PFCAND_KINEMATIC_BRANCHES,
    build_chunk_rows,
)

NEW_LABELS = ("pfcand_truthOriginLabel", "pfcand_truthVertexIndex")
LEGACY_LABELS = ("pfcand_truthOrigin", "pfcand_truthVertex")


def _toy_arrays() -> ak.Array:
    # 3 jets: a b-jet (2 constituents), a tau-jet (1), a malformed-truth jet (0).
    onehot = {name: [0, 0, 0] for name in ONEHOT_LABELS}
    onehot["recojet_isB"] = [1, 0, 0]
    onehot["recojet_isTAU"] = [0, 1, 0]  # jet 2 stays all-zero -> bad label

    scalars = {name: [1.0, 2.0, 3.0] for name in JET_SCALARS}
    scalars["jet_p"] = [10.0, 20.0, 30.0]
    scalars["jet_theta"] = [np.pi / 2, np.pi / 4, np.pi / 2]
    scalars["jet_phi"] = [3.0, 0.0, 0.0]

    feats = {name: [[], [], []] for name in (*PFCAND_FEATURES, *PFCAND_KINEMATIC_BRANCHES)}
    feats["pfcand_erel_log"] = [[-1.0, -2.0], [0.5], []]
    feats["pfcand_btagSip2dSig"] = [[3.0, -9.0], [7.0], []]
    feats["pfcand_e"] = [[4.0, 2.0], [5.0], []]
    feats["pfcand_theta"] = [[np.pi / 2, np.pi / 4], [np.pi / 2], []]
    feats["pfcand_phi"] = [[3.0, -3.0], [0.5], []]

    return ak.Array({**onehot, **scalars, **feats})


def _with_ghost(hadron: list[float], parton: list[float]) -> ak.Array:
    base = _toy_arrays()
    fields = {name: base[name] for name in base.fields}
    fields["jet_HadronGhostTruthLabelID"] = hadron
    fields["jet_PartonTruthLabelID"] = parton
    return ak.Array(fields)


GHOST_NAMES = ("jet_HadronGhostTruthLabelID", "jet_PartonTruthLabelID")


def test_flavour_label_from_ghost_hadron_label() -> None:
    # b, c and tau are settled by the hadron label alone, whatever the parton.
    arrays = _with_ghost([5.0, 4.0, 15.0], [21.0, 21.0, -1.0])
    jets, _, _ = build_chunk_rows(arrays, max_pfcand=4, jet_truth_names=GHOST_NAMES)
    assert jets["flavour_label"].tolist() == [4, 3, 6]


def test_flavour_label_from_parton_for_light_jets() -> None:
    # The hadron label calls all of u/d/s/g light; the parton label splits them.
    # PDG order: 1 is d, 2 is u.
    for parton, expected in ((2.0, 0), (1.0, 1), (3.0, 2), (21.0, 5)):
        arrays = _with_ghost([0.0, 0.0, 0.0], [parton, parton, parton])
        jets, _, _ = build_chunk_rows(arrays, max_pfcand=4, jet_truth_names=GHOST_NAMES)
        assert jets["flavour_label"].tolist() == [expected] * 3


def test_flavour_label_excludes_unresolvable_jets() -> None:
    # Light hadron label with no light parton, and a NaN, both fall through.
    arrays = _with_ghost([0.0, 0.0, np.nan], [-1.0, 5.0, 21.0])
    jets, _, _ = build_chunk_rows(arrays, max_pfcand=4, jet_truth_names=GHOST_NAMES)
    assert jets["flavour_label"].tolist() == [-1, -1, -1]


def test_ghost_label_overrides_the_file_name_one_hot() -> None:
    # Every jet in the toy carries recojet_isB / recojet_isTAU from the file
    # name. The ghost labels say otherwise and must win.
    arrays = _with_ghost([0.0, 0.0, 5.0], [21.0, 2.0, 5.0])
    jets, _, _ = build_chunk_rows(arrays, max_pfcand=4, jet_truth_names=GHOST_NAMES)
    assert jets["flavour_label"].tolist() == [5, 0, 4]
    # the one-hot is still written, so the disagreement stays measurable.
    assert jets["recojet_isB"].tolist() == [1, 0, 0]


def test_flavour_label_falls_back_to_the_one_hot() -> None:
    # Without the ghost branches the old file-name labelling is all there is.
    arrays = _toy_arrays()
    jets, _, _ = build_chunk_rows(arrays, max_pfcand=4, jet_truth_names=())
    assert jets["flavour_label"].tolist() == [4, 6, -1]


def test_flavour_label_and_truncation() -> None:
    arrays = _toy_arrays()
    jets, _pfcand, n_trunc = build_chunk_rows(arrays, max_pfcand=4, id_offset=100)

    # label index follows ONEHOT_LABELS order: B==4, TAU==6, malformed==-1.
    assert jets["flavour_label"].tolist() == [4, 6, -1]
    assert jets["eventNumber"].tolist() == [100, 101, 102]
    assert jets["n_pfcand"].tolist() == [2, 1, 0]
    assert n_trunc == 0


def test_pfcand_padding_and_valid_mask() -> None:
    arrays = _toy_arrays()
    _, pfcand, _ = build_chunk_rows(arrays, max_pfcand=4, id_offset=0)

    assert pfcand.shape == (3, 4)
    assert pfcand["valid"][0].tolist() == [True, True, False, False]
    assert pfcand["valid"][2].tolist() == [False, False, False, False]
    # real values preserved, padding zeroed.
    np.testing.assert_allclose(pfcand["pfcand_erel_log"][0], [-1.0, -2.0, 0.0, 0.0])
    np.testing.assert_allclose(pfcand["pfcand_btagSip2dSig"][1], [7.0, 0.0, 0.0, 0.0])


def test_jet_pt_eta_from_p_theta() -> None:
    arrays = _toy_arrays()
    jets, _, _ = build_chunk_rows(arrays, max_pfcand=4)
    # theta=pi/2 -> pt=p, eta=0
    assert np.isclose(jets["jet_pt"][0], 10.0)
    assert np.isclose(jets["jet_eta"][0], 0.0, atol=1e-5)


def test_pfcand_e_deta_dphi_from_e_theta_phi() -> None:
    _, pfcand, _ = build_chunk_rows(_toy_arrays(), max_pfcand=4)
    # Jet 0 sits at theta=pi/2 (eta 0), phi=3.0. Its first constituent is
    # collinear: deta=dphi=0. The second is at theta=pi/4, phi=-3.0, which is
    # 2pi-6 rad away once wrapped, not -6. The energy is copied as is.
    expected_eta = -np.log(np.tan(np.pi / 8))
    np.testing.assert_allclose(pfcand["pfcand_e"][0], [4.0, 2.0, 0, 0])
    np.testing.assert_allclose(pfcand["pfcand_deta"][0], [0.0, expected_eta, 0, 0], atol=1e-6)
    np.testing.assert_allclose(pfcand["pfcand_dphi"][0], [0.0, 2 * np.pi - 6.0, 0, 0], atol=1e-6)
    # Jet 1 is the forward one (theta=pi/4): a constituent at theta=pi/2 is
    # BELOW it in eta, so deta carries the sign.
    np.testing.assert_allclose(pfcand["pfcand_deta"][1][0], -expected_eta, atol=1e-6)
    np.testing.assert_allclose(pfcand["pfcand_dphi"][1][0], 0.5, atol=1e-6)
    # Padding is zero, not the eta of theta=0.
    assert pfcand["pfcand_deta"][2].tolist() == [0.0] * 4
    # No momentum magnitude is written: together with the energy it would be
    # the Delphes truth mass, i.e. hadron ID (see PFCAND_KINEMATICS).
    assert not {"pfcand_pt", "pfcand_p"} & set(pfcand.dtype.names)


def test_truncation_counts_dropped_constituents() -> None:
    arrays = _toy_arrays()
    jets, _, n_trunc = build_chunk_rows(arrays, max_pfcand=1)
    assert jets["n_pfcand"].tolist() == [1, 1, 0]
    assert n_trunc == 1  # jet 0 had 2, kept 1


def _with_labels(names: tuple[str, ...]) -> ak.Array:
    base = _toy_arrays()
    fields = {name: base[name] for name in base.fields}
    fields[names[0]] = [[3, 4], [6], []]
    fields[names[1]] = [[0, 1], [0], []]
    return ak.Array(fields)


def test_optional_truth_labels_carried_and_padded() -> None:
    arrays = _with_labels(NEW_LABELS)
    _, pfcand, _ = build_chunk_rows(arrays, max_pfcand=4, label_names=NEW_LABELS)

    # int dtype, real values preserved, padding slots filled with LABEL_FILL.
    assert pfcand["pfcand_truthOriginLabel"].dtype == np.int32
    assert pfcand["pfcand_truthOriginLabel"][0].tolist() == [3, 4, LABEL_FILL, LABEL_FILL]
    assert pfcand["pfcand_truthVertexIndex"][1].tolist() == [0, LABEL_FILL, LABEL_FILL, LABEL_FILL]
    assert pfcand["pfcand_truthOriginLabel"][2].tolist() == [LABEL_FILL] * 4


def test_legacy_label_names_written_under_the_new_ones() -> None:
    # Pre-ghost ntuples spell these differently; the H5 field name must not.
    arrays = _with_labels(LEGACY_LABELS)
    _, pfcand, _ = build_chunk_rows(arrays, max_pfcand=4, label_names=LEGACY_LABELS)

    assert "pfcand_truthOrigin" not in pfcand.dtype.names
    assert pfcand["pfcand_truthOriginLabel"][0].tolist() == [3, 4, LABEL_FILL, LABEL_FILL]
    assert pfcand["pfcand_truthVertexIndex"][0].tolist() == [0, 1, LABEL_FILL, LABEL_FILL]


def test_labels_absent_by_default() -> None:
    # Without label_names the pfcand block has no label fields (back-compat).
    arrays = _toy_arrays()
    _, pfcand, _ = build_chunk_rows(arrays, max_pfcand=4)
    assert "pfcand_truthOriginLabel" not in pfcand.dtype.names


def test_jet_truth_int_and_float_fields() -> None:
    # stage2 hands every branch over as float; the label/PDG/count branches have
    # to come back as ints for salt's label_map lookups, the rest stay float.
    base = _toy_arrays()
    fields = {name: base[name] for name in base.fields}
    fields["jet_HadronGhostTruthLabelID"] = [5.0, 15.0, 0.0]
    fields["jet_HadronGhostInitialTruthLabelPdgId"] = [521.0, 0.0, 0.0]
    fields["jet_GhostBHadronsFinalCount"] = [1.0, 0.0, 0.0]
    fields["jet_truthJetPt"] = [42.5, 17.25, 3.0]
    fields["jet_truthJetPurity"] = [0.95, 0.81, 0.4]
    names = (
        "jet_HadronGhostTruthLabelID",
        "jet_HadronGhostInitialTruthLabelPdgId",
        "jet_GhostBHadronsFinalCount",
        "jet_truthJetPt",
        "jet_truthJetPurity",
    )
    jets, _, _ = build_chunk_rows(ak.Array(fields), max_pfcand=4, jet_truth_names=names)

    assert jets["jet_HadronGhostTruthLabelID"].dtype == np.int32
    assert jets["jet_HadronGhostInitialTruthLabelPdgId"].dtype == np.int32
    assert jets["jet_GhostBHadronsFinalCount"].dtype == np.int32
    assert jets["jet_truthJetPt"].dtype == np.float32
    assert jets["jet_HadronGhostTruthLabelID"].tolist() == [5, 15, 0]
    assert jets["jet_HadronGhostInitialTruthLabelPdgId"].tolist() == [521, 0, 0]
    np.testing.assert_allclose(jets["jet_truthJetPurity"], [0.95, 0.81, 0.4], rtol=1e-6)


def test_legacy_jet_truth_names_written_under_the_new_ones() -> None:
    base = _toy_arrays()
    fields = {name: base[name] for name in base.fields}
    fields["jet_hadronPdgId"] = [521.0, 0.0, -411.0]
    fields["jet_truthPt"] = [42.5, 17.25, 3.0]
    names = ("jet_hadronPdgId", "jet_truthPt")
    jets, _, _ = build_chunk_rows(ak.Array(fields), max_pfcand=4, jet_truth_names=names)

    assert "jet_hadronPdgId" not in jets.dtype.names
    assert jets["jet_HadronGhostInitialTruthLabelPdgId"].tolist() == [521, 0, -411]
    np.testing.assert_allclose(jets["jet_truthJetPt"], [42.5, 17.25, 3.0], rtol=1e-6)


def test_nan_in_an_int_truth_branch_does_not_become_int_min() -> None:
    # An unmatched truth jet arrives as NaN. Casting that straight to int32
    # gives INT_MIN, which then trips salt's label_map lookups.
    base = _toy_arrays()
    fields = {name: base[name] for name in base.fields}
    fields["jet_PartonTruthLabelID"] = [21.0, np.nan, 5.0]
    fields["jet_HadronGhostTruthLabelPdgId"] = [np.nan, 0.0, 511.0]
    names = ("jet_PartonTruthLabelID", "jet_HadronGhostTruthLabelPdgId")
    jets, _, _ = build_chunk_rows(ak.Array(fields), max_pfcand=4, jet_truth_names=names)

    # -1 is "no parton", 0 is "no hadron": each branch's own empty value.
    assert jets["jet_PartonTruthLabelID"].tolist() == [21, -1, 5]
    assert jets["jet_HadronGhostTruthLabelPdgId"].tolist() == [0, 0, 511]
