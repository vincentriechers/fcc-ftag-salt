import json
import tempfile
from pathlib import Path

import numpy as np

from fcc_ftag_converter.compare import _ks, _parse_variables
from fcc_ftag_converter.physics import (
    _charged_pt_cone_over_pt,
    _lepton_track_info,
    _nearest_jet_lepton_match,
    _q_over_p_from_omega,
    _signed_lepton_id,
    _track_uncertainties,
)
from fcc_ftag_converter.report import ReportAccumulator, _Reservoir, write_conversion_report


def test_q_over_p_is_written_in_inverse_mev() -> None:
    assert _q_over_p_from_omega(0.0, 0.0, 2.0) == 0.0  # noqa: RUF069  exact guard-path zero
    assert np.isclose(_q_over_p_from_omega(1.0e-4, 0.0, 2.0), 1.6678204759907602e-4)


def test_track_q_over_p_uncertainty_is_inverse_mev() -> None:
    covariance = np.zeros(21, dtype=np.float32)
    covariance[5] = 1.0e-8
    uncertainties = _track_uncertainties(
        covariance,
        z0=0.0,
        omega=1.0e-4,
        tan_lambda=0.0,
        magnetic_field_t=2.0,
    )
    assert np.isclose(uncertainties["qOverPUncertainty"], 1.6678204759907602e-4)


def test_lepton_and_isolation_helpers() -> None:
    assert _signed_lepton_id(11, -1.0) == 11
    assert _signed_lepton_id(13, 1.0) == -13

    pt = np.asarray([10.0, 2.0, 3.0, 4.0], dtype=np.float32)
    eta = np.asarray([0.0, 0.1, 0.4, 0.1], dtype=np.float32)
    phi = np.asarray([0.0, 0.1, 0.0, 0.1], dtype=np.float32)
    charge = np.asarray([-1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    assert np.isclose(_charged_pt_cone_over_pt(0, pt, eta, phi, charge), 0.2)


def test_track_type_label_atlas_sign_convention() -> None:
    """ATLAS ftagTruthTypeLabel: sign of label == sign of charge."""
    from fcc_ftag_converter.truth import _track_type_label_from_mc

    pdg = np.asarray(
        [211, -211, 321, -321, 11, -11, 13, -13, 3122, -3122, 22, 999, 0], dtype=np.int64
    )
    # π+ → +2 (charge +1)
    assert _track_type_label_from_mc([0], pdg) == 2
    # π- → -2 (charge -1)
    assert _track_type_label_from_mc([1], pdg) == -2
    # K+ → +3
    assert _track_type_label_from_mc([2], pdg) == 3
    # K- → -3
    assert _track_type_label_from_mc([3], pdg) == -3
    # e- (PDG +11) → -5 (charge -1)
    assert _track_type_label_from_mc([4], pdg) == -5
    # e+ (PDG -11) → +5
    assert _track_type_label_from_mc([5], pdg) == 5
    # μ- → -6
    assert _track_type_label_from_mc([6], pdg) == -6
    # μ+ → +6
    assert _track_type_label_from_mc([7], pdg) == 6
    # Λ (neutral) → 4
    assert _track_type_label_from_mc([8], pdg) == 4
    assert _track_type_label_from_mc([9], pdg) == 4
    # γ → 7
    assert _track_type_label_from_mc([10], pdg) == 7
    # other → 1, no MC → 0
    assert _track_type_label_from_mc([11], pdg) == 1
    assert _track_type_label_from_mc([], pdg) == 0


def test_cone_label_pt_cut_rejects_soft_hadrons() -> None:
    """Heavy hadrons below the pT cut should not contribute to jet labels."""
    from fcc_ftag_converter.truth import _cone_exclusive_truth_labels

    jet_eta = np.asarray([0.0], dtype=np.float32)
    jet_phi = np.asarray([0.0], dtype=np.float32)
    # Soft b-hadron (pT = 2 GeV) near the jet — should NOT label as b.
    pdg = np.asarray([511], dtype=np.int64)
    mc_px = np.asarray([2.0], dtype=np.float32)  # pT = 2 GeV
    mc_py = np.asarray([0.0], dtype=np.float32)
    mc_pz = np.asarray([0.0], dtype=np.float32)
    empty = np.asarray([0], dtype=np.int64)
    labels, _ext, _pdg, _midx = _cone_exclusive_truth_labels(
        jet_eta,
        jet_phi,
        pdg_ids=pdg,
        mc_px=mc_px,
        mc_py=mc_py,
        mc_pz=mc_pz,
        parent_begin=empty,
        parent_end=empty,
        parent_index=empty,
        children_map={},
        radius=0.3,
        hadron_pt_min_gev=5.0,
    )
    assert int(labels[0]) == 0

    # Hard b-hadron (pT = 30 GeV) should label.
    mc_px[0] = 30.0
    labels, _ext, _pdg, _midx = _cone_exclusive_truth_labels(
        jet_eta,
        jet_phi,
        pdg_ids=pdg,
        mc_px=mc_px,
        mc_py=mc_py,
        mc_pz=mc_pz,
        parent_begin=empty,
        parent_end=empty,
        parent_index=empty,
        children_map={},
        radius=0.3,
        hadron_pt_min_gev=5.0,
    )
    assert int(labels[0]) == 5


def test_origin_label_basic_classes() -> None:
    """ATLAS ftagTruthOriginLabel = 2/3/4/5/6/7 depending on ancestry."""
    from fcc_ftag_converter.truth import _origin_label_from_ancestors

    # Ancestry tree:
    #   0: u-quark (primary)
    #   1: B0 (PDG 511, b-hadron) — parent: 0
    #   2: D0 (PDG 421, c-hadron) — parent: 1 (c from b)
    #   3: π+ from D0 — parent: 2 (FromBC)
    #   4: e+ from B0 directly — parent: 1 (FromB)
    #   5: standalone D+ (PDG 411) — parent: 0
    #   6: π from D+ — parent: 5 (FromC)
    #   7: tau (PDG 15) — parent: 0
    #   8: π from tau — parent: 7 (FromTau)
    #   9: primary pion — parent: 0 (Primary)
    pdg = np.asarray([2, 511, 421, 211, -11, 411, 211, 15, 211, 211], dtype=np.int64)
    parent_begin = np.asarray([0, 0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    parent_end = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
    parent_index = np.asarray([0, 1, 2, 1, 0, 5, 0, 7, 0], dtype=np.int64)
    cache: dict[int, int] = {}

    assert (
        _origin_label_from_ancestors(3, pdg, parent_begin, parent_end, parent_index, cache) == 4
    )  # FromBC
    assert (
        _origin_label_from_ancestors(4, pdg, parent_begin, parent_end, parent_index, cache) == 3
    )  # FromB
    assert (
        _origin_label_from_ancestors(6, pdg, parent_begin, parent_end, parent_index, cache) == 5
    )  # FromC
    assert (
        _origin_label_from_ancestors(8, pdg, parent_begin, parent_end, parent_index, cache) == 6
    )  # FromTau
    assert (
        _origin_label_from_ancestors(9, pdg, parent_begin, parent_end, parent_index, cache) == 2
    )  # Primary
    assert _origin_label_from_ancestors(-1, pdg, parent_begin, parent_end, parent_index, cache) == 7


def test_cone_extended_labels_double_b_and_tau() -> None:
    """HadronConeExclExtendedTruthLabelID encodes counts: 55, 54, 44, 1515."""
    from fcc_ftag_converter.truth import _cone_exclusive_truth_labels

    # All test particles at η=φ=0 to keep cone geometry trivial.
    jet_eta = np.asarray([0.0], dtype=np.float32)
    jet_phi = np.asarray([0.0], dtype=np.float32)

    # Case A: two b-hadrons inside the cone → label 5, extended 55.
    pdg = np.asarray([511, 521], dtype=np.int64)
    mc_px = np.asarray([30.0, 30.0], dtype=np.float32)
    mc_py = np.asarray([0.0, 0.0], dtype=np.float32)
    mc_pz = np.asarray([0.0, 0.5], dtype=np.float32)  # tiny dη, well inside cone
    empty = np.zeros(2, dtype=np.int64)
    labels, ext, _pdg, _midx = _cone_exclusive_truth_labels(
        jet_eta,
        jet_phi,
        pdg_ids=pdg,
        mc_px=mc_px,
        mc_py=mc_py,
        mc_pz=mc_pz,
        parent_begin=empty,
        parent_end=empty,
        parent_index=empty,
        children_map={},
        radius=0.3,
        hadron_pt_min_gev=5.0,
    )
    assert int(labels[0]) == 5
    assert int(ext[0]) == 55

    # Case B: b + c → label 5, extended 54.
    pdg = np.asarray([511, 411], dtype=np.int64)
    labels, ext, _pdg, _midx = _cone_exclusive_truth_labels(
        jet_eta,
        jet_phi,
        pdg_ids=pdg,
        mc_px=mc_px,
        mc_py=mc_py,
        mc_pz=mc_pz,
        parent_begin=empty,
        parent_end=empty,
        parent_index=empty,
        children_map={},
        radius=0.3,
        hadron_pt_min_gev=5.0,
    )
    assert int(labels[0]) == 5
    assert int(ext[0]) == 54

    # Case C: two c-hadrons → label 4, extended 44.
    pdg = np.asarray([411, 421], dtype=np.int64)
    labels, ext, _pdg, _midx = _cone_exclusive_truth_labels(
        jet_eta,
        jet_phi,
        pdg_ids=pdg,
        mc_px=mc_px,
        mc_py=mc_py,
        mc_pz=mc_pz,
        parent_begin=empty,
        parent_end=empty,
        parent_index=empty,
        children_map={},
        radius=0.3,
        hadron_pt_min_gev=5.0,
    )
    assert int(labels[0]) == 4
    assert int(ext[0]) == 44

    # Case D: tau-lepton → label 15.
    pdg = np.asarray([15], dtype=np.int64)
    mc_px = np.asarray([30.0], dtype=np.float32)
    mc_py = np.asarray([0.0], dtype=np.float32)
    mc_pz = np.asarray([0.0], dtype=np.float32)
    empty = np.zeros(1, dtype=np.int64)
    labels, ext, _pdg, _midx = _cone_exclusive_truth_labels(
        jet_eta,
        jet_phi,
        pdg_ids=pdg,
        mc_px=mc_px,
        mc_py=mc_py,
        mc_pz=mc_pz,
        parent_begin=empty,
        parent_end=empty,
        parent_index=empty,
        children_map={},
        radius=0.3,
        hadron_pt_min_gev=5.0,
    )
    assert int(labels[0]) == 15
    assert int(ext[0]) == 15


def test_lepton_decay_label_encoding() -> None:
    """LeptonDecayLabel: digits 1/2/3 per lepton flavour in the decay chain."""
    from fcc_ftag_converter.truth import _build_children_map, _lepton_decay_label_for_hadron

    # MC tree:
    #   0 = B0 (PDG 511)
    #   1 = e- (parent: B0)         → label 1
    #   2 = D+ (parent: B0)         → c daughter
    #   3 = mu+ (parent: D+)        → label 2
    pdg = np.asarray([511, 11, 411, -13], dtype=np.int64)
    parent_begin = np.asarray([0, 0, 1, 2], dtype=np.int64)
    parent_end = np.asarray([0, 1, 2, 3], dtype=np.int64)
    parent_index = np.asarray([0, 0, 2], dtype=np.int64)
    cm = _build_children_map(parent_begin, parent_end, parent_index, len(pdg))

    # B0 decay chain has both e and μ → label 12
    assert _lepton_decay_label_for_hadron(0, pdg, cm) == 12
    # D+ alone (just μ) → label 2
    assert _lepton_decay_label_for_hadron(2, pdg, cm) == 2
    # Bare electron has no descendants → 0
    assert _lepton_decay_label_for_hadron(1, pdg, cm) == 0
    # Out-of-range → 0
    assert _lepton_decay_label_for_hadron(-1, pdg, cm) == 0


def test_track_vertex_index_clusters_and_pv() -> None:
    """PV = largest cluster → index 0; SVs distinct, merged within 0.1 mm."""
    from fcc_ftag_converter.truth import _assign_track_vertex_indices

    # 4 MC particles. 0,1 at IR (~0,0,0); 2 at SV (5,0,0); 3 at another SV (10,0,0).
    mc_vx = np.asarray([0.0, 0.02, 5.0, 10.0], dtype=np.float32)
    mc_vy = np.asarray([0.0, 0.00, 0.0, 0.0], dtype=np.float32)
    mc_vz = np.asarray([0.0, 0.00, 0.0, 0.0], dtype=np.float32)
    # 4 tracks linked to MCs 0,1,2,3 respectively
    track_mc = [(100, 0), (101, 1), (102, 2), (103, 3)]
    result = _assign_track_vertex_indices(track_mc, mc_vx, mc_vy, mc_vz, merge_radius_mm=0.1)
    # MC 0 and 1 are in the same PV cluster → index 0
    assert result[100] == 0
    assert result[101] == 0
    # MC 2 and 3 are in distinct SV clusters
    assert result[102] in {1, 2}
    assert result[103] in {1, 2}
    assert result[102] != result[103]

    # Track with no MC link → -2
    result2 = _assign_track_vertex_indices([(200, -1)], mc_vx, mc_vy, mc_vz)
    assert result2[200] == -2


def test_compare_helpers() -> None:
    assert _parse_variables(["jets.pt", "tracks_ghost.qOverP"]) == [
        ("jets", "pt"),
        ("tracks_ghost", "qOverP"),
    ]
    assert _ks(np.asarray([0.0, 1.0]), np.asarray([0.0, 1.0])) == 0.0  # noqa: RUF069  identical inputs


# --- nearest-jet matching ---


def test_nearest_jet_match_basic() -> None:
    reco_eta = np.asarray([0.1, 2.0], dtype=np.float32)
    reco_phi = np.asarray([0.0, 0.0], dtype=np.float32)
    jet_eta = np.asarray([0.0, 2.1], dtype=np.float32)
    jet_phi = np.asarray([0.0, 0.0], dtype=np.float32)
    result = _nearest_jet_lepton_match({0, 1}, reco_eta, reco_phi, jet_eta, jet_phi, max_dr=0.4)
    assert result[0] == 0
    assert result[1] == 1


def test_nearest_jet_match_outside_dr() -> None:
    reco_eta = np.asarray([1.0], dtype=np.float32)
    reco_phi = np.asarray([0.0], dtype=np.float32)
    jet_eta = np.asarray([0.0], dtype=np.float32)
    jet_phi = np.asarray([0.0], dtype=np.float32)
    result = _nearest_jet_lepton_match({0}, reco_eta, reco_phi, jet_eta, jet_phi, max_dr=0.4)
    assert 0 not in result


def test_nearest_jet_match_no_duplicate_per_jet() -> None:
    reco_eta = np.asarray([0.1, 0.2], dtype=np.float32)
    reco_phi = np.asarray([0.0, 0.0], dtype=np.float32)
    jet_eta = np.asarray([0.0, 5.0], dtype=np.float32)
    jet_phi = np.asarray([0.0, 0.0], dtype=np.float32)
    result = _nearest_jet_lepton_match({0, 1}, reco_eta, reco_phi, jet_eta, jet_phi, max_dr=0.4)
    assert len(result) == len(set(result.keys()))
    assert result.get(0) == 0
    assert result.get(1) == 0


def test_nearest_jet_match_empty_jets() -> None:
    reco_eta = np.asarray([0.1], dtype=np.float32)
    reco_phi = np.asarray([0.0], dtype=np.float32)
    result = _nearest_jet_lepton_match(
        {0},
        reco_eta,
        reco_phi,
        np.empty(0, dtype=np.float32),
        np.empty(0, dtype=np.float32),
        max_dr=0.4,
    )
    assert result == {}


def test_lepton_track_info_no_track() -> None:
    empty = np.empty(0, dtype=np.int64)
    empty_f = np.empty(0, dtype=np.float32)
    empty_i = np.empty(0, dtype=np.int32)
    empty_cov = np.empty((0, 21), dtype=np.float32)
    info = _lepton_track_info(
        0,
        np.asarray([0], dtype=np.int64),
        np.asarray([0], dtype=np.int64),
        empty,
        empty,
        empty,
        empty,
        empty_f,
        empty_f,
        empty_f,
        empty_f,
        empty_i,
        empty_cov,
        2.0,
    )
    assert info == {"qOverP": 0.0, "d0": 0.0, "d0_significance": 0.0}


# --- streaming reservoir ---


def test_reservoir_under_capacity_keeps_all() -> None:
    r = _Reservoir(k=100, seed=0)
    r.update(np.arange(1, 51, dtype=np.float64))
    assert r.n_seen == 50
    assert len(r.values) == 50
    assert np.isclose(r.median(), 25.5)


def test_reservoir_overflow_stays_capped() -> None:
    r = _Reservoir(k=1000, seed=42)
    rng = np.random.default_rng(0)
    for _ in range(20):
        r.update(rng.uniform(0, 100, 1000))
    assert r.n_seen == 20_000
    assert len(r.values) == 1000
    # Reservoir sample should approximate the underlying uniform-on-[0,100] median ≈ 50.
    assert 40 < r.median() < 60


def test_reservoir_filters_invalid() -> None:
    r = _Reservoir(k=10, seed=0)
    samples = np.array([0.0, np.nan, np.inf, 5.0, 7.0])
    r.update(samples)
    assert sorted(r.values) == [5.0, 7.0]


# --- report accumulator + writer ---


def _track_rows(n_jets: int, n_slots: int) -> np.ndarray:
    dtype = np.dtype([("qOverP", "f4"), ("valid", "?")])
    rows = np.zeros((n_jets, n_slots), dtype=dtype)
    rows["qOverP"] = np.random.default_rng(0).uniform(1e-5, 5e-4, (n_jets, n_slots))
    rows["valid"] = True
    return rows


def _pt_rows(n_jets: int, n_slots: int) -> np.ndarray:
    dtype = np.dtype([("pt", "f4"), ("valid", "?")])
    rows = np.zeros((n_jets, n_slots), dtype=dtype)
    rows["pt"] = np.random.default_rng(1).uniform(1000, 50000, (n_jets, n_slots))
    rows["valid"] = True
    return rows


def test_report_accumulator_two_chunks_match_single_chunk() -> None:
    """Streaming across two chunks must give same valid counts as one big update."""
    jet_dtype = np.dtype([("pt_btagJes", "f4"), ("eta_btagJes", "f4")])
    jet_rows_a = np.zeros(5, dtype=jet_dtype)
    jet_rows_a["pt_btagJes"] = np.linspace(10000, 50000, 5)
    jet_rows_b = np.zeros(5, dtype=jet_dtype)
    jet_rows_b["pt_btagJes"] = np.linspace(50000, 90000, 5)

    track_a, track_b = _track_rows(5, 50), _track_rows(5, 50)
    flow_a, flow_b = _pt_rows(5, 50), _pt_rows(5, 50)
    elec_a, elec_b = _pt_rows(5, 10), _pt_rows(5, 10)
    muon_a, muon_b = _pt_rows(5, 5), _pt_rows(5, 5)

    acc = ReportAccumulator(reservoir_size=10_000)
    acc.update(jet_rows_a, track_a, flow_a, elec_a, muon_a)
    acc.update(jet_rows_b, track_b, flow_b, elec_b, muon_b)
    summary = acc.finalize()

    assert summary["n_jets"] == 10
    assert summary["valid_counts"]["tracks_ghost"] == 10 * 50
    assert summary["valid_counts"]["flows"] == 10 * 50
    assert summary["valid_counts"]["electrons"] == 10 * 10
    assert summary["valid_counts"]["muons"] == 10 * 5
    assert "jets.pt_btagJes" in summary["stats_median_p95"]


def test_write_conversion_report_creates_files() -> None:
    jet_dtype = np.dtype([("pt_btagJes", "f4"), ("eta_btagJes", "f4")])
    jet_rows = np.zeros(8, dtype=jet_dtype)
    jet_rows["pt_btagJes"] = np.linspace(20000, 100000, 8)

    acc = ReportAccumulator(reservoir_size=1_000)
    acc.update(jet_rows, _track_rows(8, 50), _pt_rows(8, 50), _pt_rows(8, 10), _pt_rows(8, 5))

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.h5"
        output_path.touch()
        write_conversion_report(
            output_path,
            acc,
            inputs=["file_a.root"],
            max_events=100,
            missing_collections={"Muon#0"},
            lepton_match_mode="constituent",
        )

        md_path = Path(tmpdir) / "test.summary.md"
        json_path = Path(tmpdir) / "test.summary.json"
        assert md_path.exists()
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["n_jets"] == 8
        assert data["missing_collections"] == ["Muon#0"]
        assert data["valid_counts"]["tracks_ghost"] == 8 * 50
        assert "jets.pt_btagJes" in data["stats_median_p95"]
        assert "Muon#0" in md_path.read_text()
