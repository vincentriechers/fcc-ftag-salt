"""End-to-end test: synthetic EDM4hep chunk → Salt-shaped row buffers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

ak = pytest.importorskip("awkward")
h5py = pytest.importorskip("h5py")

from fcc_ftag_converter.cli import convert_fcc_edm4hep_to_h5  # noqa: E402
from fcc_ftag_converter.reader import build_rows_from_chunk  # noqa: E402
from fcc_ftag_converter.schema import _load_reference_schema, default_schema_path  # noqa: E402


def _synthetic_chunk(n_events: int = 2) -> ak.Array:
    """Build a minimal EDM4hep-shaped awkward chunk with required + optional cols.

    Per event:
      - 2 jets (one b-like via constituent, one light)
      - 4 reco particles, 2 of which are jet constituents
      - 1 track per first reco particle
      - 1 Electron#0 ref to reco index 0, 1 Muon#0 ref to reco index 2
      - Minimal MCRecoAssociations and Particle truth so labelling helpers
        don't blow up.

    Parameters
    ----------
    n_events : int, optional
        How many copies of the synthetic event to emit.

    Returns
    -------
    ak.Array
        A chunk of ``n_events`` identical synthetic events.
    """
    rng = np.random.default_rng(42)
    one_event = {
        # Jets
        "Jet/Jet.energy": [50.0, 30.0],
        "Jet/Jet.mass": [5.0, 4.0],
        "Jet/Jet.momentum.x": [40.0, 20.0],
        "Jet/Jet.momentum.y": [10.0, 5.0],
        "Jet/Jet.momentum.z": [20.0, 10.0],
        "Jet/Jet.particles_begin": [0, 2],
        "Jet/Jet.particles_end": [2, 4],
        # Jet#2 -> ReconstructedParticles links: 4 relations total, all colId=15
        "Jet#2/Jet#2.index": [0, 1, 2, 3],
        "Jet#2/Jet#2.collectionID": [15, 15, 15, 15],
        # ReconstructedParticles
        "ReconstructedParticles/ReconstructedParticles.momentum.x": [
            25.0,
            5.0,
            15.0,
            3.0,
        ],
        "ReconstructedParticles/ReconstructedParticles.momentum.y": [5.0, 2.0, 3.0, 1.0],
        "ReconstructedParticles/ReconstructedParticles.momentum.z": [10.0, 3.0, 8.0, 2.0],
        "ReconstructedParticles/ReconstructedParticles.energy": [30.0, 7.0, 17.0, 4.0],
        "ReconstructedParticles/ReconstructedParticles.charge": [-1.0, 1.0, -1.0, 0.0],
        "ReconstructedParticles/ReconstructedParticles.tracks_begin": [0, 1, 2, 3],
        "ReconstructedParticles/ReconstructedParticles.tracks_end": [1, 2, 3, 3],
        # ReconstructedParticles#1 -> EFlowTrack: 3 relations (last reco has no track)
        "ReconstructedParticles#1/ReconstructedParticles#1.index": [0, 1, 2],
        "ReconstructedParticles#1/ReconstructedParticles#1.collectionID": [6, 6, 6],
        # Electron#0 -> reco index 0; Muon#0 -> reco index 2
        "Electron#0/Electron#0.index": [0],
        "Electron#0/Electron#0.collectionID": [15],
        "Muon#0/Muon#0.index": [2],
        "Muon#0/Muon#0.collectionID": [15],
        # EFlowTrack -> EFlowTrack_1 trackState ranges
        "EFlowTrack/EFlowTrack.trackStates_begin": [0, 1, 2],
        "EFlowTrack/EFlowTrack.trackStates_end": [1, 2, 3],
        # EFlowTrack_1 TrackState rows (3, one per track)
        "EFlowTrack_1/EFlowTrack_1.D0": [0.01, 0.02, 0.03],
        "EFlowTrack_1/EFlowTrack_1.phi": [0.2, 0.3, 0.4],
        "EFlowTrack_1/EFlowTrack_1.Z0": [0.05, 0.06, 0.07],
        "EFlowTrack_1/EFlowTrack_1.tanLambda": [0.1, 0.2, 0.3],
        "EFlowTrack_1/EFlowTrack_1.omega": [1.0e-4, 1.5e-4, 2.0e-4],
        "EFlowTrack_1/EFlowTrack_1.location": [1, 1, 1],  # AtIP
        "EFlowTrack_1/EFlowTrack_1.covMatrix[21]": [
            [1e-6] * 21,
            [1e-6] * 21,
            [1e-6] * 21,
        ],
        # MCRecoAssociations
        "MCRecoAssociations#0/MCRecoAssociations#0.index": [0, 1, 2, 3],
        "MCRecoAssociations#0/MCRecoAssociations#0.collectionID": [15, 15, 15, 15],
        "MCRecoAssociations#1/MCRecoAssociations#1.index": [0, 1, 2, 3],
        "MCRecoAssociations#1/MCRecoAssociations#1.collectionID": [8, 8, 8, 8],
        # Particle truth
        "Particle/Particle.PDG": [11, 211, 13, 22],  # e, pi, mu, gamma
        "Particle/Particle.momentum.x": [25.0, 5.0, 15.0, 3.0],
        "Particle/Particle.momentum.y": [5.0, 2.0, 3.0, 1.0],
        "Particle/Particle.momentum.z": [10.0, 3.0, 8.0, 2.0],
        "Particle/Particle.vertex.x": [0.0, 0.0, 0.0, 0.0],
        "Particle/Particle.vertex.y": [0.0, 0.0, 0.0, 0.0],
        "Particle/Particle.vertex.z": [0.0, 0.0, 0.0, 0.0],
        "Particle/Particle.parents_begin": [0, 0, 0, 0],
        "Particle/Particle.parents_end": [0, 0, 0, 0],
        "Particle#0/Particle#0.index": [],
        "Particle#0/Particle#0.collectionID": [],
    }
    chunk = {key: [value for _ in range(n_events)] for key, value in one_event.items()}
    _ = rng
    return ak.Array(chunk)


def test_build_rows_from_chunk_basic_shapes() -> None:
    schema = _load_reference_schema(
        default_schema_path(),
        ("jets", "tracks_ghost", "tracks_ghost_rich_ip", "flows", "electrons", "muons", "calo"),
    )
    chunk = _synthetic_chunk(n_events=2)
    (
        global_rows,
        jet_rows,
        track_rows,
        _rich_track_rows,
        flow_rows,
        electron_rows,
        muon_rows,
        diagnostics,
    ) = build_rows_from_chunk(
        chunk,
        ref_schema=schema,
        has_electrons=True,
        has_muons=True,
    )

    assert len(jet_rows) == 4  # 2 events × 2 jets
    assert global_rows.shape == (4,)
    assert track_rows.ndim == 2
    assert track_rows.shape[0] == 4
    assert flow_rows.shape[0] == 4
    assert electron_rows.shape[0] == 4
    assert muon_rows.shape[0] == 4

    # jets.pt_btagJes should be in MeV (≈ 1000× input GeV).
    assert np.all(jet_rows["pt_btagJes"] > 0)
    assert np.median(jet_rows["pt_btagJes"]) > 1000  # not in GeV

    assert diagnostics["n_events_total"] == 2
    assert diagnostics["n_events_empty"] == 0
    assert diagnostics["n_jets_total"] == 4
    assert diagnostics["n_jets_bad_kinematics"] == 0

    # The first jet of each event has 2 constituents → at least 1 valid flow slot.
    flow_valid = flow_rows["valid"].sum() if "valid" in flow_rows.dtype.names else 0
    assert flow_valid > 0


def test_build_rows_from_chunk_streaming_matches_singlepass() -> None:
    """Two consecutive chunks of 1 event must give the same rows as 1 chunk of 2.

    This guards against bugs at chunk boundaries (offsets, schema reuse, etc.).
    """
    schema = _load_reference_schema(
        default_schema_path(),
        ("jets", "tracks_ghost", "tracks_ghost_rich_ip", "flows", "electrons", "muons", "calo"),
    )

    one_pass = build_rows_from_chunk(
        _synthetic_chunk(n_events=2),
        ref_schema=schema,
        has_electrons=True,
        has_muons=True,
    )
    chunk_a = build_rows_from_chunk(
        _synthetic_chunk(n_events=1),
        ref_schema=schema,
        has_electrons=True,
        has_muons=True,
    )
    chunk_b = build_rows_from_chunk(
        _synthetic_chunk(n_events=1),
        ref_schema=schema,
        has_electrons=True,
        has_muons=True,
    )

    for label, idx in (("global", 0), ("jets", 1), ("tracks", 2), ("flows", 4)):
        merged = np.concatenate([chunk_a[idx], chunk_b[idx]], axis=0)
        assert merged.shape == one_pass[idx].shape, label
        assert merged.dtype == one_pass[idx].dtype, label
        # Compare a representative scalar field
        if label == "jets":
            np.testing.assert_array_equal(merged["pt_btagJes"], one_pass[idx]["pt_btagJes"])


def test_full_h5_pipeline_with_placeholder_resize() -> None:
    """End-to-end: synthetic chunk → SaltH5Writer → placeholders auto-filled.

    Validates that the calo dataset (placeholder) ends up with the right
    number of rows AND has valid=False everywhere thanks to fillvalue-on-resize.
    """
    schema_path = default_schema_path()

    chunk = _synthetic_chunk(n_events=3)

    from fcc_ftag_converter.reader import build_rows_from_chunk as _build
    from fcc_ftag_converter.schema import (
        FCC_TRAINING_DATASETS,
        GLOBAL_DTYPE,
        SaltH5Writer,
    )
    from fcc_ftag_converter.schema import (
        _load_reference_schema as _load,
    )

    ref_schema = _load(
        schema_path,
        tuple(s.name for s in FCC_TRAINING_DATASETS if s.name != "global"),
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "test.h5"
        with SaltH5Writer(
            out,
            schema_path,
            datasets=FCC_TRAINING_DATASETS,
            dtype_overrides={"global": GLOBAL_DTYPE},
        ) as writer:
            rows = _build(chunk, ref_schema=ref_schema, has_electrons=True, has_muons=True)
            (gr, jr, tr, rich, fl, el, mu, _diag) = rows
            n = len(jr)
            writer.append("global", gr)
            writer.append("jets", jr)
            writer.append("tracks_ghost", tr)
            writer.append("tracks_ghost_rich_ip", rich)
            writer.append("flows", fl)
            writer.append("electrons", el)
            writer.append("muons", mu)
            writer.extend_with_fill("calo", n)

        with h5py.File(out, "r") as f:
            assert f["jets"].shape == (n,)
            assert f["calo"].shape == (n, 100)
            if "valid" in f["calo"].dtype.names:
                # All calo rows must be valid=False because we only resized.
                assert not f["calo"]["valid"][...].any()


def test_schema_loader_preserves_nan_for_float_placeholders() -> None:
    """Reference H5 has NaN fillvalues for many float fields; the JSON loader
    must restore those so 'unset' is distinguishable from a real 0.0.
    """
    from fcc_ftag_converter.schema import _load_schema_from_json

    schema = _load_schema_from_json(default_schema_path(), ("calo", "jets", "muons"))
    for name in ("calo", "jets", "muons"):
        dtype, _shape, fillvalue = schema[name]
        if fillvalue is None:
            continue
        for field in dtype.names:
            sub = dtype.fields[field][0]
            if not np.issubdtype(sub, np.floating):
                continue
            # At least some float field must end up as NaN, not 0.0.
            # We test by checking a known-NaN field per dataset.
        # spot-check: jets.ptFromTruthJet, calo.ISOLATION, muons.momentumBalanceSignificance
    jets_fv = schema["jets"][2]
    assert np.isnan(jets_fv["ptFromTruthJet"])
    assert np.isnan(jets_fv["deltaRToNearestJet"])
    calo_fv = schema["calo"][2]
    assert np.isnan(calo_fv["ISOLATION"])
    muons_fv = schema["muons"][2]
    assert np.isnan(muons_fv["momentumBalanceSignificance"])
    # Integer placeholders stay at their concrete value (0 or -1), not NaN.
    assert int(jets_fv["matchedToTruthJet"]) == 0


def test_convert_fcc_edm4hep_to_h5_via_synthetic_chunk(monkeypatch) -> None:
    """Drive the full streaming orchestrator with a monkeypatched chunk source.

    Bypasses uproot entirely so the test has zero external deps beyond
    awkward + h5py + numpy.
    """
    import fcc_ftag_converter.cli as cli_mod

    def fake_iter(*_args, **_kwargs):
        yield _synthetic_chunk(n_events=2), set(), "fake.root", 0

    monkeypatch.setattr(cli_mod, "iter_event_chunks", fake_iter)

    with tempfile.TemporaryDirectory() as tmpdir:
        out = Path(tmpdir) / "out.h5"
        convert_fcc_edm4hep_to_h5(
            ["fake.root"],
            out,
            reference=str(default_schema_path()),
            quiet=True,
        )
        assert out.exists()
        assert (out.parent / f"{out.stem}.summary.md").exists()

        with h5py.File(out, "r") as f:
            assert f["jets"].shape[0] == 4
            assert f["calo"].shape == (4, 100)
            assert not f["calo"]["valid"][...].any()
            # eventNumber is stamped per event (2 events -> 2 distinct values),
            # shared by all jets of an event so the train/val/test split on
            # eventNumber % N never leaks a single event across the split.
            ev = f["jets"]["eventNumber"][...]
            assert len(np.unique(ev)) == 2
            assert (ev != 0).any()
            assert (f["jets"]["jetFoldHash"][...] != 0).any()
