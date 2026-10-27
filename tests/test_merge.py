"""Tests for fcc_ftag_converter.merge."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from fcc_ftag_converter.merge import merge_h5_files  # noqa: E402
from fcc_ftag_converter.parallel import _split_evenly  # noqa: E402
from fcc_ftag_converter.schema import (  # noqa: E402
    FCC_TRAINING_DATASETS,
    GLOBAL_DTYPE,
    SaltH5Writer,
    _load_reference_schema,
    default_schema_path,
)


def _write_dummy_h5(path: Path, n_jets: int, pt_offset: float) -> None:
    """Write a minimal Salt-shaped H5 with ``n_jets`` jets, all valid=False except a known pt."""
    ref_schema = _load_reference_schema(
        default_schema_path(),
        tuple(s.name for s in FCC_TRAINING_DATASETS if s.name != "global"),
    )
    with SaltH5Writer(
        path,
        default_schema_path(),
        datasets=FCC_TRAINING_DATASETS,
        dtype_overrides={"global": GLOBAL_DTYPE},
    ) as writer:
        from fcc_ftag_converter.schema import empty_rows_from_reference

        gr = np.zeros(n_jets, dtype=GLOBAL_DTYPE)
        gr["pt_btagJes"] = np.arange(n_jets, dtype=np.float32) + pt_offset
        jr = empty_rows_from_reference(ref_schema, "jets", n_jets)
        jr["pt_btagJes"] = np.arange(n_jets, dtype=np.float32) + pt_offset
        writer.append("global", gr)
        writer.append("jets", jr)
        for name in ("tracks_ghost", "tracks_ghost_rich_ip", "flows", "electrons", "muons"):
            writer.append(name, empty_rows_from_reference(ref_schema, name, n_jets))
        writer.extend_with_fill("calo", n_jets)


def _write_dummy_higgs_decay_h5(path: Path, n_jets: int, pt_offset: float) -> None:
    """Write a higgs_decay-shaped H5: only ``jets`` + ``pfcand``, no ATLAS streams."""
    jets = np.zeros(n_jets, dtype=[("jet_pt", "<f4"), ("flavour_label", "<i4")])
    jets["jet_pt"] = np.arange(n_jets, dtype=np.float32) + pt_offset
    pfcand = np.zeros((n_jets, 4), dtype=[("pfcand_erel_log", "<f4"), ("valid", "?")])
    pfcand["pfcand_erel_log"] = np.arange(n_jets * 4, dtype=np.float32).reshape(n_jets, 4)
    pfcand["valid"] = True
    with h5py.File(path, "w") as f:
        f.create_dataset("jets", data=jets, maxshape=(None,), chunks=(2048,))
        f.create_dataset("pfcand", data=pfcand, maxshape=(None, 4), chunks=(2048, 4))


def test_split_evenly_roundrobin() -> None:
    assert _split_evenly(["a", "b", "c", "d", "e"], 2) == [["a", "c", "e"], ["b", "d"]]
    assert _split_evenly(["a"], 4) == [["a"]]
    assert _split_evenly([], 4) == []


def test_merge_two_files_preserves_order_and_totals() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        a = Path(tmpdir) / "a.h5"
        b = Path(tmpdir) / "b.h5"
        out = Path(tmpdir) / "merged.h5"
        _write_dummy_h5(a, n_jets=5, pt_offset=100)
        _write_dummy_h5(b, n_jets=3, pt_offset=1000)
        info = merge_h5_files([a, b], out, verbose=False)

        assert info["n_inputs"] == 2
        assert info["n_jets"] == 8

        with h5py.File(out, "r") as f:
            assert f["jets"].shape == (8,)
            assert f["calo"].shape == (8, 100)
            assert not f["calo"]["valid"][...].any()
            pts = f["jets"]["pt_btagJes"][...]
            np.testing.assert_array_equal(pts[:5], np.arange(5) + 100)
            np.testing.assert_array_equal(pts[5:], np.arange(3) + 1000)


def test_merge_idempotent_for_single_input() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        a = Path(tmpdir) / "a.h5"
        out = Path(tmpdir) / "merged.h5"
        _write_dummy_h5(a, n_jets=4, pt_offset=42)
        merge_h5_files([a], out, verbose=False)

        with h5py.File(a, "r") as src, h5py.File(out, "r") as merged:
            np.testing.assert_array_equal(
                src["jets"]["pt_btagJes"][...], merged["jets"]["pt_btagJes"][...]
            )
            assert src["calo"].shape == merged["calo"].shape


def test_merge_rejects_incompatible_schema() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        a = Path(tmpdir) / "a.h5"
        b = Path(tmpdir) / "b.h5"
        _write_dummy_h5(a, n_jets=2, pt_offset=0)
        with h5py.File(b, "w") as f:
            f.create_dataset("foo", data=np.arange(3))
        with pytest.raises(ValueError, match="incompatible"):
            merge_h5_files([a, b], Path(tmpdir) / "merged.h5", verbose=False)


def test_merge_accumulator_picks_up_stats() -> None:
    from fcc_ftag_converter.report import ReportAccumulator

    with tempfile.TemporaryDirectory() as tmpdir:
        a = Path(tmpdir) / "a.h5"
        b = Path(tmpdir) / "b.h5"
        _write_dummy_h5(a, n_jets=10, pt_offset=1000)
        _write_dummy_h5(b, n_jets=10, pt_offset=2000)
        acc = ReportAccumulator()
        merge_h5_files([a, b], Path(tmpdir) / "out.h5", accumulator=acc, verbose=False)
        summary = acc.finalize()
        assert summary["n_jets"] == 20
        assert "jets.pt_btagJes" in summary["stats_median_p95"]
        # Median of [1000..1009, 2000..2009] = 1504.5
        assert 1400 < summary["stats_median_p95"]["jets.pt_btagJes"]["median"] < 1700


def test_merge_higgs_decay_copies_pfcand_and_reports() -> None:
    """higgs_decay files have only jets + pfcand: pfcand must be copied, not filled."""
    from fcc_ftag_converter.report import ReportAccumulator

    with tempfile.TemporaryDirectory() as tmpdir:
        a = Path(tmpdir) / "a.h5"
        b = Path(tmpdir) / "b.h5"
        out = Path(tmpdir) / "merged.h5"
        _write_dummy_higgs_decay_h5(a, n_jets=5, pt_offset=10)
        _write_dummy_higgs_decay_h5(b, n_jets=3, pt_offset=100)
        acc = ReportAccumulator()
        info = merge_h5_files([a, b], out, accumulator=acc, verbose=False)

        assert info["n_jets"] == 8
        with h5py.File(out, "r") as f:
            assert f["pfcand"].shape == (8, 4)
            assert f["pfcand"]["valid"][...].all()
            np.testing.assert_array_equal(
                f["pfcand"]["pfcand_erel_log"][:5],
                np.arange(20, dtype=np.float32).reshape(5, 4),
            )
            np.testing.assert_array_equal(f["jets"]["jet_pt"][5:], np.arange(3) + 100)
        assert acc.finalize()["n_jets"] == 8
