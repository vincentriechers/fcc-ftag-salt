"""Tests for fcc_ftag_converter.streaming URL and open-option handling."""

from __future__ import annotations

import pytest

from fcc_ftag_converter.streaming import (
    _normalise_uproot_path,
    _open_options,
    iter_event_chunks,
)


def test_normalise_rewrites_eospublic_root_urls() -> None:
    url = "root://eospublic.cern.ch//eos/experiment/fcc/ee/x/events_1.root"
    assert (
        _normalise_uproot_path(url)
        == "https://eospublic.cern.ch/eos/experiment/fcc/ee/x/events_1.root"
    )


@pytest.mark.parametrize(
    "path",
    [
        "/srv/data/events.root",
        "outputs/events.root",
        "root://other.host//eos/x.root",
    ],
)
def test_normalise_leaves_other_paths_alone(path: str) -> None:
    assert _normalise_uproot_path(path) == path


def test_open_options_empty_for_non_http_paths() -> None:
    """Only http(s) inputs need the uproot HTTP source and its timeout."""
    assert _open_options("/srv/data/events.root") == {}
    assert _open_options("root://eospublic.cern.ch//eos/x.root") == {}


def test_iter_event_chunks_rejects_empty_input() -> None:
    pytest.importorskip("uproot")
    with pytest.raises(ValueError, match="no inputs given"):
        next(iter_event_chunks([], max_events=None, chunk_size=10))
