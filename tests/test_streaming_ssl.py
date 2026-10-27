"""The unverified SSL context must stay scoped to uproot's own connections."""

from __future__ import annotations

import ssl

import pytest

uproot = pytest.importorskip("uproot")

from fcc_ftag_converter.streaming import _patch_uproot_https_verification  # noqa: E402


def test_patch_does_not_touch_the_global_https_context() -> None:
    before = ssl._create_default_https_context
    _patch_uproot_https_verification()
    assert ssl._create_default_https_context is before


def test_patch_is_idempotent() -> None:
    import uproot.source.http as uproot_http

    _patch_uproot_https_verification()
    patched = uproot_http.make_connection
    _patch_uproot_https_verification()
    assert uproot_http.make_connection is patched
    assert uproot_http._fcc_unverified_https is True


def test_patched_connection_uses_an_unverified_context() -> None:
    import urllib.parse

    import uproot.source.http as uproot_http

    _patch_uproot_https_verification()
    conn = uproot_http.make_connection(urllib.parse.urlparse("https://example.invalid/x.root"), 5)
    assert conn._context.verify_mode == ssl.CERT_NONE
    assert conn._context.check_hostname is False
