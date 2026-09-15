"""Tests for the NumPy 2.0 legacy-alias compatibility shim."""

import numpy as np

from app.core import numpy_compat


def test_shim_restores_aliases_removed_in_numpy_2(monkeypatch):
    """The aliases ChromaDB < 0.5.0 needs at import time must resolve."""
    # Simulate NumPy 2 behaviour: drop the names the shim is meant to restore.
    for name in ("float_", "unicode_", "Inf", "NaN"):
        monkeypatch.delattr(np, name, raising=False)

    restored = numpy_compat.install_legacy_numpy_aliases()

    assert "float_" in restored
    assert np.float_ is np.float64
    assert np.unicode_ is np.str_
    assert np.Inf == np.inf
    assert np.isnan(np.NaN)


def test_shim_never_overrides_existing_attributes():
    """On NumPy 1.x (or a partially restored env) nothing must be replaced."""
    sentinel = object()
    original = np.float_
    np.float_ = sentinel
    try:
        restored = numpy_compat.install_legacy_numpy_aliases()
        assert np.float_ is sentinel
        assert "float_" not in restored
    finally:
        np.float_ = original


def test_chromadb_imports_after_shim():
    """Importing chromadb must not raise AttributeError on numpy 2.x."""
    chromadb = __import__("chromadb")
    assert chromadb is not None
