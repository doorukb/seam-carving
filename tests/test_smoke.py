"""Smoke test: package imports from installed / src layout."""

from __future__ import annotations

import numpy as np

import seam_carver as sc


def test_import_public_api() -> None:
    assert hasattr(sc, "compute_energy")
    assert hasattr(sc, "find_vertical_seam")
    assert hasattr(sc, "find_horizontal_seam")
    assert hasattr(sc, "carve_vertical_seams")
    assert hasattr(sc, "carve_horizontal_seams")


def test_compute_energy_runs() -> None:
    img = np.zeros((8, 10, 3), dtype=np.uint8)
    e = sc.compute_energy(img)
    assert e.shape == (8, 10)
    assert e.dtype == np.float64
