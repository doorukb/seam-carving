"""Carving tests."""

from __future__ import annotations

import numpy as np

from seam_carver import carve_horizontal_seams, carve_vertical_seams


def test_carve_vertical_shrink_width() -> None:
    img = np.random.randint(0, 255, size=(24, 32, 3), dtype=np.uint8)
    out = carve_vertical_seams(img, 4)
    assert out.shape == (24, 28, 3)
    assert out.dtype == np.uint8


def test_carve_horizontal_shrink_height() -> None:
    img = np.random.randint(0, 255, size=(24, 32, 3), dtype=np.uint8)
    out = carve_horizontal_seams(img, 5)
    assert out.shape == (19, 32, 3)
    assert out.dtype == np.uint8


def test_carve_vertical_zero_seams_unchanged_shape() -> None:
    img = np.random.randint(0, 255, size=(8, 10, 3), dtype=np.uint8)
    out = carve_vertical_seams(img, 0)
    assert out.shape == img.shape
    np.testing.assert_array_equal(out, img)
