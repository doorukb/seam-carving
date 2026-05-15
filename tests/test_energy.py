import numpy as np
import pytest

import seam_carver as sc


def test_compute_energy_dtype_and_border():
    img = np.zeros((5, 6, 3), dtype=np.uint8)
    img[1:-1, 1:-1] = 100
    e = sc.compute_energy(img)
    assert e.dtype == np.float32
    assert np.all(e[0, :] == 1000)
    assert np.all(e[-1, :] == 1000)
    assert np.all(e[:, 0] == 1000)
    assert np.all(e[:, -1] == 1000)


def test_compute_energy_matches_luminance_gradients():
    """Interior matches sqrt(dx^2+dy^2) on luminance, same stencil as legacy RGB."""
    rng = np.random.RandomState(42)
    img = rng.randint(0, 256, size=(8, 9, 3), dtype=np.uint8)
    lum = (img.astype(np.float32) * sc._LUMA_W.reshape(1, 1, 3)).sum(axis=-1)
    dx = lum[:, 2:] - lum[:, :-2]
    dy = lum[2:, :] - lum[:-2, :]
    expected = np.sqrt(dx[1:-1, :] ** 2 + dy[:, 1:-1] ** 2).astype(np.float32)
    got = sc.compute_energy(img)[1:-1, 1:-1]
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_grayscale_image_energy():
    g = np.arange(12, dtype=np.uint8).reshape(3, 4)
    e2 = sc.compute_energy(g)
    e3 = sc.compute_energy(np.stack([g, g, g], axis=-1))
    np.testing.assert_allclose(e2, e3, rtol=1e-5, atol=1e-5)
