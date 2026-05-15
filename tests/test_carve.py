import numpy as np
import pytest

import seam_carver as sc


def _carve_vertical_mask(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    mask = np.full(image.shape, True, dtype=bool)
    mask[np.arange(h), seam, :] = False
    return image[mask].reshape(h, w - 1, c)


def _carve_horizontal_reference(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    out = np.empty((h - 1, w, c), dtype=image.dtype)
    for j in range(w):
        s = int(seam[j])
        out[:s, j] = image[:s, j]
        out[s:, j] = image[s + 1 :, j]
    return out


def test_remove_vertical_matches_mask():
    rng = np.random.RandomState(0)
    img = rng.randint(0, 256, size=(7, 9, 3), dtype=np.uint8)
    seam = np.array([1, 2, 1, 3, 2, 0, 4], dtype=np.int64)
    a = sc.remove_vertical_seam(img, seam)
    b = _carve_vertical_mask(img, seam)
    np.testing.assert_array_equal(a, b)


def test_remove_horizontal_matches_mask():
    rng = np.random.RandomState(1)
    img = rng.randint(0, 256, size=(8, 5, 3), dtype=np.uint8)
    seam = np.array([2, 1, 0, 3, 4], dtype=np.int64)
    a = sc.remove_horizontal_seam(img, seam)
    b = _carve_horizontal_reference(img, seam)
    np.testing.assert_array_equal(a, b)


def test_remove_vertical_roundtrip_with_found_seam():
    img = np.random.RandomState(3).randint(0, 256, size=(10, 12, 3), dtype=np.uint8)
    seam = sc.find_vertical_seam(img)
    carved = sc.remove_vertical_seam(img, seam)
    assert carved.shape == (10, 11, 3)
