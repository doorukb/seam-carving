"""Seam finder regression tests."""

from __future__ import annotations

import numpy as np

from seam_carver import compute_energy, find_vertical_seam


def apply_vertical_seam_noise(energy: np.ndarray, image: np.ndarray) -> np.ndarray:
    random_state = np.random.get_state()
    np.random.seed(0)
    noise = np.random.randn(*energy.shape) / (1000 * (image.size**0.5))
    out = energy + noise
    np.random.set_state(random_state)
    return out


def _reference_find_vertical_seam(image: np.ndarray, energy: np.ndarray | None = None) -> np.ndarray:
    if energy is None:
        energy = compute_energy(image)
        random_state = np.random.get_state()
        np.random.seed(0)
        noise = np.random.randn(*energy.shape) / (1000 * (image.size**0.5))
        energy = energy + noise
        np.random.set_state(random_state)

    image_height = energy.shape[0]
    image_width = energy.shape[1]
    opt = energy.copy().astype(float)
    row = image_height - 2
    while row >= 0:
        previous_row = opt[row + 1]
        left_costs = np.full(image_width, np.inf)
        left_costs[1:] = previous_row[:-1]
        right_costs = np.full(image_width, np.inf)
        right_costs[:-1] = previous_row[1:]
        straight_costs = previous_row
        best_costs_below = np.minimum(np.minimum(left_costs, straight_costs), right_costs)
        opt[row] += best_costs_below
        row -= 1

    seam = np.zeros(image_height, dtype=int)
    seam[0] = int(np.argmin(opt[0]))
    j = 1
    while j < image_height:
        previous_column = seam[j - 1]
        minimum_column = previous_column - 1
        maximum_column = previous_column + 1
        if minimum_column < 0:
            minimum_column = 0
        if maximum_column > image_width - 1:
            maximum_column = image_width - 1
        sliced = opt[j][minimum_column : maximum_column + 1]
        best = int(np.argmin(sliced))
        seam[j] = minimum_column + best
        j += 1
    return seam


def test_vertical_seam_matches_reference_small() -> None:
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(17, 23, 3), dtype=np.uint8)
    e = compute_energy(img)
    e_noisy = apply_vertical_seam_noise(e, img)
    ref = _reference_find_vertical_seam(img, e_noisy)
    got = find_vertical_seam(img, e_noisy)
    np.testing.assert_array_equal(got, ref)


def test_vertical_seam_auto_energy_matches_reference() -> None:
    rng = np.random.default_rng(2)
    img = rng.integers(0, 256, size=(25, 31, 3), dtype=np.uint8)
    ref = _reference_find_vertical_seam(img, None)
    got = find_vertical_seam(img)
    np.testing.assert_array_equal(got, ref)


def test_vertical_seam_explicit_energy_no_noise() -> None:
    img = np.zeros((10, 12, 3), dtype=np.uint8)
    e = np.ones((10, 12), dtype=np.float64)
    s = find_vertical_seam(img, e)
    assert s.shape == (10,)
    assert np.all((s >= 0) & (s < 12))
