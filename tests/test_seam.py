import numpy as np
import pytest
from unittest import mock

import seam_carver as sc


def _reference_vertical_seam(energy: np.ndarray) -> np.ndarray:
    """Legacy full OPT matrix + forward traceback (float64)."""
    energy = np.asarray(energy, dtype=np.float64)
    image_height, image_width = energy.shape
    opt = energy.copy()
    row = image_height - 2
    while row >= 0:
        previous_row = opt[row + 1]
        left_costs = np.full(image_width, np.inf)
        left_costs[1:] = previous_row[:-1]
        right_costs = np.full(image_width, np.inf)
        right_costs[:-1] = previous_row[1:]
        straight_costs = previous_row
        best_costs_below = np.minimum(
            np.minimum(left_costs, straight_costs), right_costs
        )
        opt[row] += best_costs_below
        row -= 1

    seam = np.zeros(image_height, dtype=int)
    seam[0] = np.argmin(opt[0])
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
        best = np.argmin(sliced)
        seam[j] = minimum_column + best
        j += 1
    return seam


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_vertical_seam_matches_legacy_dp(seed):
    rng = np.random.RandomState(seed)
    e = rng.rand(12, 15).astype(np.float32) * 10 + 0.01
    ref = _reference_vertical_seam(e)
    got = sc.find_vertical_seam_from_energy(e)
    np.testing.assert_array_equal(got, ref)


def test_tie_breaking_three_way_equal():
    """When three neighbors tie, legacy picks 'left' (col -1 below)."""
    e = np.ones((4, 5), dtype=np.float32) * 3
    ref = _reference_vertical_seam(e)
    got = sc.find_vertical_seam_from_energy(e)
    np.testing.assert_array_equal(got, ref)


def test_horizontal_seam_energy_path_no_vertical_seam_on_image():
    """Precomputed energy must not run find_vertical_seam on a transposed RGB tensor."""
    energy = np.arange(20, dtype=np.float32).reshape(4, 5)
    dummy = np.zeros((4, 5, 3), dtype=np.uint8)

    def _boom(*_a, **_k):
        raise AssertionError("transpose RGB / find_vertical_seam(image,…) path must not run")

    with mock.patch.object(sc, "find_vertical_seam", _boom):
        seam = sc.find_horizontal_seam(dummy, energy=energy)

    assert seam.shape == (5,)
    et = np.transpose(energy)
    assert np.shares_memory(energy, et)


def test_horizontal_equals_vertical_on_transpose_no_noise():
    img = np.random.RandomState(7).randint(0, 256, size=(6, 8, 3), dtype=np.uint8)
    e = sc.compute_energy(img)
    h1 = sc.find_horizontal_seam(img, energy=e)
    v_on_t = sc.find_vertical_seam_from_energy(np.transpose(e))
    np.testing.assert_array_equal(h1, v_on_t)
