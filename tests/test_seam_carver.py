"""Headless tests for the seam-carving core."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from seam_carver import (  # noqa: E402
    carve_horizontal_seams,
    carve_vertical_seams,
    compute_energy,
    find_vertical_seam,
    remove_vertical_seam,
)


def _stripe_image(h=30, w=30):
    """Flat gray image with a bright, high-contrast vertical stripe."""
    img = np.full((h, w, 3), 60, dtype=np.uint8)
    img[:, 14:17] = 250
    return img


def test_energy_is_zero_on_uniform_interior():
    img = np.full((20, 20, 3), 128, dtype=np.uint8)
    energy = compute_energy(img)
    assert np.all(energy[1:-1, 1:-1] == 0)


def test_energy_rejects_unknown_gradient():
    with pytest.raises(ValueError):
        compute_energy(np.zeros((5, 5)), gradient="prewitt")


@pytest.mark.parametrize("gradient", ["central", "sobel", "scharr"])
def test_energy_highlights_the_stripe(gradient):
    energy = compute_energy(_stripe_image(), gradient=gradient)
    interior = energy[1:-1, 1:-1]
    stripe_edges = interior[:, 12:17]
    flat = interior[:, :10]
    assert stripe_edges.max() > flat.max()


def test_vertical_seam_is_connected_and_valid():
    img = _stripe_image()
    seam = np.asarray(find_vertical_seam(img))
    assert seam.shape[0] == img.shape[0]
    assert np.all(seam >= 0) and np.all(seam < img.shape[1])
    assert np.all(np.abs(np.diff(seam)) <= 1), "seam must be 8-connected"


def test_remove_vertical_seam_shrinks_width_by_one():
    img = _stripe_image()
    seam = np.asarray(find_vertical_seam(img))
    out = remove_vertical_seam(img, seam)
    assert out.shape == (img.shape[0], img.shape[1] - 1, 3)


def test_carving_preserves_high_energy_content():
    img = _stripe_image()
    out = carve_vertical_seams(img, 8)
    assert out.shape[1] == img.shape[1] - 8
    # the bright stripe is the most significant content; it must survive
    bright_per_row = (out == 250).all(axis=2).sum(axis=1)
    assert np.all(bright_per_row >= 3)


def test_horizontal_carving_shrinks_height():
    img = np.ascontiguousarray(_stripe_image().transpose(1, 0, 2))
    out = carve_horizontal_seams(img, 6)
    assert out.shape[0] == img.shape[0] - 6
