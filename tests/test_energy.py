"""Energy map tests."""

from __future__ import annotations

import numpy as np

from seam_carver import compute_energy


def test_backward_border_constant() -> None:
    img = np.random.randint(0, 255, size=(12, 16, 3), dtype=np.uint8)
    e = compute_energy(img)
    assert np.all(e[0, :] == 1000)
    assert np.all(e[-1, :] == 1000)
    assert np.all(e[:, 0] == 1000)
    assert np.all(e[:, -1] == 1000)
