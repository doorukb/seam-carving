"""Pure NumPy seam carving (energy map, seam finding, carving)."""

from __future__ import annotations

import numpy as np


def compute_energy(image: np.ndarray) -> np.ndarray:
    image_height = image.shape[0]
    image_width = image.shape[1]
    energy = np.full((image_height, image_width), 1000, dtype=float)
    image_float = image.astype(float)

    left = image_float[:, :-2]
    right = image_float[:, 2:]
    x_difference = right - left

    top = image_float[:-2]
    bottom = image_float[2:]
    y_difference = bottom - top

    energy[1:-1, 1:-1] = np.sqrt(
        np.sum(x_difference[1:-1] ** 2, axis=-1) + np.sum(y_difference[:, 1:-1] ** 2, axis=-1)
    )
    return energy


def find_vertical_seam(image: np.ndarray, energy: np.ndarray | None = None) -> np.ndarray:
    if energy is None:
        energy = compute_energy(image)

        random_state = np.random.get_state()
        np.random.seed(0)
        noise = np.random.randn(*energy.shape) / (1000 * (image.size ** (0.5)))
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


def find_horizontal_seam(image: np.ndarray, energy: np.ndarray | None = None) -> np.ndarray:
    if energy is not None:
        energy = energy.transpose(1, 0)
    return find_vertical_seam(image.transpose(1, 0, 2), energy=energy)


def _remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    rows = np.arange(h, dtype=int)
    mask = np.ones((h, w), dtype=bool)
    mask[rows, seam] = False
    return image[mask].reshape(h, w - 1, c)


def _remove_horizontal_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    h, w, c = image.shape
    cols = np.arange(w, dtype=int)
    mask = np.full(image.shape, True, dtype=bool)
    mask[seam, cols] = False
    return (
        image.transpose(1, 0, 2)[mask.transpose(1, 0, 2)]
        .reshape(w, h - 1, c)
        .transpose(1, 0, 2)
    )


def carve_vertical_seams(image: np.ndarray, n_seams: int) -> np.ndarray:
    """Remove ``n_seams`` vertical seams. Carves on integer RGB; returns ``uint8``."""
    work = np.asarray(image, dtype=np.int64)
    if work.ndim != 3 or work.shape[2] != 3:
        raise ValueError("Expected RGB array with shape (height, width, 3).")
    n = int(n_seams)
    if n < 0:
        raise ValueError("n_seams must be non-negative.")
    max_removable = max(work.shape[1] - 1, 0)
    n = min(n, max_removable)
    for _ in range(n):
        seam = find_vertical_seam(work)
        work = _remove_vertical_seam(work, seam)
    return np.clip(work, 0, 255).astype(np.uint8)


def carve_horizontal_seams(image: np.ndarray, n_seams: int) -> np.ndarray:
    """Remove ``n_seams`` horizontal seams. Carves on integer RGB; returns ``uint8``."""
    work = np.asarray(image, dtype=np.int64)
    if work.ndim != 3 or work.shape[2] != 3:
        raise ValueError("Expected RGB array with shape (height, width, 3).")
    n = int(n_seams)
    if n < 0:
        raise ValueError("n_seams must be non-negative.")
    max_removable = max(work.shape[0] - 1, 0)
    n = min(n, max_removable)
    for _ in range(n):
        seam = find_horizontal_seam(work)
        work = _remove_horizontal_seam(work, seam)
    return np.clip(work, 0, 255).astype(np.uint8)
