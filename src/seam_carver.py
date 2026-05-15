"""Pure NumPy seam carving (energy map, seam finding, carving)."""

from __future__ import annotations

import numpy as np

# BT.601 luma weights (same order as RGB)
_LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def compute_energy(image: np.ndarray) -> np.ndarray:
    """Sobel-like energy on luminance (float32); boundary padded with 1000."""
    image_height = image.shape[0]
    image_width = image.shape[1]
    energy = np.full((image_height, image_width), 1000.0, dtype=np.float32)

    if image.ndim == 2:
        lum = image.astype(np.float32, copy=False)
    else:
        lum = (image.astype(np.float32, copy=False) * _LUMA_W).sum(axis=-1)

    dx = lum[:, 2:] - lum[:, :-2]
    dy = lum[2:, :] - lum[:-2, :]
    energy[1:-1, 1:-1] = np.sqrt(dx[1:-1, :] ** 2 + dy[:, 1:-1] ** 2).astype(np.float32)
    return energy


def _vertical_seam_dp(energy: np.ndarray) -> np.ndarray:
    """
    Rolling two-row cumulative DP + int8 backpointers (-1, 0, +1).
    Tie-breaking matches legacy np.minimum(np.minimum(left, straight), right).
    """
    energy = np.asarray(energy, dtype=np.float32)
    h, w = energy.shape
    if h == 0:
        return np.zeros(0, dtype=np.int64)
    if w == 1:
        return np.zeros(h, dtype=np.int64)

    if h == 1:
        return np.array([int(np.argmin(energy[0]))], dtype=np.int64)

    below = energy[h - 1].copy()
    backptr = np.zeros((h - 1, w), dtype=np.int8)
    left = np.empty(w, dtype=np.float32)
    right = np.empty(w, dtype=np.float32)
    cur = np.empty(w, dtype=np.float32)

    for r in range(h - 2, -1, -1):
        left[0] = np.inf
        left[1:] = below[:-1]
        right[-1] = np.inf
        right[:-1] = below[1:]
        straight = below
        best = np.minimum(np.minimum(left, straight), right)
        take_left = (left <= straight) & (left <= right)
        take_mid = (~take_left) & (straight <= right)
        ptr = np.where(take_left, np.int8(-1), np.where(take_mid, np.int8(0), np.int8(1)))
        cur[:] = energy[r] + best
        backptr[r] = ptr
        below, cur = cur, below

    seam = np.zeros(h, dtype=np.int64)
    seam[0] = int(np.argmin(below))
    for r in range(h - 1):
        seam[r + 1] = seam[r] + int(backptr[r, seam[r]])
    return seam


def find_vertical_seam_from_energy(energy: np.ndarray) -> np.ndarray:
    """Minimum vertical seam from a precomputed 2D energy map (no image access)."""
    return _vertical_seam_dp(energy)


def find_vertical_seam(image: np.ndarray, energy: np.ndarray | None = None) -> np.ndarray:
    if energy is None:
        energy = compute_energy(image)

        random_state = np.random.get_state()
        np.random.seed(0)
        noise = np.random.randn(*energy.shape).astype(np.float32)
        noise /= np.float32(1000.0 * (image.size ** 0.5))
        energy = energy + noise
        np.random.set_state(random_state)

    return find_vertical_seam_from_energy(energy)


def find_horizontal_seam(image: np.ndarray, energy: np.ndarray | None = None) -> np.ndarray:
    if energy is not None:
        et = np.transpose(energy)
        if not et.flags.c_contiguous:
            et = np.ascontiguousarray(et)
        return find_vertical_seam_from_energy(et)
    return find_vertical_seam(np.transpose(image, (1, 0, 2)))


def remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Drop one pixel per row along ``seam`` without boolean mask indexing."""
    h, w, c = image.shape
    out = np.empty((h, w - 1, c), dtype=image.dtype)
    for i in range(h):
        s = int(seam[i])
        out[i, :s] = image[i, :s]
        out[i, s:] = image[i, s + 1 :]
    return out


def remove_horizontal_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    """Drop one pixel per column along ``seam`` without boolean mask indexing."""
    h, w, c = image.shape
    out = np.empty((h - 1, w, c), dtype=image.dtype)
    for j in range(w):
        s = int(seam[j])
        out[:s, j] = image[:s, j]
        out[s:, j] = image[s + 1 :, j]
    return out


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
