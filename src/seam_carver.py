import sys
from collections.abc import Callable
import numpy as np
from PIL import Image

GIF_MAX_SEAMS = 40
GIF_FRAME_MAX_SIZE = (480, 480)
SAMPLE_IMAGE_PATH = "your_image.jpg"


def compute_energy(image: np.ndarray, gradient: str = "central") -> np.ndarray:
    if gradient not in ("central", "sobel", "scharr"):
        raise ValueError("gradient must be 'central', 'sobel', or 'scharr'")

    image_height = image.shape[0]
    image_width = image.shape[1]
    energy = np.full((image_height, image_width), 1000.0, dtype=np.float32)

    if image.ndim == 2:
        gray = image.astype(np.float32, copy=False)
    else:
        r = image[..., 0].astype(np.float32, copy=False)
        g = image[..., 1].astype(np.float32, copy=False)
        b = image[..., 2].astype(np.float32, copy=False)
        gray = 0.299 * r + 0.587 * g + 0.114 * b

    if gradient == "central":
        left = gray[:, :-2]
        right = gray[:, 2:]
        x_difference = right - left

        top = gray[:-2]
        bottom = gray[2:]
        y_difference = bottom - top

        energy[1:-1, 1:-1] = np.sqrt(
            x_difference[1:-1] ** 2 + y_difference[:, 1:-1] ** 2
        ).astype(np.float32, copy=False)
        return energy

    if gradient == "sobel":
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
        ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
    else:
        kx = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float32)
        ky = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float32)

    gx = np.zeros_like(gray, dtype=np.float32)
    gy = np.zeros_like(gray, dtype=np.float32)
    gx[1:-1, 1:-1] = (
        kx[0, 0] * gray[:-2, :-2]
        + kx[0, 1] * gray[:-2, 1:-1]
        + kx[0, 2] * gray[:-2, 2:]
        + kx[1, 0] * gray[1:-1, :-2]
        + kx[1, 1] * gray[1:-1, 1:-1]
        + kx[1, 2] * gray[1:-1, 2:]
        + kx[2, 0] * gray[2:, :-2]
        + kx[2, 1] * gray[2:, 1:-1]
        + kx[2, 2] * gray[2:, 2:]
    )
    gy[1:-1, 1:-1] = (
        ky[0, 0] * gray[:-2, :-2]
        + ky[0, 1] * gray[:-2, 1:-1]
        + ky[0, 2] * gray[:-2, 2:]
        + ky[1, 0] * gray[1:-1, :-2]
        + ky[1, 1] * gray[1:-1, 1:-1]
        + ky[1, 2] * gray[1:-1, 2:]
        + ky[2, 0] * gray[2:, :-2]
        + ky[2, 1] * gray[2:, 1:-1]
        + ky[2, 2] * gray[2:, 2:]
    )
    energy[1:-1, 1:-1] = np.sqrt(gx[1:-1, 1:-1] ** 2 + gy[1:-1, 1:-1] ** 2).astype(np.float32, copy=False)
    return energy


def _vertical_dp_with_backpointers(energy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image_height, image_width = energy.shape
    dtype = np.float32
    e = np.asarray(energy, dtype=dtype)
    backptr = np.zeros((image_height, image_width), dtype=np.int8)

    below = e[image_height - 1].copy()
    tmp_left = np.empty(image_width, dtype=dtype)
    tmp_right = np.empty(image_width, dtype=dtype)
    stacked = np.empty((3, image_width), dtype=dtype)

    row = image_height - 2
    while row >= 0:
        tmp_left[:] = np.inf
        tmp_left[1:] = below[:-1]
        tmp_right[:] = np.inf
        tmp_right[:-1] = below[1:]
        stacked[0] = tmp_left
        stacked[1] = below
        stacked[2] = tmp_right
        best_below = np.min(stacked, axis=0)
        cur = e[row] + best_below
        idx = np.argmin(stacked, axis=0).astype(np.int8)
        backptr[row] = idx - 1
        below = cur
        row -= 1

    return below, backptr


def find_vertical_seam(image: np.ndarray, energy=None, gradient: str = "central"):
    if energy is None:
        energy = compute_energy(image, gradient=gradient)

        # Tiny deterministic tie-break on quantized RGB energy; restore global RNG afterward.
        random_state = np.random.get_state()
        np.random.seed(0)
        noise = np.random.randn(*energy.shape).astype(np.float32) / (1000 * (image.size ** (0.5)))
        energy = energy + noise
        np.random.set_state(random_state)

    energy = np.asarray(energy, dtype=np.float32)
    image_height = energy.shape[0]
    image_width = energy.shape[1]

    top_costs, backptr = _vertical_dp_with_backpointers(energy)
    seam = np.zeros(image_height, dtype=int)
    seam[0] = int(np.argmin(top_costs))

    j = 1
    while j < image_height:
        seam[j] = seam[j - 1] + int(backptr[j - 1, seam[j - 1]])
        j += 1

    return seam


def find_horizontal_seam(image: np.ndarray, energy=None, gradient: str = "central"):
    if energy is not None:
        energy = energy.transpose(1, 0)
    return find_vertical_seam(image.transpose(1, 0, 2), energy=energy, gradient=gradient)


def remove_vertical_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    # Remove one vertical seam using an (H, W-1, C) output and row-wise copies (no HxWx3 mask).
    h, w, channels = image.shape
    seam = np.asarray(seam, dtype=int)
    out = np.empty((h, w - 1, channels), dtype=image.dtype)
    for i in range(h):
        c = int(seam[i])
        out[i, :c] = image[i, :c]
        out[i, c:] = image[i, c + 1 :]
    return out

def remove_horizontal_seam(image: np.ndarray, seam: np.ndarray) -> np.ndarray:
    # seam length equals image width (one row index per column)
    img_t = np.transpose(image, (1, 0, 2))
    out_t = remove_vertical_seam(img_t, seam)
    return np.transpose(out_t, (1, 0, 2))

def _stderr_seam_progress(label: str, iteration_index: int, total: int) -> None:
    err = sys.stderr
    msg = f"{label} {iteration_index + 1}/{total}"
    if err.isatty():
        err.write(f"\r{msg}")
        err.flush()
    else:
        err.write(f"{msg}\n")
        err.flush()

def _stderr_seam_progress_finish() -> None:
    err = sys.stderr
    if err.isatty():
        err.write("\n")
        err.flush()

def carve_vertical_seams(image: np.ndarray, n_seams: int, show_progress: bool = False, on_seam_step: Callable[[], None] | None = None) -> np.ndarray:
    # Remove n_seams vertical seams. Carves on integer RGB; returns uint8
    work = np.asarray(image, dtype=np.int64)
    if work.ndim != 3 or work.shape[2] != 3:
        raise ValueError("Expected RGB array with shape (height, width, 3).")
    n = int(n_seams)
    if n < 0:
        raise ValueError("n_seams must be non-negative.")
    max_removable = max(work.shape[1] - 1, 0)
    n = min(n, max_removable)
    for i in range(n):
        if show_progress:
            _stderr_seam_progress("Vertical carve (final):", i, n)
        seam = find_vertical_seam(work)
        work = remove_vertical_seam(work, seam)
        if on_seam_step is not None:
            on_seam_step()
    if show_progress and n > 0:
        _stderr_seam_progress_finish()
    return np.clip(work, 0, 255).astype(np.uint8)

def carve_horizontal_seams(image: np.ndarray, n_seams: int, show_progress: bool = False, on_seam_step: Callable[[], None] | None = None) -> np.ndarray:
    work = np.asarray(image, dtype=np.int64)
    if work.ndim != 3 or work.shape[2] != 3:
        raise ValueError("Expected RGB array with shape (height, width, 3).")
    n = int(n_seams)
    if n < 0:
        raise ValueError("n_seams must be non-negative.")
    max_removable = max(work.shape[0] - 1, 0)
    n = min(n, max_removable)
    for i in range(n):
        if show_progress:
            _stderr_seam_progress("Horizontal carve (final):", i, n)
        seam = find_horizontal_seam(work)
        work = remove_horizontal_seam(work, seam)
        if on_seam_step is not None:
            on_seam_step()
    if show_progress and n > 0:
        _stderr_seam_progress_finish()
    return np.clip(work, 0, 255).astype(np.uint8)

def _append_black_padding(image: np.ndarray, original_shape: tuple[int, ...], axis: int) -> np.ndarray:
    oh, ow, oc = original_shape
    h, w, c = image.shape
    if axis == 1:
        pad = ow - w
        if pad <= 0:
            return image
        return np.append(image, np.zeros((h, pad, c)), axis=1)
    if axis == 0:
        pad = oh - h
        if pad <= 0:
            return image
        return np.append(image, np.zeros((pad, w, c)), axis=0)
    raise ValueError("axis must be 0 or 1")


def _append_gif_frame(sequence: list, frame_array: np.ndarray) -> None:
    im = Image.fromarray(frame_array.astype(np.uint8, copy=False))
    im.thumbnail(GIF_FRAME_MAX_SIZE, Image.Resampling.LANCZOS)
    sequence.append(im)

def main():
    p = Image.open(SAMPLE_IMAGE_PATH).convert("RGB")
    p.thumbnail(size=(800, 500))
    R, G, B = 0, 1, 2

    # Energy map (interior only; border is 1000 and would dominate normalization)
    image = np.array(p).astype(int)
    energy = compute_energy(image)[1:-1, 1:-1]
    energy = energy.astype(np.float64, copy=False)
    energy -= np.min(energy)
    energy = energy / np.max(energy)
    energy *= 256
    energy = np.floor(energy)
    energy[energy == 256] = 255
    Image.fromarray(energy.astype(np.uint8), mode="L").save("energy.png")

    image = np.array(p).astype(int)
    original_shape = image.shape
    sequence = []

    n_vertical = min(GIF_MAX_SEAMS, original_shape[1])
    for _ in range(n_vertical):
        vertical_indices = tuple(np.arange(image.shape[0]))
        horizontal_indices = tuple(find_vertical_seam(image))
        image[vertical_indices, horizontal_indices, R] = 255
        image[vertical_indices, horizontal_indices, G] = 0
        image[vertical_indices, horizontal_indices, B] = 0
        _append_gif_frame(sequence, _append_black_padding(image, original_shape, axis=1))
        image = remove_vertical_seam(image, np.asarray(horizontal_indices))
        _append_gif_frame(sequence, _append_black_padding(image, original_shape, axis=1))

    Image.fromarray(image.astype(np.uint8)).save("vertical_carving_final.png")
    p.save("vertical_carving.gif", save_all=True, append_images=sequence)

    image = np.array(p).astype(int)
    original_shape = image.shape
    sequence = []

    n_horizontal = min(GIF_MAX_SEAMS, original_shape[0])
    for _ in range(n_horizontal):
        vertical_indices = tuple(find_horizontal_seam(image))
        horizontal_indices = tuple(np.arange(image.shape[1]))
        image[vertical_indices, horizontal_indices, R] = 255
        image[vertical_indices, horizontal_indices, G] = 0
        image[vertical_indices, horizontal_indices, B] = 0

        _append_gif_frame(sequence, _append_black_padding(image, original_shape, axis=0))
        image = remove_horizontal_seam(image, np.asarray(vertical_indices))
        _append_gif_frame(sequence, _append_black_padding(image, original_shape, axis=0))
    Image.fromarray(image.astype(np.uint8)).save("horizontal_carving_final.png")
    p.save("horizontal_carving.gif", save_all=True, append_images=sequence)

if __name__ == "__main__":
    main()