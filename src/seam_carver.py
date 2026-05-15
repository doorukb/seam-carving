import numpy as np
from PIL import Image

SAMPLE_IMAGE_PATH = "your_image.jpg"

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


def find_vertical_seam(image: np.ndarray, energy=None):
    if energy is None:
        energy = compute_energy(image)

        random_state = np.random.get_state()
        np.random.seed(0)
        noise = np.random.randn(*energy.shape).astype(np.float32)
        noise /= np.float32(1000.0 * (image.size ** 0.5))
        energy = energy + noise
        np.random.set_state(random_state)

    return find_vertical_seam_from_energy(energy)


def find_horizontal_seam(image: np.ndarray, energy=None):
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


def main():
    # Generate a visualization of the energy and 2 visualizations of the seam carving algorithm.
    # open image with pillow, an active fork of the defunct PIL library
    p = Image.open(SAMPLE_IMAGE_PATH)
    # image could be other modes like RGBA, YCbCr, or L or something
    p = p.convert(mode="RGB")
    # limit max size
    p.thumbnail(size=(800, 500))
    R, G, B = 0, 1, 2

    # -------------------- energy -------------------------------

    # convert image to an array. Since default datatype for an RGB
    # image is in unsigned 8 bit integer, convert it to a regular int
    # to avoid hard-to-debug shenanigans like over/underflow
    image = np.array(p).astype(int)
    # compute the energy. Since the boundary is 1000 and would reduce
    # the visibility of the more interesting parts after normalization,
    # crop the "frame" to improve visualization
    energy = compute_energy(image)[1:-1, 1:-1]
    # uncomment to see the log-adjusted intensity.
    # energy = np.log(energy + 1)

    # normalization for visualization
    # darken the least value to black
    energy = energy - np.min(energy)
    # lighten the greatest value to white
    energy = energy / np.max(energy)
    # fit the value between [0, 255]
    energy *= 256
    energy = np.floor(energy)
    energy[energy == 256] = 255

    # convert values to an image
    energy_visualization = Image.fromarray(energy.astype(np.uint8), mode="L")
    energy_visualization.save("energy.png")

    # -------------------- vertical carving ---------------------

    # for an image of shape (height, width, channel)
    # build a visualization by gradually carving axis 1
    image = np.array(p).astype(int)
    original_shape = image.shape

    # sequence of frames to be animated
    sequence = []

    # cap number of seams to carve at 200
    for _ in range(min([200, original_shape[1]])):
        # Create a frame for the seam to be carved away in red
        vertical_indices = tuple(np.arange(image.shape[0]))
        horizontal_indices = tuple(find_vertical_seam(image))
        image[vertical_indices, horizontal_indices, R] = 255
        image[vertical_indices, horizontal_indices, G] = 0
        image[vertical_indices, horizontal_indices, B] = 0

        # append black pixels to make up for pixels carved away
        sequence.append(Image.fromarray(
            np.append(image, np.zeros((
                original_shape[0],
                original_shape[1] - image.shape[1],
                original_shape[2],
            )), axis=1).astype(np.uint8)
        ))

        image = remove_vertical_seam(image, np.asarray(horizontal_indices))

        # append black pixels to make up for pixels carved away
        sequence.append(Image.fromarray(
            np.append(image, np.zeros((
                original_shape[0],
                original_shape[1] - image.shape[1],
                original_shape[2],
            )), axis=1).astype(np.uint8)
        ))

    # save the final, carved image
    final_image = Image.fromarray(image.astype(np.uint8))
    final_image.save("vertical_carving_final.png")

    # build GIF
    p.save(
        "vertical_carving.gif",
        save_all=True,
        append_images=sequence,
        # uncomment this line to create infinite looping GIF
        # loop=0,
        # uncomment this line to control the speed of GIF
        # duration=40,
    )

    # -------------------- horizontal carving -------------------

    # for an image of shape (height, width, channel)
    # build a visualization by gradually carving axis 0
    image = np.array(p).astype(int)
    original_shape = image.shape

    # sequence of frames to be animated
    sequence = []

    # cap number of seams to carve at 200
    for _ in range(min([200, original_shape[0]])):
        # Create a frame for the seam to be carved away in red
        vertical_indices = tuple(find_horizontal_seam(image))
        horizontal_indices = tuple(np.arange(image.shape[1]))
        image[vertical_indices, horizontal_indices, R] = 255
        image[vertical_indices, horizontal_indices, G] = 0
        image[vertical_indices, horizontal_indices, B] = 0

        # append black pixels to make up for pixels carved away
        sequence.append(Image.fromarray(
            np.append(image, np.zeros((
                original_shape[0] - image.shape[0],
                original_shape[1],
                original_shape[2],
            )), axis=0).astype(np.uint8)
        ))

        image = remove_horizontal_seam(image, np.asarray(vertical_indices))

        # append black pixels to make up for pixels carved away
        sequence.append(Image.fromarray(
            np.append(image, np.zeros((
                original_shape[0] - image.shape[0],
                original_shape[1],
                original_shape[2],
            )), axis=0).astype(np.uint8)
        ))

    # save the final, carved image
    final_image = Image.fromarray(image.astype(np.uint8))
    final_image.save("horizontal_carving_final.png")

    # build GIF
    p.save(
        "horizontal_carving.gif",
        save_all=True,
        append_images=sequence,
        # uncomment this line to create infinite looping GIF
        # loop=0,
        # uncomment this line to control the speed of GIF
        # duration=40,
    )

if __name__ == "__main__":
    main()
