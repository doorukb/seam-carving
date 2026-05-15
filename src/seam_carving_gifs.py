"""Energy export and seam-carving GIF / frame generation (Pillow I/O)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from seam_carver import compute_energy, find_horizontal_seam, find_vertical_seam


def energy_grayscale_uint8(image_rgb: np.ndarray) -> np.ndarray:
    """Cropped interior energy map normalized to ``uint8`` grayscale (H', W')."""
    image = np.asarray(image_rgb).astype(int)
    energy = compute_energy(image)[1:-1, 1:-1]
    energy = energy - np.min(energy)
    denom = float(np.max(energy)) if float(np.max(energy)) > 0 else 1.0
    energy = energy / denom
    energy *= 256
    energy = np.floor(energy)
    energy[energy == 256] = 255
    return energy.astype(np.uint8)


def save_energy_png(image_rgb: np.ndarray, path: str | Path) -> None:
    arr = energy_grayscale_uint8(image_rgb)
    Image.fromarray(arr, mode="L").save(path)


def frames_for_vertical_carving(image: np.ndarray, n_seams: int) -> list[Image.Image]:
    """Build padded RGB frames: seam highlighted in red, then carved (two frames per seam)."""
    r, g, b = 0, 1, 2
    image = np.asarray(image).astype(int)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB (H, W, 3).")
    original_shape = image.shape
    n = min(int(n_seams), max(image.shape[1] - 1, 0))
    sequence: list[Image.Image] = []
    for _ in range(n):
        vertical_indices = tuple(np.arange(image.shape[0]))
        horizontal_indices = tuple(find_vertical_seam(image))
        image[vertical_indices, horizontal_indices, r] = 255
        image[vertical_indices, horizontal_indices, g] = 0
        image[vertical_indices, horizontal_indices, b] = 0

        sequence.append(
            Image.fromarray(
                np.append(
                    image,
                    np.zeros(
                        (
                            original_shape[0],
                            original_shape[1] - image.shape[1],
                            original_shape[2],
                        )
                    ),
                    axis=1,
                ).astype(np.uint8)
            )
        )

        mask = np.full(image.shape, True, dtype=bool)
        mask[vertical_indices, horizontal_indices] = False
        image = image[mask].reshape((image.shape[0], image.shape[1] - 1, image.shape[2]))

        sequence.append(
            Image.fromarray(
                np.append(
                    image,
                    np.zeros(
                        (
                            original_shape[0],
                            original_shape[1] - image.shape[1],
                            original_shape[2],
                        )
                    ),
                    axis=1,
                ).astype(np.uint8)
            )
        )
    return sequence


def frames_for_horizontal_carving(image: np.ndarray, n_seams: int) -> list[Image.Image]:
    """Build padded RGB frames for horizontal carving (two frames per seam)."""
    r, g, b = 0, 1, 2
    image = np.asarray(image).astype(int)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected RGB (H, W, 3).")
    original_shape = image.shape
    n = min(int(n_seams), max(image.shape[0] - 1, 0))
    sequence: list[Image.Image] = []
    for _ in range(n):
        vertical_indices = tuple(find_horizontal_seam(image))
        horizontal_indices = tuple(np.arange(image.shape[1]))
        image[vertical_indices, horizontal_indices, r] = 255
        image[vertical_indices, horizontal_indices, g] = 0
        image[vertical_indices, horizontal_indices, b] = 0

        sequence.append(
            Image.fromarray(
                np.append(
                    image,
                    np.zeros(
                        (
                            original_shape[0] - image.shape[0],
                            original_shape[1],
                            original_shape[2],
                        )
                    ),
                    axis=0,
                ).astype(np.uint8)
            )
        )

        mask = np.full(image.shape, True, dtype=bool)
        mask[vertical_indices, horizontal_indices] = False
        image = (
            image.transpose(1, 0, 2)[mask.transpose(1, 0, 2)]
            .reshape((image.shape[1], image.shape[0] - 1, image.shape[2]))
            .transpose(1, 0, 2)
        )

        sequence.append(
            Image.fromarray(
                np.append(
                    image,
                    np.zeros(
                        (
                            original_shape[0] - image.shape[0],
                            original_shape[1],
                            original_shape[2],
                        )
                    ),
                    axis=0,
                ).astype(np.uint8)
            )
        )
    return sequence


def save_carving_gif(
    frames: Iterable[Image.Image],
    path: str | Path,
    *,
    duration_ms: int | None = None,
    loop: int | None = None,
    first_frame: Image.Image | None = None,
) -> None:
    """Save a GIF from frames. Optionally prepend ``first_frame`` (e.g. original)."""
    frame_list = list(frames)
    if not frame_list and first_frame is None:
        raise ValueError("No frames to save.")
    imgs = [first_frame, *frame_list] if first_frame is not None else frame_list
    base = imgs[0]
    rest = imgs[1:]
    kwargs: dict[str, object] = {"save_all": True, "append_images": rest}
    if duration_ms is not None:
        kwargs["duration"] = duration_ms
    if loop is not None:
        kwargs["loop"] = loop
    base.save(path, format="GIF", **kwargs)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export energy PNG and seam-carving GIFs.")
    p.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Input image path (default: seam_carving_demo.jpg in cwd if present).",
    )
    p.add_argument("--seams", type=int, default=50, help="Number of seams to carve per axis.")
    p.add_argument("--out-dir", type=Path, default=Path("."), help="Output directory.")
    p.add_argument("--energy-png", type=Path, default=None, help="Energy map output path.")
    p.add_argument("--vertical-gif", type=Path, default=None)
    p.add_argument("--horizontal-gif", type=Path, default=None)
    p.add_argument("--vertical-final-png", type=Path, default=None)
    p.add_argument("--horizontal-final-png", type=Path, default=None)
    p.add_argument("--gif-duration-ms", type=int, default=None)
    p.add_argument("--gif-loop", type=int, default=None)
    p.add_argument("--thumb-max", type=int, nargs=2, default=(800, 500), metavar=("W", "H"))
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    img_path = args.image
    if img_path is None:
        candidate = Path("seam_carving_demo.jpg")
        img_path = candidate if candidate.is_file() else None
    if img_path is None or not img_path.is_file():
        raise SystemExit("Provide --image or place seam_carving_demo.jpg in the working directory.")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    pil = Image.open(img_path).convert("RGB")
    pil.thumbnail((int(args.thumb_max[0]), int(args.thumb_max[1])))
    rgb = np.array(pil).astype(int)
    stem = img_path.stem

    energy_path = args.energy_png or (out_dir / f"{stem}_energy.png")
    save_energy_png(rgb, energy_path)

    n = int(args.seams)
    v_frames = frames_for_vertical_carving(rgb.copy(), n)
    h_frames = frames_for_horizontal_carving(rgb.copy(), n)

    v_gif = args.vertical_gif or (out_dir / f"{stem}_vertical_carving.gif")
    h_gif = args.horizontal_gif or (out_dir / f"{stem}_horizontal_carving.gif")
    save_carving_gif(
        v_frames,
        v_gif,
        duration_ms=args.gif_duration_ms,
        loop=args.gif_loop,
        first_frame=pil,
    )
    save_carving_gif(
        h_frames,
        h_gif,
        duration_ms=args.gif_duration_ms,
        loop=args.gif_loop,
        first_frame=pil,
    )

    from seam_carver import carve_horizontal_seams, carve_vertical_seams

    v_final = args.vertical_final_png or (out_dir / f"{stem}_vertical_carving_final.png")
    h_final = args.horizontal_final_png or (out_dir / f"{stem}_horizontal_carving_final.png")
    Image.fromarray(carve_vertical_seams(np.array(pil), n), mode="RGB").save(v_final)
    Image.fromarray(carve_horizontal_seams(np.array(pil), n), mode="RGB").save(h_final)


if __name__ == "__main__":
    main()
