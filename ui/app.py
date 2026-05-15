"""Tkinter UI for seam carving (upload, carve in background, optional GIF export)."""

from __future__ import annotations

import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Literal

_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import numpy as np
from PIL import Image, ImageTk

from seam_carver import carve_horizontal_seams, carve_vertical_seams
from seam_carving_gifs import frames_for_horizontal_carving, frames_for_vertical_carving, save_carving_gif

Orientation = Literal["vertical", "horizontal"]


class SeamCarverApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Seam Carver")
        self.geometry("1100x720")

        self._src_array: np.ndarray | None = None
        self._result_array: np.ndarray | None = None
        self._gif_frames: list[Image.Image] | None = None
        self._photo_orig: ImageTk.PhotoImage | None = None
        self._photo_res: ImageTk.PhotoImage | None = None
        self._q: queue.Queue[tuple[str, Any]] = queue.Queue()
        self._running = False

        self._orient_var = tk.StringVar(value="vertical")
        self._seams_var = tk.IntVar(value=1)

        top = ttk.Frame(self, padding=8)
        top.pack(side=tk.TOP, fill=tk.X)

        self.btn_upload = ttk.Button(top, text="Upload", command=self.on_upload)
        self.btn_upload.pack(side=tk.LEFT, padx=4)

        ttk.Label(top, text="Direction").pack(side=tk.LEFT, padx=(8, 0))
        self.rb_v = ttk.Radiobutton(
            top, text="Vertical", variable=self._orient_var, value="vertical", command=self._on_orient_change
        )
        self.rb_v.pack(side=tk.LEFT, padx=2)
        self.rb_h = ttk.Radiobutton(
            top,
            text="Horizontal",
            variable=self._orient_var,
            value="horizontal",
            command=self._on_orient_change,
        )
        self.rb_h.pack(side=tk.LEFT, padx=2)

        ttk.Label(top, text="Seams").pack(side=tk.LEFT, padx=(12, 0))
        self.seam_scale = ttk.Scale(
            top, from_=0, to=1, orient=tk.HORIZONTAL, length=220, command=self._on_seam_scale
        )
        self.seam_scale.pack(side=tk.LEFT, padx=4)
        self.seam_label = ttk.Label(top, text="1")
        self.seam_label.pack(side=tk.LEFT)

        self.btn_run = ttk.Button(top, text="Run", command=self.on_run)
        self.btn_run.pack(side=tk.LEFT, padx=8)
        self.btn_save_png = ttk.Button(top, text="Save carved image", command=self.on_save_carved)
        self.btn_save_png.pack(side=tk.LEFT, padx=4)
        self.btn_save_gif = ttk.Button(top, text="Save GIF", command=self.on_save_gif, state=tk.DISABLED)
        self.btn_save_gif.pack(side=tk.LEFT, padx=4)
        self.btn_new = ttk.Button(top, text="New", command=self.on_new)
        self.btn_new.pack(side=tk.LEFT, padx=4)

        self.status = ttk.Label(top, text="Load an image to begin.")
        self.status.pack(side=tk.LEFT, padx=12)

        mid = ttk.Frame(self, padding=8)
        mid.pack(fill=tk.BOTH, expand=True)

        left = ttk.LabelFrame(mid, text="Original")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_orig = tk.Canvas(left, bg="#222", highlightthickness=0)
        self.canvas_orig.pack(fill=tk.BOTH, expand=True)

        right = ttk.LabelFrame(mid, text="Carved result")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_res = tk.Canvas(right, bg="#222", highlightthickness=0)
        self.canvas_res.pack(fill=tk.BOTH, expand=True)

        self.bind("<Configure>", self._on_resize)
        self.after(100, self._poll_queue)

    def _max_seams(self) -> int:
        if self._src_array is None:
            return 1
        h, w = self._src_array.shape[:2]
        if self._orient_var.get() == "vertical":
            return max(w - 1, 0)
        return max(h - 1, 0)

    def _on_orient_change(self) -> None:
        self._sync_seam_bounds()

    def _sync_seam_bounds(self) -> None:
        mx = self._max_seams()
        cur = int(self._seams_var.get())
        if mx <= 0:
            self._seams_var.set(0)
            self.seam_scale.configure(from_=0, to=1)
            self.seam_scale.set(0)
            self.seam_label.configure(text="0")
            return
        self.seam_scale.configure(from_=1, to=float(mx))
        clamped = max(1, min(cur, mx))
        self._seams_var.set(clamped)
        self.seam_scale.set(float(clamped))
        self.seam_label.configure(text=str(clamped))

    def _on_seam_scale(self, _evt: str | float | None = None) -> None:
        mx = self._max_seams()
        if mx <= 0:
            self._seams_var.set(0)
            self.seam_label.configure(text="0")
            return
        v = int(round(float(self.seam_scale.get())))
        v = max(1, min(v, mx))
        self._seams_var.set(v)
        self.seam_label.configure(text=str(v))

    def _set_running(self, running: bool) -> None:
        self._running = running
        state = tk.DISABLED if running else tk.NORMAL
        for w in (
            self.btn_upload,
            self.btn_run,
            self.btn_save_png,
            self.btn_new,
            self.rb_v,
            self.rb_h,
            self.seam_scale,
        ):
            w.configure(state=state)
        if running or not self._gif_frames:
            self.btn_save_gif.configure(state=tk.DISABLED)
        else:
            self.btn_save_gif.configure(state=tk.NORMAL)

    def _poll_queue(self) -> None:
        try:
            while True:
                kind, payload = self._q.get_nowait()
                if kind == "status":
                    self.status.configure(text=str(payload))
                elif kind == "result":
                    self._result_array = payload
                    self.status.configure(text="Done.")
                    self._set_running(False)
                    self._refresh_canvases()
                elif kind == "frames":
                    self._gif_frames = payload
                    if self._gif_frames:
                        self.btn_save_gif.configure(state=tk.NORMAL)
                elif kind == "error":
                    self._set_running(False)
                    messagebox.showerror("Error", str(payload))
                    self.status.configure(text="Error.")
        except queue.Empty:
            pass
        self.after(100, self._poll_queue)

    def on_upload(self) -> None:
        if self._running:
            return
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp;*.gif")])
        if not path:
            return
        im = Image.open(path).convert("RGB")
        self._src_array = np.array(im, dtype=np.uint8)
        self._result_array = None
        self._gif_frames = None
        self.btn_save_gif.configure(state=tk.DISABLED)
        h, w = self._src_array.shape[:2]
        self._sync_seam_bounds()
        self.status.configure(text=f"Loaded {w}x{h}")
        self._refresh_canvases()

    def on_new(self) -> None:
        if self._running:
            return
        self._src_array = None
        self._result_array = None
        self._gif_frames = None
        self.btn_save_gif.configure(state=tk.DISABLED)
        self.status.configure(text="Cleared.")
        self._refresh_canvases()

    def on_save_carved(self) -> None:
        if self._running:
            return
        if self._result_array is None:
            messagebox.showinfo("Save", "Run seam carving first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg;*.jpeg")],
        )
        if not path:
            return
        Image.fromarray(self._result_array, mode="RGB").save(path)
        self.status.configure(text=f"Saved {path}")

    def on_save_gif(self) -> None:
        if self._running:
            return
        if not self._gif_frames:
            messagebox.showinfo("Save GIF", "No GIF frames available. Run carving first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF", "*.gif")])
        if not path:
            return
        try:
            if self._src_array is None:
                raise RuntimeError("Missing source image for GIF header frame.")
            first = Image.fromarray(self._src_array, mode="RGB")
            save_carving_gif(self._gif_frames, path, first_frame=first)
            self.status.configure(text=f"Saved GIF {path}")
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("GIF save failed", str(exc))

    def on_run(self) -> None:
        if self._src_array is None:
            messagebox.showinfo("Run", "Upload an image first.")
            return
        if self._running:
            return
        src = self._src_array.copy()
        n_seams = int(self._seams_var.get())
        orient: Orientation = "vertical" if self._orient_var.get() == "vertical" else "horizontal"

        def worker() -> None:
            try:
                self._q.put(("status", "Running…"))
                work_int = src.astype(np.int64)
                if orient == "vertical":
                    frames = frames_for_vertical_carving(work_int.copy(), n_seams)
                    out = carve_vertical_seams(src, n_seams)
                else:
                    frames = frames_for_horizontal_carving(work_int.copy(), n_seams)
                    out = carve_horizontal_seams(src, n_seams)
                self._q.put(("frames", frames))
                self._q.put(("result", out))
            except Exception as exc:  # pragma: no cover
                self._q.put(("error", str(exc)))

        self._result_array = None
        self._gif_frames = None
        self.btn_save_gif.configure(state=tk.DISABLED)
        self._refresh_canvases()
        self._set_running(True)
        self.status.configure(text="Running in background…")
        threading.Thread(target=worker, daemon=True).start()

    def _on_resize(self, _evt: tk.Event[tk.Misc] | None = None) -> None:
        self._refresh_canvases()

    def _fit_pil(self, arr: np.ndarray, max_w: int, max_h: int) -> Image.Image:
        im = Image.fromarray(arr, mode="RGB")
        im.thumbnail((max_w, max_h), Image.Resampling.LANCZOS)
        return im

    def _refresh_canvases(self) -> None:
        cw = max(self.canvas_orig.winfo_width(), 2)
        ch = max(self.canvas_orig.winfo_height(), 2)

        if self._src_array is not None:
            thumb = self._fit_pil(self._src_array, cw, ch)
            self._photo_orig = ImageTk.PhotoImage(thumb)
            self.canvas_orig.delete("all")
            self.canvas_orig.create_image(cw // 2, ch // 2, image=self._photo_orig)
        else:
            self.canvas_orig.delete("all")

        self.canvas_res.delete("all")
        if self._result_array is not None:
            thumb = self._fit_pil(self._result_array, cw, ch)
            self._photo_res = ImageTk.PhotoImage(thumb)
            self.canvas_res.create_image(cw // 2, ch // 2, image=self._photo_res)


def main() -> None:
    app = SeamCarverApp()
    app.mainloop()


if __name__ == "__main__":
    main()
