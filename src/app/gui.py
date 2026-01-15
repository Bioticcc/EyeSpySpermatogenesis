from __future__ import annotations

import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageTk

from src.app.inference import (
    load_detector,
    phase2_load_stage_model,
    phase2_predict_stage,
    predict_image,
    save_results_csv,
    save_results_json,
)
from src.app.resources import bundled_detector_path, bundled_stage_model_path
from src.utils.logging import get_logger
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


class DetectorApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tubule Detector")
        self.geometry("1100x700")
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device("cpu")
        self.file_paths: List[Path] = []
        self.results: Dict[str, Dict] = {}
        self.current_index = 0
        self.display_scale = 1.0
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.canvas_image_id: Optional[int] = None

        self._build_ui()
        self._try_load_default_model()

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self)
        toolbar.pack(fill=tk.X, padx=6, pady=6)

        tk.Button(toolbar, text="Load Weights", command=self.load_weights).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Select TIFFs", command=self.select_files).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Prev", command=self.prev_image).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Export JSON", command=self.export_json).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Export CSV", command=self.export_csv).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var).pack(fill=tk.X, padx=6)

        content = tk.Frame(self)
        content.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(content, bg="black")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)

        panel = tk.Frame(content)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=6, pady=6)

        tk.Label(panel, text="Detections").pack(anchor="w")
        columns = ("idx", "x", "y", "w", "h", "score")
        self.tree = ttk.Treeview(panel, columns=columns, show="headings", height=18)
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="w", width=80)
        self.tree.pack(fill=tk.BOTH, expand=True)

    def _try_load_default_model(self) -> None:
        path = bundled_detector_path()
        if path.exists():
            try:
                self.model = load_detector(path, device=self.device)
                self.status_var.set(f"Loaded detector from {path}")
            except Exception as exc:
                logger.warning("Failed to load detector: %s", exc)
                self.status_var.set("Failed to load detector. Use 'Load Weights'.")
        else:
            self.status_var.set("No detector weights found. Train & export, then click 'Load Weights'.")

    def load_weights(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pt")])
        if not file_path:
            return
        try:
            self.model = load_detector(Path(file_path), device=self.device)
            self.status_var.set(f"Loaded detector from {file_path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load weights: {exc}")

    def select_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("TIFF files", "*.tif *.tiff")])
        if not paths:
            return
        if self.model is None:
            messagebox.showwarning("No model", "Detector weights not loaded. Load weights first.")
            return
        self.file_paths = [Path(p) for p in paths]
        self.results.clear()
        self.current_index = 0
        self.run_inference_all()
        self.show_current()

    def run_inference_all(self) -> None:
        if self.model is None:
            return
        for path in self.file_paths:
            result = predict_image(path, self.model, device=self.device)
            self.results[str(path)] = result
        self.status_var.set(f"Inference complete on {len(self.file_paths)} file(s).")

    def show_current(self) -> None:
        if not self.file_paths:
            return
        path = self.file_paths[self.current_index]
        result = self.results.get(str(path))
        if result is None:
            return
        array = read_tiff(path)
        image, scale = self._prepare_display_image(array, max_size=900)
        self.display_scale = scale
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.config(width=image.width, height=image.height)
        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=self.photo)
        self.canvas.delete("box")
        for det in result["detections"]:
            x, y, w, h = det["bbox_xywh"]
            x0 = x * scale
            y0 = y * scale
            x1 = (x + w) * scale
            y1 = (y + h) * scale
            self.canvas.create_rectangle(x0, y0, x1, y1, outline="lime", width=2, tags="box")
        self._populate_table(result["detections"])
        self.status_var.set(f"{path.name} ({self.current_index + 1}/{len(self.file_paths)})")

    def _populate_table(self, detections: List[Dict]) -> None:
        self.tree.delete(*self.tree.get_children())
        for idx, det in enumerate(detections):
            x, y, w, h = det["bbox_xywh"]
            score = det["score"]
            self.tree.insert("", tk.END, values=(idx, f"{x:.1f}", f"{y:.1f}", f"{w:.1f}", f"{h:.1f}", f"{score:.3f}"))

    @staticmethod
    def _prepare_display_image(array: np.ndarray, max_size: int) -> tuple[Image.Image, float]:
        image = Image.fromarray((array * 255).clip(0, 255).astype("uint8"))
        scale = 1.0
        if max(image.size) > max_size:
            scale = max_size / float(max(image.size))
            new_w = int(round(image.size[0] * scale))
            new_h = int(round(image.size[1] * scale))
            image = image.resize((new_w, new_h), Image.BILINEAR)
        return image, scale

    def next_image(self) -> None:
        if not self.file_paths:
            return
        if self.current_index + 1 >= len(self.file_paths):
            return
        self.current_index += 1
        self.show_current()

    def prev_image(self) -> None:
        if not self.file_paths:
            return
        if self.current_index == 0:
            return
        self.current_index -= 1
        self.show_current()

    def export_json(self) -> None:
        if not self.results:
            messagebox.showinfo("No results", "Run inference before exporting.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=[("JSON", "*.json")])
        if not save_path:
            return
        save_results_json(list(self.results.values()), Path(save_path))
        self.status_var.set(f"Saved results to {save_path}")
        messagebox.showinfo("Saved", f"Results saved to {save_path}")

    def export_csv(self) -> None:
        if not self.results:
            messagebox.showinfo("No results", "Run inference before exporting.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not save_path:
            return
        save_results_csv(list(self.results.values()), Path(save_path))
        self.status_var.set(f"Saved results to {save_path}")
        messagebox.showinfo("Saved", f"Results saved to {save_path}")


def main() -> None:
    app = DetectorApp()
    app.mainloop()


if __name__ == "__main__":
    main()


# Phase 2 / future: legacy stage classifier UI kept for reference.
class Phase2StageClassifierApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tubule Stage Classifier (Phase 2)")
        self.geometry("900x500")
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device("cpu")
        self.results: List = []

        self._build_ui()
        self._try_load_default_model()

    def _build_ui(self) -> None:
        toolbar = tk.Frame(self)
        toolbar.pack(fill=tk.X, padx=6, pady=6)

        tk.Button(toolbar, text="Load Weights", command=self.load_weights).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Select TIFFs", command=self.select_files).pack(side=tk.LEFT, padx=4)
        tk.Button(toolbar, text="Save CSV", command=self.save_csv).pack(side=tk.LEFT, padx=4)

        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self, textvariable=self.status_var).pack(fill=tk.X, padx=6)

        columns = ("file", "prediction", "confidence", "top3")
        self.tree = ttk.Treeview(self, columns=columns, show="headings")
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, anchor="w", width=200 if col == "top3" else 150)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

    def _try_load_default_model(self) -> None:
        path = bundled_stage_model_path()
        if path.exists():
            try:
                self.model = phase2_load_stage_model(path, device=self.device)
                self.status_var.set(f"Loaded model from {path}")
            except Exception as exc:
                logger.warning("Failed to load model: %s", exc)
                self.status_var.set("Failed to load model. Use 'Load Weights'.")
        else:
            self.status_var.set("No model found. Train & export, then click 'Load Weights'.")

    def load_weights(self) -> None:
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pt")])
        if not file_path:
            return
        try:
            self.model = phase2_load_stage_model(Path(file_path), device=self.device)
            self.status_var.set(f"Loaded model from {file_path}")
        except Exception as exc:
            messagebox.showerror("Error", f"Failed to load weights: {exc}")

    def select_files(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=[("TIFF files", "*.tif *.tiff")])
        if not paths:
            return
        if self.model is None:
            messagebox.showwarning("No model", "Model weights not loaded. Load weights first.")
            return
        self.tree.delete(*self.tree.get_children())
        self.results.clear()
        for path_str in paths:
            path = Path(path_str)
            try:
                result = phase2_predict_stage(path, self.model, device=self.device)
                top3_str = "; ".join([f"{label}:{prob:.2f}" for label, prob in result["top3"]])
                self.tree.insert("", tk.END, values=(path.name, result["predicted"], f"{result['confidence']:.3f}", top3_str))
                self.results.append((path.name, result["predicted"], result["confidence"], result["top3"]))
            except Exception as exc:
                messagebox.showerror("Error", f"Failed on {path.name}: {exc}")
        self.status_var.set(f"Inference complete on {len(paths)} file(s).")

    def save_csv(self) -> None:
        if not self.results:
            messagebox.showinfo("No results", "Run inference before saving.")
            return
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not save_path:
            return
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["file", "prediction", "confidence", "top3"])
            for file, pred, conf, top3 in self.results:
                top3_str = "; ".join([f"{label}:{prob:.3f}" for label, prob in top3])
                writer.writerow([file, pred, f"{conf:.3f}", top3_str])
        self.status_var.set(f"Saved results to {save_path}")
        messagebox.showinfo("Saved", f"Results saved to {save_path}")
