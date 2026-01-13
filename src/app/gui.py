from __future__ import annotations

import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
from typing import List, Optional, Tuple

import torch

from src.app.inference import load_model, predict_image
from src.app.resources import bundled_model_path
from src.utils.logging import get_logger

logger = get_logger(__name__)


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Tubule Stage Classifier")
        self.geometry("900x500")
        self.model: Optional[torch.nn.Module] = None
        self.device = torch.device("cpu")
        self.results: List[Tuple[str, str, float, List[Tuple[str, float]]]] = []

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
        path = bundled_model_path()
        if path.exists():
            try:
                self.model = load_model(path, device=self.device)
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
            self.model = load_model(Path(file_path), device=self.device)
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
                result = predict_image(path, self.model, device=self.device)
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


def main() -> None:
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
