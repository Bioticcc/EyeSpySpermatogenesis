from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import tkinter as tk
import tkinter.messagebox as messagebox
import tkinter.ttk as ttk
from PIL import Image, ImageTk

from src.dataset.annotations import (
    BoxAnnotation,
    ImageAnnotation,
    PolygonAnnotation,
    TubulePolygon,
    annotation_path,
    load_annotation,
    load_polygon_annotation,
    new_box_id,
    save_annotation,
    save_polygon_annotation,
)
from src.dataset.indexing import load_labels, save_labels, update_index
from src.dataset.schema import IncludeFlag, LabelRecord, NegativeTrainingFlag, Orientation, Quality, now_iso
from src.utils.logging import get_logger
from src.utils.paths import annotations_dir, labels_dir, raw_data_dir
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


def filter_records(
    records: Dict[str, LabelRecord],
    annotations_path: Path,
    include_filter: Optional[IncludeFlag],
    include_yes_only: bool,
    only_unlabeled: bool,
) -> List[LabelRecord]:
    filtered: List[LabelRecord] = []
    for record in records.values():
        if include_yes_only and record.include != IncludeFlag.YES:
            continue
        if include_filter is not None and record.include != include_filter:
            continue
        if include_filter is None and record.include == IncludeFlag.NO:
            continue
        if only_unlabeled and annotation_path(annotations_path, record.image_id).exists():
            continue
        filtered.append(record)
    filtered.sort(key=lambda r: r.rel_path)
    return filtered


def _auto_contrast(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        base = array
    else:
        base = array.mean(axis=2)
    if base.size == 0:
        return array
    p_low, p_high = np.percentile(base, (2.0, 98.0))
    if p_high <= p_low:
        p_low, p_high = float(base.min()), float(base.max())
    if p_high <= p_low:
        return array
    scaled = (array - p_low) / (p_high - p_low)
    return np.clip(scaled, 0.0, 1.0)


def build_preview_image(array: np.ndarray, max_preview: int) -> Tuple[Image.Image, float]:
    scaled = _auto_contrast(array)
    preview = (scaled * 255.0).clip(0, 255).astype("uint8")
    image = Image.fromarray(preview)
    scale = 1.0
    if max(image.size) > max_preview:
        scale = max_preview / float(max(image.size))
        new_w = int(round(image.size[0] * scale))
        new_h = int(round(image.size[1] * scale))
        image = image.resize((new_w, new_h), Image.BILINEAR)
    return image, scale


def point_in_polygon(x: float, y: float, polygon: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi)
        if intersects:
            inside = not inside
        j = i
    return inside


def validate_polygon(points: List[Tuple[float, float]], width: int, height: int) -> Tuple[bool, str]:
    if len(points) < 3:
        return False, "Polygon needs at least 3 points."
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if not np.isfinite(xs).all() or not np.isfinite(ys).all():
        return False, "Polygon has invalid coordinates."
    if max(xs) < 0 or max(ys) < 0:
        return False, "Polygon is outside the image."
    if min(xs) >= width or min(ys) >= height:
        return False, "Polygon is outside the image."
    if (max(xs) - min(xs)) < 1.0 or (max(ys) - min(ys)) < 1.0:
        return False, "Polygon is too small."
    return True, ""


class PolygonAnnotationTool:
    def __init__(
        self,
        raw_dir: Path,
        labels_path: Path,
        annotations_path: Path,
        records: List[LabelRecord],
        include_filter: Optional[IncludeFlag] = None,
        include_yes_only: bool = False,
        only_unlabeled: bool = False,
        start_image_id: str = "",
        labeled_by: str = "",
        max_preview: int = 1024,
    ) -> None:
        self.raw_dir = raw_dir
        self.labels_path = labels_path
        self.annotations_path = annotations_path
        self.all_records = list(records)
        self.labels = load_labels(labels_path)
        self.include_filter = include_filter
        self.include_yes_only = include_yes_only
        self.only_unlabeled = only_unlabeled
        self.start_image_id = start_image_id
        self.max_preview = max_preview
        self.index = 0
        self.display_scale = 1.0
        self.current_tubules: List[TubulePolygon] = []
        self.current_points: List[Tuple[float, float]] = []
        self.drawing_mode = False
        self.selected_tubule_idx: Optional[int] = None
        self.selected_vertex_idx: Optional[int] = None
        self.dragging_vertex = False
        self.last_cursor_pos: Optional[Tuple[int, int]] = None
        self.labeled_by_default = labeled_by
        self.current_image_size: Tuple[int, int] = (0, 0)
        self.active_tab = "included"
        self.tab_positions: Dict[str, Optional[str]] = {
            "included": start_image_id or None,
            "excluded": None,
        }
        self.records: List[LabelRecord] = []

        self.root = tk.Tk()
        self.root.title("Tubule Polygon Annotation Tool")

        self.tab_control = ttk.Notebook(self.root)
        self.tab_included = ttk.Frame(self.tab_control)
        self.tab_excluded = ttk.Frame(self.tab_control)
        self.tab_control.add(self.tab_included, text="Include: yes/unknown")
        self.tab_control.add(self.tab_excluded, text="Include: no")
        self.tab_control.grid(row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 0))
        self.tab_control.bind("<<NotebookTabChanged>>", self.on_tab_change)

        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)

        sidebar = tk.Frame(self.root)
        sidebar.grid(row=1, column=1, sticky="n", padx=6, pady=6)

        tk.Button(sidebar, text="Prev", command=self.prev_record).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Next", command=self.next_record).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Save", command=self.save_current).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Save & Next", command=self.save_and_next).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="New tubule", command=self.start_new_tubule).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Delete selected", command=self.delete_selected).pack(fill=tk.X, pady=2)

        tk.Label(sidebar, text="Quality").pack(anchor="w", pady=(8, 0))
        self.quality_var = tk.StringVar(value=Quality.UNKNOWN.value)
        tk.OptionMenu(sidebar, self.quality_var, *[q.value for q in Quality]).pack(fill=tk.X)

        tk.Label(sidebar, text="Include").pack(anchor="w", pady=(8, 0))
        self.include_var = tk.StringVar(value=IncludeFlag.UNKNOWN.value)
        tk.OptionMenu(sidebar, self.include_var, *[i.value for i in IncludeFlag]).pack(fill=tk.X)

        tk.Label(sidebar, text="Negative training").pack(anchor="w", pady=(8, 0))
        self.negative_training_var = tk.StringVar(value=NegativeTrainingFlag.NO.value)
        tk.OptionMenu(sidebar, self.negative_training_var, *[n.value for n in NegativeTrainingFlag]).pack(fill=tk.X)

        tk.Label(sidebar, text="Orientation (selected tubule)").pack(anchor="w", pady=(8, 0))
        self.orientation_var = tk.StringVar(value=Orientation.UNKNOWN.value)
        tk.OptionMenu(
            sidebar,
            self.orientation_var,
            *[o.value for o in Orientation],
            command=self.on_orientation_change,
        ).pack(fill=tk.X)

        tk.Label(sidebar, text="Notes").pack(anchor="w", pady=(8, 0))
        self.notes_entry = tk.Text(sidebar, height=4, width=28)
        self.notes_entry.pack(fill=tk.X)

        tk.Label(sidebar, text="Labeled by").pack(anchor="w", pady=(8, 0))
        self.labeled_by_entry = tk.Entry(sidebar)
        self.labeled_by_entry.pack(fill=tk.X)
        if self.labeled_by_default:
            self.labeled_by_entry.insert(0, self.labeled_by_default)

        tk.Label(sidebar, text="Tubules").pack(anchor="w", pady=(8, 0))
        self.tubule_list = tk.Listbox(sidebar, height=8)
        self.tubule_list.pack(fill=tk.BOTH, expand=False)
        self.tubule_list.bind("<<ListboxSelect>>", self.on_tubule_select)

        self.status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status_var).grid(row=2, column=0, columnspan=2, sticky="ew", padx=6)

        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self.on_left_click)
        self.canvas.bind("<B1-Motion>", self.on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_left_release)
        self.canvas.bind("<Button-3>", self.on_right_click)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.root.bind("<Return>", lambda _event: self.finish_polygon())
        self.root.bind("<BackSpace>", lambda _event: self.remove_last_point())
        self.root.bind("<Delete>", lambda _event: self.delete_selected())
        self.root.bind("i", lambda _event: self.cycle_include())
        self.root.bind("q", lambda _event: self.cycle_quality())
        self.root.bind("o", lambda _event: self.cycle_orientation())
        self.root.bind("O", lambda _event: self.cycle_orientation())

        self.photo: Optional[ImageTk.PhotoImage] = None
        self.image_id_on_canvas: Optional[int] = None

        self.refresh_records()

    def start_new_tubule(self) -> None:
        self.drawing_mode = True
        self.current_points = []
        self.selected_tubule_idx = None
        self.selected_vertex_idx = None
        self.orientation_var.set(Orientation.UNKNOWN.value)
        self.status_var.set("Drawing mode: left-click to add vertices, right-click or Enter to finish.")
        self.redraw_overlays()

    def remove_last_point(self) -> None:
        if not self.drawing_mode:
            return
        if self.current_points:
            self.current_points.pop()
        self.redraw_overlays()

    def on_left_click(self, event: tk.Event) -> None:
        if self.photo is None:
            return
        if self.drawing_mode:
            self.current_points.append(self.display_to_original(event.x, event.y))
            self.redraw_overlays()
            return
        self.select_nearest(event.x, event.y)

    def on_left_drag(self, event: tk.Event) -> None:
        if not self.dragging_vertex or self.selected_tubule_idx is None or self.selected_vertex_idx is None:
            return
        ox, oy = self.display_to_original(event.x, event.y)
        tubule = self.current_tubules[self.selected_tubule_idx]
        points = list(tubule.polygon_xy)
        points[self.selected_vertex_idx] = [ox, oy]
        self.current_tubules[self.selected_tubule_idx] = TubulePolygon(
            tubule_id=tubule.tubule_id,
            polygon_xy=points,
            orientation=tubule.orientation,
            notes=tubule.notes,
        )
        self.redraw_overlays()

    def on_left_release(self, _event: tk.Event) -> None:
        self.dragging_vertex = False

    def on_right_click(self, _event: tk.Event) -> None:
        self.finish_polygon()

    def on_mouse_move(self, event: tk.Event) -> None:
        self.last_cursor_pos = (event.x, event.y)
        if self.drawing_mode:
            self.redraw_overlays()

    def finish_polygon(self) -> None:
        if not self.drawing_mode:
            return
        if not self.current_points:
            self.drawing_mode = False
            self.redraw_overlays()
            return
        width, height = self.current_image_size
        valid, message = validate_polygon(self.current_points, width, height)
        if not valid:
            messagebox.showwarning("Invalid polygon", message)
            return
        tubule_id = self.next_tubule_id()
        polygon_xy = [[float(x), float(y)] for x, y in self.current_points]
        self.current_tubules.append(
            TubulePolygon(
                tubule_id=tubule_id,
                polygon_xy=polygon_xy,
                orientation=Orientation.UNKNOWN,
                notes="",
            )
        )
        self.drawing_mode = False
        self.current_points = []
        self.refresh_tubule_list()
        self.select_tubule(len(self.current_tubules) - 1)
        self.redraw_overlays()

    def select_nearest(self, x: float, y: float) -> None:
        if not self.current_tubules:
            self.selected_tubule_idx = None
            self.selected_vertex_idx = None
            self.orientation_var.set(Orientation.UNKNOWN.value)
            return
        threshold = 8.0
        best = None
        best_dist = threshold
        for idx, tubule in enumerate(self.current_tubules):
            points = [self.original_to_display(px, py) for px, py in tubule.polygon_xy]
            for vidx, (vx, vy) in enumerate(points):
                dist = ((vx - x) ** 2 + (vy - y) ** 2) ** 0.5
                if dist <= best_dist:
                    best = (idx, vidx)
                    best_dist = dist
        if best is not None:
            self.select_tubule(best[0])
            self.selected_vertex_idx = best[1]
            self.dragging_vertex = True
            return

        for idx, tubule in enumerate(self.current_tubules):
            points = [self.original_to_display(px, py) for px, py in tubule.polygon_xy]
            if point_in_polygon(x, y, points):
                self.select_tubule(idx)
                return
        self.selected_tubule_idx = None
        self.selected_vertex_idx = None
        self.orientation_var.set(Orientation.UNKNOWN.value)
        self.redraw_overlays()

    def delete_selected(self) -> None:
        if self.drawing_mode:
            self.current_points = []
            self.drawing_mode = False
            self.redraw_overlays()
            return
        if self.selected_tubule_idx is None:
            return
        self.current_tubules.pop(self.selected_tubule_idx)
        self.selected_tubule_idx = None
        self.selected_vertex_idx = None
        self.orientation_var.set(Orientation.UNKNOWN.value)
        self.refresh_tubule_list()
        self.redraw_overlays()

    def on_tubule_select(self, _event: tk.Event) -> None:
        selection = self.tubule_list.curselection()
        if not selection:
            return
        self.select_tubule(selection[0])

    def select_tubule(self, idx: Optional[int]) -> None:
        self.selected_tubule_idx = idx
        self.selected_vertex_idx = None
        self.refresh_tubule_list()
        if idx is not None:
            self.tubule_list.selection_clear(0, tk.END)
            self.tubule_list.selection_set(idx)
            self.tubule_list.activate(idx)
            self.orientation_var.set(self.current_tubules[idx].orientation.value)
        else:
            self.orientation_var.set(Orientation.UNKNOWN.value)
        self.redraw_overlays()

    def refresh_tubule_list(self) -> None:
        self.tubule_list.delete(0, tk.END)
        for idx, tubule in enumerate(self.current_tubules):
            self.tubule_list.insert(tk.END, f"{idx}: {len(tubule.polygon_xy)} pts | {tubule.orientation.value}")

    def redraw_overlays(self) -> None:
        self.canvas.delete("overlay")
        for idx, tubule in enumerate(self.current_tubules):
            points = [self.original_to_display(px, py) for px, py in tubule.polygon_xy]
            if len(points) < 2:
                continue
            outline = "red" if idx == self.selected_tubule_idx else "lime"
            flat = [coord for pt in points for coord in pt]
            self.canvas.create_polygon(*flat, outline=outline, fill="", width=2, tags="overlay")
            for vx, vy in points:
                self.canvas.create_oval(vx - 3, vy - 3, vx + 3, vy + 3, outline=outline, fill=outline, tags="overlay")
            label_text = f"{idx}: {tubule.orientation.value}"
            self.canvas.create_text(
                points[0][0] + 6,
                points[0][1] + 6,
                text=label_text,
                fill="yellow",
                tags="overlay",
            )

        if self.drawing_mode and self.current_points:
            draw_points = [self.original_to_display(px, py) for px, py in self.current_points]
            if self.last_cursor_pos is not None:
                draw_points.append(self.last_cursor_pos)
            flat = [coord for pt in draw_points for coord in pt]
            if len(draw_points) >= 2:
                self.canvas.create_line(*flat, fill="yellow", width=2, tags="overlay")
            for vx, vy in draw_points[:-1]:
                self.canvas.create_oval(vx - 3, vy - 3, vx + 3, vy + 3, outline="yellow", fill="yellow", tags="overlay")

    def original_to_display(self, x: float, y: float) -> Tuple[float, float]:
        return x * self.display_scale, y * self.display_scale

    def display_to_original(self, x: float, y: float) -> Tuple[float, float]:
        inv = 1.0 / self.display_scale if self.display_scale > 0 else 1.0
        ox = max(0.0, x * inv)
        oy = max(0.0, y * inv)
        width, height = self.current_image_size
        ox = min(ox, max(0.0, width - 1.0))
        oy = min(oy, max(0.0, height - 1.0))
        return ox, oy

    def cycle_include(self) -> None:
        values = [i.value for i in IncludeFlag]
        current = self.include_var.get()
        if current not in values:
            self.include_var.set(IncludeFlag.UNKNOWN.value)
            return
        idx = (values.index(current) + 1) % len(values)
        self.include_var.set(values[idx])

    def cycle_quality(self) -> None:
        values = [q.value for q in Quality]
        current = self.quality_var.get()
        if current not in values:
            self.quality_var.set(Quality.UNKNOWN.value)
            return
        idx = (values.index(current) + 1) % len(values)
        self.quality_var.set(values[idx])

    def on_orientation_change(self, value: str) -> None:
        self.set_selected_orientation(value)

    def set_selected_orientation(self, value: str) -> None:
        if self.selected_tubule_idx is None:
            return
        try:
            orientation = Orientation(value)
        except Exception:
            orientation = Orientation.UNKNOWN
        self.orientation_var.set(orientation.value)
        tubule = self.current_tubules[self.selected_tubule_idx]
        self.current_tubules[self.selected_tubule_idx] = TubulePolygon(
            tubule_id=tubule.tubule_id,
            polygon_xy=list(tubule.polygon_xy),
            orientation=orientation,
            notes=tubule.notes,
        )
        self.refresh_tubule_list()
        self.redraw_overlays()

    def cycle_orientation(self) -> None:
        if self.selected_tubule_idx is None:
            return
        values = [o.value for o in Orientation]
        current = self.current_tubules[self.selected_tubule_idx].orientation.value
        if current not in values:
            current = Orientation.UNKNOWN.value
        idx = (values.index(current) + 1) % len(values)
        self.set_selected_orientation(values[idx])

    def next_tubule_id(self) -> str:
        record = self.records[self.index]
        prefix = f"{record.image_id}__"
        indices = []
        for tubule in self.current_tubules:
            if tubule.tubule_id.startswith(prefix):
                suffix = tubule.tubule_id[len(prefix) :]
                if suffix.isdigit():
                    indices.append(int(suffix))
        next_idx = max(indices) + 1 if indices else len(self.current_tubules)
        return f"{record.image_id}__{next_idx}"

    def load_record(self, record: LabelRecord) -> None:
        image_path = self.raw_dir / record.rel_path
        if not image_path.exists():
            self.status_var.set(f"Missing file: {image_path}")
            return
        array = read_tiff(image_path)
        image, scale = build_preview_image(array, self.max_preview)
        self.display_scale = scale
        self.current_image_size = (array.shape[1], array.shape[0])
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.config(width=image.width, height=image.height)
        if self.image_id_on_canvas is None:
            self.image_id_on_canvas = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.canvas.itemconfig(self.image_id_on_canvas, image=self.photo)

        self.current_tubules = []
        ann = load_polygon_annotation(annotation_path(self.annotations_path, record.image_id))
        if ann is not None:
            self.current_tubules = ann.tubules
        self.quality_var.set(record.quality.value)
        self.include_var.set(record.include.value)
        self.negative_training_var.set(record.negative_training.value)
        self.notes_entry.delete("1.0", tk.END)
        self.notes_entry.insert(tk.END, record.notes)
        self.orientation_var.set(Orientation.UNKNOWN.value)
        self.selected_tubule_idx = None
        self.selected_vertex_idx = None
        self.drawing_mode = False
        self.current_points = []
        self.refresh_tubule_list()
        self.redraw_overlays()
        self.status_var.set(
            f"{self.index + 1}/{len(self.records)}: {record.image_id} | include={record.include.value} "
            f"| negative={record.negative_training.value} | quality={record.quality.value}"
        )

    def save_current(self) -> None:
        record = self.records[self.index]
        record.quality = Quality(self.quality_var.get())
        record.include = IncludeFlag(self.include_var.get())
        record.negative_training = NegativeTrainingFlag(self.negative_training_var.get())
        record.notes = self.notes_entry.get("1.0", tk.END).strip()
        record.labeled_by = self.labeled_by_entry.get().strip()
        record.labeled_at = now_iso()
        self.labels[record.rel_path] = record
        save_labels(self.labels_path, self.labels)

        annotation = PolygonAnnotation(
            image_id=record.image_id,
            rel_path=record.rel_path,
            include=record.include,
            quality=record.quality,
            tubules=self.current_tubules,
            labeled_by=record.labeled_by,
            labeled_at=record.labeled_at,
        )
        save_polygon_annotation(annotation, annotation_path(self.annotations_path, record.image_id))
        self.status_var.set(f"Saved {record.rel_path} ({len(self.current_tubules)} tubules)")
        self.refresh_records(current_image_id=record.image_id)

    def next_record(self) -> None:
        if not self.records:
            return
        if self.index + 1 >= len(self.records):
            messagebox.showinfo("Annotation Tool", "Reached end of list.")
            return
        self.index += 1
        self.load_record(self.records[self.index])

    def prev_record(self) -> None:
        if not self.records:
            return
        if self.index == 0:
            return
        self.index -= 1
        self.load_record(self.records[self.index])

    def save_and_next(self) -> None:
        self.save_current()
        self.next_record()

    def run(self) -> None:
        self.root.mainloop()

    def on_tab_change(self, _event: tk.Event) -> None:
        next_tab = "included" if self.tab_control.index("current") == 0 else "excluded"
        if next_tab == self.active_tab:
            return
        if self.records and 0 <= self.index < len(self.records):
            self.tab_positions[self.active_tab] = self.records[self.index].image_id
        self.active_tab = next_tab
        self.refresh_records()

    def refresh_records(self, current_image_id: Optional[str] = None) -> None:
        self.records = self._records_for_tab(self.active_tab)
        if not self.records:
            self.index = 0
            self.current_tubules = []
            self.redraw_overlays()
            self.status_var.set(f"No records in '{self.active_tab}' tab.")
            return
        target_id = current_image_id or self.tab_positions.get(self.active_tab)
        if target_id:
            new_idx = next((i for i, r in enumerate(self.records) if r.image_id == target_id), None)
            if new_idx is not None:
                self.index = new_idx
            else:
                self.index = min(self.index, len(self.records) - 1)
        else:
            self.index = min(self.index, len(self.records) - 1)
        self.load_record(self.records[self.index])

    def _records_for_tab(self, tab: str) -> List[LabelRecord]:
        filtered: List[LabelRecord] = []
        for record in self.all_records:
            if self.include_filter is not None and record.include != self.include_filter:
                continue
            if self.include_yes_only and record.include != IncludeFlag.YES:
                continue
            if tab == "included":
                if record.include == IncludeFlag.NO:
                    continue
            else:
                if record.include != IncludeFlag.NO:
                    continue
            if self.only_unlabeled and annotation_path(self.annotations_path, record.image_id).exists():
                continue
            filtered.append(record)
        filtered.sort(key=lambda r: r.rel_path)
        return filtered


class BoxAnnotationTool:
    def __init__(
        self,
        raw_dir: Path,
        labels_path: Path,
        annotations_path: Path,
        records: List[LabelRecord],
        labeled_by: str = "",
        max_preview: int = 1024,
    ) -> None:
        self.raw_dir = raw_dir
        self.labels_path = labels_path
        self.annotations_path = annotations_path
        self.records = records
        self.labels = load_labels(labels_path)
        self.max_preview = max_preview
        self.index = 0
        self.display_scale = 1.0
        self.current_boxes: List[BoxAnnotation] = []
        self.selected_box_id: Optional[str] = None
        self.redraw_mode = False
        self.labeled_by_default = labeled_by

        self.root = tk.Tk()
        self.root.title("Tubule Box Annotation Tool (Legacy)")

        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.grid(row=0, column=0, rowspan=6, sticky="nsew", padx=6, pady=6)

        sidebar = tk.Frame(self.root)
        sidebar.grid(row=0, column=1, sticky="n", padx=6, pady=6)

        tk.Button(sidebar, text="Prev", command=self.prev_record).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Next", command=self.next_record).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Save", command=self.save_current).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Save & Next", command=self.save_and_next).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Redraw Selected", command=self.enable_redraw).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Delete Selected", command=self.delete_selected).pack(fill=tk.X, pady=2)

        tk.Label(sidebar, text="Quality").pack(anchor="w", pady=(8, 0))
        self.quality_var = tk.StringVar(value=Quality.UNKNOWN.value)
        tk.OptionMenu(sidebar, self.quality_var, *[q.value for q in Quality]).pack(fill=tk.X)

        tk.Label(sidebar, text="Include").pack(anchor="w", pady=(8, 0))
        self.include_var = tk.StringVar(value=IncludeFlag.UNKNOWN.value)
        tk.OptionMenu(sidebar, self.include_var, *[i.value for i in IncludeFlag]).pack(fill=tk.X)

        tk.Label(sidebar, text="Negative training").pack(anchor="w", pady=(8, 0))
        self.negative_training_var = tk.StringVar(value=NegativeTrainingFlag.NO.value)
        tk.OptionMenu(sidebar, self.negative_training_var, *[n.value for n in NegativeTrainingFlag]).pack(fill=tk.X)

        tk.Label(sidebar, text="Notes").pack(anchor="w", pady=(8, 0))
        self.notes_entry = tk.Text(sidebar, height=4, width=28)
        self.notes_entry.pack(fill=tk.X)

        tk.Label(sidebar, text="Labeled by").pack(anchor="w", pady=(8, 0))
        self.labeled_by_entry = tk.Entry(sidebar)
        self.labeled_by_entry.pack(fill=tk.X)
        if self.labeled_by_default:
            self.labeled_by_entry.insert(0, self.labeled_by_default)

        tk.Label(sidebar, text="Boxes").pack(anchor="w", pady=(8, 0))
        self.box_list = tk.Listbox(sidebar, height=8)
        self.box_list.pack(fill=tk.BOTH, expand=False)
        self.box_list.bind("<<ListboxSelect>>", self.on_box_select)

        self.status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status_var).grid(row=6, column=0, columnspan=2, sticky="ew", padx=6)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.root.bind("<Delete>", lambda _event: self.delete_selected())

        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_rect_id: Optional[int] = None
        self.photo: Optional[ImageTk.PhotoImage] = None
        self.image_id_on_canvas: Optional[int] = None

        if self.records:
            self.load_record(self.records[0])
        else:
            messagebox.showinfo("Annotation Tool", "No records to annotate.")
            self.root.destroy()

    def on_mouse_down(self, event: tk.Event) -> None:
        if self.photo is None:
            return
        self.drag_start = (event.x, event.y)
        if self.drag_rect_id is not None:
            self.canvas.delete(self.drag_rect_id)
            self.drag_rect_id = None

    def on_mouse_drag(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        if self.drag_rect_id is None:
            self.drag_rect_id = self.canvas.create_rectangle(x0, y0, event.x, event.y, outline="yellow", width=2)
        else:
            self.canvas.coords(self.drag_rect_id, x0, y0, event.x, event.y)

    def on_mouse_up(self, event: tk.Event) -> None:
        if self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        self.drag_start = None
        if self.drag_rect_id is not None:
            self.canvas.delete(self.drag_rect_id)
            self.drag_rect_id = None
        if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
            self.select_box_at(event.x, event.y)
            return
        x_min, x_max = sorted([x0, x1])
        y_min, y_max = sorted([y0, y1])
        bbox = self.display_to_original(x_min, y_min, x_max, y_max)
        if self.redraw_mode and self.selected_box_id:
            self.replace_selected_box(bbox)
        else:
            self.add_box(bbox)
        self.redraw_mode = False

    def add_box(self, bbox_xywh: List[float]) -> None:
        box = BoxAnnotation(box_id=new_box_id(), bbox_xywh=bbox_xywh, label="tubule", notes="")
        self.current_boxes.append(box)
        self.refresh_boxes()
        self.select_box(box.box_id)

    def replace_selected_box(self, bbox_xywh: List[float]) -> None:
        for idx, box in enumerate(self.current_boxes):
            if box.box_id == self.selected_box_id:
                self.current_boxes[idx] = BoxAnnotation(
                    box_id=box.box_id,
                    bbox_xywh=bbox_xywh,
                    label=box.label,
                    notes=box.notes,
                )
                break
        self.refresh_boxes()
        self.select_box(self.selected_box_id)

    def delete_selected(self) -> None:
        if not self.selected_box_id:
            return
        self.current_boxes = [b for b in self.current_boxes if b.box_id != self.selected_box_id]
        self.selected_box_id = None
        self.refresh_boxes()

    def enable_redraw(self) -> None:
        if not self.selected_box_id:
            messagebox.showinfo("Redraw", "Select a box to redraw.")
            return
        self.redraw_mode = True
        self.status_var.set("Redraw mode: drag a new box to replace selection.")

    def on_box_select(self, _event: tk.Event) -> None:
        selection = self.box_list.curselection()
        if not selection:
            return
        idx = selection[0]
        if idx >= len(self.current_boxes):
            return
        box_id = self.current_boxes[idx].box_id
        self.select_box(box_id)

    def select_box(self, box_id: Optional[str]) -> None:
        self.selected_box_id = box_id
        self.refresh_boxes()
        if box_id is None:
            return
        for idx, box in enumerate(self.current_boxes):
            if box.box_id == box_id:
                self.box_list.selection_clear(0, tk.END)
                self.box_list.selection_set(idx)
                self.box_list.activate(idx)
                break

    def select_box_at(self, x: float, y: float) -> None:
        if not self.current_boxes:
            return
        inv = 1.0 / self.display_scale if self.display_scale > 0 else 1.0
        ox = x * inv
        oy = y * inv
        for box in self.current_boxes:
            bx, by, bw, bh = box.bbox_xywh
            if bx <= ox <= bx + bw and by <= oy <= by + bh:
                self.select_box(box.box_id)
                return

    def refresh_boxes(self) -> None:
        self.box_list.delete(0, tk.END)
        self.canvas.delete("box")
        for box in self.current_boxes:
            x, y, w, h = box.bbox_xywh
            x0, y0 = self.original_to_display(x, y)
            x1, y1 = self.original_to_display(x + w, y + h)
            outline = "lime" if box.box_id != self.selected_box_id else "red"
            self.canvas.create_rectangle(x0, y0, x1, y1, outline=outline, width=2, tags="box")
            self.box_list.insert(tk.END, f"{box.box_id[:6]}: {int(x)},{int(y)},{int(w)},{int(h)}")

    def original_to_display(self, x: float, y: float) -> Tuple[float, float]:
        return x * self.display_scale, y * self.display_scale

    def display_to_original(self, x0: float, y0: float, x1: float, y1: float) -> List[float]:
        inv = 1.0 / self.display_scale if self.display_scale > 0 else 1.0
        ox0 = max(0.0, x0 * inv)
        oy0 = max(0.0, y0 * inv)
        ox1 = max(0.0, x1 * inv)
        oy1 = max(0.0, y1 * inv)
        return [ox0, oy0, max(0.0, ox1 - ox0), max(0.0, oy1 - oy0)]

    def load_record(self, record: LabelRecord) -> None:
        image_path = self.raw_dir / record.rel_path
        if not image_path.exists():
            self.status_var.set(f"Missing file: {image_path}")
            return
        array = read_tiff(image_path)
        image, scale = build_preview_image(array, self.max_preview)
        self.display_scale = scale
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.config(width=image.width, height=image.height)
        if self.image_id_on_canvas is None:
            self.image_id_on_canvas = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.canvas.itemconfig(self.image_id_on_canvas, image=self.photo)
        self.current_boxes = []
        ann = load_annotation(annotation_path(self.annotations_path, record.image_id))
        if ann is not None:
            self.current_boxes = ann.boxes
        self.quality_var.set(record.quality.value)
        self.include_var.set(record.include.value)
        self.negative_training_var.set(record.negative_training.value)
        self.notes_entry.delete("1.0", tk.END)
        self.notes_entry.insert(tk.END, record.notes)
        self.status_var.set(f"{self.index + 1}/{len(self.records)}: {record.rel_path}")
        self.selected_box_id = None
        self.refresh_boxes()

    def save_current(self) -> None:
        record = self.records[self.index]
        record.quality = Quality(self.quality_var.get())
        record.include = IncludeFlag(self.include_var.get())
        record.negative_training = NegativeTrainingFlag(self.negative_training_var.get())
        record.notes = self.notes_entry.get("1.0", tk.END).strip()
        record.labeled_by = self.labeled_by_entry.get().strip()
        record.labeled_at = now_iso()
        self.labels[record.rel_path] = record
        save_labels(self.labels_path, self.labels)

        annotation = ImageAnnotation(
            image_id=record.image_id,
            rel_path=record.rel_path,
            tissue="testis_tubule",
            boxes=self.current_boxes,
            labeled_by=record.labeled_by,
            labeled_at=record.labeled_at,
        )
        save_annotation(annotation, annotation_path(self.annotations_path, record.image_id))
        self.status_var.set(f"Saved {record.rel_path} ({len(self.current_boxes)} boxes)")

    def next_record(self) -> None:
        if not self.records:
            return
        if self.index + 1 >= len(self.records):
            messagebox.showinfo("Annotation Tool", "Reached end of list.")
            return
        self.index += 1
        self.load_record(self.records[self.index])

    def prev_record(self) -> None:
        if not self.records:
            return
        if self.index == 0:
            return
        self.index -= 1
        self.load_record(self.records[self.index])

    def save_and_next(self) -> None:
        self.save_current()
        self.next_record()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate seminiferous tubules with polygons (default) or boxes (legacy)")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir())
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--annotations-dir", type=Path, default=annotations_dir())
    parser.add_argument("--include-yes-only", action="store_true", help="Only include include=yes images")
    parser.add_argument("--include", type=str, choices=[i.value for i in IncludeFlag], help="Filter include flag")
    parser.add_argument("--only-unlabeled", action="store_true", help="Only show images without annotation JSON")
    parser.add_argument("--start", type=str, default="", help="Start at a specific image_id")
    parser.add_argument("--labeled-by", type=str, default="", help="Annotator name/initials")
    parser.add_argument("--max-preview", type=int, default=1024, help="Max preview size for display")
    parser.add_argument("--mode", type=str, choices=["polygon", "box"], default="polygon")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_index(args.raw_dir, args.labels_path, prune_missing=False)
    labels = load_labels(args.labels_path)
    include_filter = IncludeFlag(args.include) if args.include else None

    if args.mode == "box":
        records = filter_records(labels, args.annotations_dir, include_filter, args.include_yes_only, args.only_unlabeled)
        if not records:
            logger.info("No images available for annotation. Check data/raw and labels.csv.")
            return
        if args.start:
            start_idx = next((i for i, r in enumerate(records) if r.image_id == args.start), None)
            if start_idx is not None:
                records = records[start_idx:] + records[:start_idx]
        tool = BoxAnnotationTool(
            args.raw_dir,
            args.labels_path,
            args.annotations_dir,
            records,
            labeled_by=args.labeled_by,
            max_preview=args.max_preview,
        )
    else:
        records = list(labels.values())
        records.sort(key=lambda r: r.rel_path)
        if not records:
            logger.info("No images available for annotation. Check data/raw and labels.csv.")
            return
        tool = PolygonAnnotationTool(
            args.raw_dir,
            args.labels_path,
            args.annotations_dir,
            records,
            include_filter=include_filter,
            include_yes_only=args.include_yes_only,
            only_unlabeled=args.only_unlabeled,
            start_image_id=args.start,
            labeled_by=args.labeled_by,
            max_preview=args.max_preview,
        )
    tool.run()


if __name__ == "__main__":
    main()
