from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
import tkinter.messagebox as messagebox
from PIL import Image, ImageTk

from src.dataset.annotations import ImageAnnotation, BoxAnnotation, annotation_path, load_annotation, new_box_id, save_annotation
from src.dataset.indexing import load_labels, save_labels, update_index
from src.dataset.schema import IncludeFlag, LabelRecord, Quality, now_iso
from src.utils.logging import get_logger
from src.utils.paths import annotations_dir, labels_dir, raw_data_dir
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


def filter_records(records: Dict[str, LabelRecord], include_yes_only: bool) -> List[LabelRecord]:
    filtered: List[LabelRecord] = []
    for record in records.values():
        if include_yes_only and record.include != IncludeFlag.YES:
            continue
        if not include_yes_only and record.include == IncludeFlag.NO:
            continue
        filtered.append(record)
    filtered.sort(key=lambda r: r.rel_path)
    return filtered


class AnnotationTool:
    def __init__(
        self,
        raw_dir: Path,
        labels_path: Path,
        annotations_dir: Path,
        records: List[LabelRecord],
        labeled_by: str = "",
        max_preview: int = 1024,
    ) -> None:
        self.raw_dir = raw_dir
        self.labels_path = labels_path
        self.annotations_dir = annotations_dir
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
        self.root.title("Tubule Annotation Tool")

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
        image = Image.fromarray((array * 255).clip(0, 255).astype("uint8"))
        image.thumbnail((self.max_preview, self.max_preview))
        self.display_scale = image.width / array.shape[1] if array.shape[1] else 1.0
        self.photo = ImageTk.PhotoImage(image=image)
        self.canvas.config(width=image.width, height=image.height)
        if self.image_id_on_canvas is None:
            self.image_id_on_canvas = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.canvas.itemconfig(self.image_id_on_canvas, image=self.photo)
        self.current_boxes = []
        ann = load_annotation(annotation_path(self.annotations_dir, record.image_id))
        if ann is not None:
            self.current_boxes = ann.boxes
        self.quality_var.set(record.quality.value)
        self.include_var.set(record.include.value)
        self.notes_entry.delete("1.0", tk.END)
        self.notes_entry.insert(tk.END, record.notes)
        self.status_var.set(f"{self.index + 1}/{len(self.records)}: {record.rel_path}")
        self.selected_box_id = None
        self.refresh_boxes()

    def save_current(self) -> None:
        record = self.records[self.index]
        record.quality = Quality(self.quality_var.get())
        record.include = IncludeFlag(self.include_var.get())
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
        save_annotation(annotation, annotation_path(self.annotations_dir, record.image_id))
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
    parser = argparse.ArgumentParser(description="Annotate seminiferous tubules with bounding boxes")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir())
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--annotations-dir", type=Path, default=annotations_dir())
    parser.add_argument("--include-yes-only", action="store_true", help="Only include include=yes images")
    parser.add_argument("--labeled-by", type=str, default="", help="Annotator name/initials")
    parser.add_argument("--max-preview", type=int, default=1024, help="Max preview size for display")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_index(args.raw_dir, args.labels_path, prune_missing=False)
    labels = load_labels(args.labels_path)
    records = filter_records(labels, args.include_yes_only)
    if not records:
        logger.info("No images available for annotation. Check data/raw and labels.csv.")
        return
    tool = AnnotationTool(
        args.raw_dir,
        args.labels_path,
        args.annotations_dir,
        records,
        labeled_by=args.labeled_by,
        max_preview=args.max_preview,
    )
    tool.run()


if __name__ == "__main__":
    main()
