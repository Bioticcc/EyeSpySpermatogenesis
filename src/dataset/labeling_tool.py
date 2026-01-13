from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from PIL import Image, ImageTk
import tkinter as tk
import tkinter.messagebox as messagebox

from src.dataset.indexing import load_labels, save_labels
from src.dataset.schema import IncludeFlag, LabelRecord, Quality, StageCluster, LABEL_CSV_COLUMNS, now_iso
from src.utils.logging import get_logger
from src.utils.paths import labels_dir, raw_data_dir
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


def filter_records(records: Dict[str, LabelRecord], only_unlabeled: bool, include_not_yes: bool) -> List[LabelRecord]:
    filtered: List[LabelRecord] = []
    for record in records.values():
        if only_unlabeled and record.stage_cluster != StageCluster.UNKNOWN:
            continue
        if include_not_yes and record.include == IncludeFlag.YES:
            continue
        filtered.append(record)
    filtered.sort(key=lambda r: r.rel_path)
    return filtered


def list_records(records: Iterable[LabelRecord], limit: int = 20) -> None:
    records_list = list(records)
    for idx, record in enumerate(records_list):
        if idx >= limit:
            logger.info("... and more (%d total)", len(records_list))
            break
        logger.info(
            "%s | stage=%s | quality=%s | include=%s | notes=%s",
            record.rel_path,
            record.stage_cluster.value,
            record.quality.value,
            record.include.value,
            record.notes,
        )


def update_single_record(
    labels_path: Path,
    rel_path: str,
    stage: Optional[str],
    quality: Optional[str],
    include: Optional[str],
    notes: Optional[str],
    labeled_by: Optional[str],
) -> None:
    labels = load_labels(labels_path)
    if rel_path not in labels:
        raise SystemExit(f"{rel_path} not found in labels. Run indexing first.")
    record = labels[rel_path]
    if stage:
        record.stage_cluster = StageCluster(stage)
    if quality:
        record.quality = Quality(quality)
    if include:
        record.include = IncludeFlag(include)
    if notes is not None:
        record.notes = notes
    if labeled_by is not None:
        record.labeled_by = labeled_by
    record.labeled_at = now_iso()
    labels[rel_path] = record
    save_labels(labels_path, labels)
    logger.info("Updated %s", rel_path)


class Reviewer:
    def __init__(self, raw_dir: Path, labels_path: Path, records: List[LabelRecord]) -> None:
        self.raw_dir = raw_dir
        self.labels_path = labels_path
        self.labels = load_labels(labels_path)
        self.records = records
        self.index = 0

        self.root = tk.Tk()
        self.root.title("Labeling Tool")

        self.image_label = tk.Label(self.root)
        self.image_label.pack()

        control = tk.Frame(self.root)
        control.pack(fill=tk.X, padx=4, pady=4)

        tk.Label(control, text="Stage").grid(row=0, column=0, sticky="w")
        self.stage_var = tk.StringVar(value=StageCluster.UNKNOWN.value)
        tk.OptionMenu(control, self.stage_var, *[c.value for c in StageCluster]).grid(row=0, column=1, sticky="w")

        tk.Label(control, text="Quality").grid(row=1, column=0, sticky="w")
        self.quality_var = tk.StringVar(value=Quality.UNKNOWN.value)
        tk.OptionMenu(control, self.quality_var, *[q.value for q in Quality]).grid(row=1, column=1, sticky="w")

        tk.Label(control, text="Include").grid(row=2, column=0, sticky="w")
        self.include_var = tk.StringVar(value=IncludeFlag.UNKNOWN.value)
        tk.OptionMenu(control, self.include_var, *[i.value for i in IncludeFlag]).grid(row=2, column=1, sticky="w")

        tk.Label(control, text="Notes").grid(row=3, column=0, sticky="nw")
        self.notes_entry = tk.Text(control, height=3, width=40)
        self.notes_entry.grid(row=3, column=1, sticky="w")

        buttons = tk.Frame(self.root)
        buttons.pack(fill=tk.X, padx=4, pady=4)
        tk.Button(buttons, text="Save & Next", command=self.save_and_next).pack(side=tk.LEFT, padx=2)
        tk.Button(buttons, text="Skip", command=self.next_record).pack(side=tk.LEFT, padx=2)
        tk.Button(buttons, text="Quit", command=self.root.destroy).pack(side=tk.LEFT, padx=2)

        self.status_var = tk.StringVar(value="")
        tk.Label(self.root, textvariable=self.status_var).pack(fill=tk.X)

        if self.records:
            self.load_record(self.records[0])
        else:
            messagebox.showinfo("Labeling Tool", "No records to review.")
            self.root.destroy()

    def load_record(self, record: LabelRecord) -> None:
        image_path = self.raw_dir / record.rel_path
        if not image_path.exists():
            self.status_var.set(f"Missing file: {image_path}")
            return
        array = read_tiff(image_path)
        image = Image.fromarray((array * 255).astype("uint8"))
        image.thumbnail((768, 768))
        self.photo = ImageTk.PhotoImage(image=image)
        self.image_label.configure(image=self.photo)

        self.stage_var.set(record.stage_cluster.value)
        self.quality_var.set(record.quality.value)
        self.include_var.set(record.include.value)
        self.notes_entry.delete("1.0", tk.END)
        self.notes_entry.insert(tk.END, record.notes)
        self.status_var.set(f"{self.index + 1}/{len(self.records)}: {record.rel_path}")

    def save_current(self) -> None:
        record = self.records[self.index]
        record.stage_cluster = StageCluster(self.stage_var.get())
        record.quality = Quality(self.quality_var.get())
        record.include = IncludeFlag(self.include_var.get())
        record.notes = self.notes_entry.get("1.0", tk.END).strip()
        record.labeled_at = now_iso()
        self.labels[record.rel_path] = record
        save_labels(self.labels_path, self.labels)
        self.status_var.set(f"Saved {record.rel_path}")

    def next_record(self) -> None:
        if not self.records:
            return
        self.index += 1
        if self.index >= len(self.records):
            messagebox.showinfo("Labeling Tool", "Review complete.")
            self.root.destroy()
            return
        self.load_record(self.records[self.index])

    def save_and_next(self) -> None:
        self.save_current()
        self.next_record()

    def run(self) -> None:
        self.root.mainloop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Labeling and tagging tool")
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir())
    parser.add_argument("--list", action="store_true", help="List records")
    parser.add_argument("--only-unlabeled", action="store_true", help="Filter to unlabeled records")
    parser.add_argument("--include-not-yes", action="store_true", help="Filter to include flag not yes")
    parser.add_argument("--image", type=str, help="Relative path of image to update")
    parser.add_argument("--stage", type=str, choices=[c.value for c in StageCluster], help="Stage cluster value")
    parser.add_argument("--quality", type=str, choices=[q.value for q in Quality], help="Quality flag")
    parser.add_argument("--include", dest="include_flag", type=str, choices=[i.value for i in IncludeFlag], help="Include flag")
    parser.add_argument("--notes", type=str, help="Notes to attach")
    parser.add_argument("--labeled-by", type=str, help="Annotator name/initials")
    parser.add_argument("--review", action="store_true", help="Launch Tkinter review UI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels_path)
    if args.image:
        update_single_record(
            args.labels_path,
            args.image,
            stage=args.stage,
            quality=args.quality,
            include=args.include_flag,
            notes=args.notes,
            labeled_by=args.labeled_by,
        )
        return

    records = filter_records(labels, args.only_unlabeled, args.include_not_yes)
    if args.list:
        list_records(records)
    if args.review:
        reviewer = Reviewer(args.raw_dir, args.labels_path, records)
        reviewer.run()
    if not args.list and not args.review and not args.image:
        logger.info("Nothing to do. Use --list, --review or --image to update.")


if __name__ == "__main__":
    main()
