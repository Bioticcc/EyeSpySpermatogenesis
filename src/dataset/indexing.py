from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List

from src.dataset.schema import LabelRecord, default_label_row, label_csv_columns_for_path
from src.utils.logging import get_logger
from src.utils.paths import labels_dir, raw_data_dir

logger = get_logger(__name__)


def find_tiffs(raw_dir: Path) -> List[Path]:
    patterns = ["*.tif", "*.tiff", "*.TIF", "*.TIFF"]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(raw_dir.rglob(pattern))
    return sorted(files)


def load_labels(path: Path) -> Dict[str, LabelRecord]:
    if not path.exists():
        return {}
    labels: Dict[str, LabelRecord] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = LabelRecord.from_row(row)
            labels[record.rel_path] = record
    return labels


def save_labels(path: Path, labels: Dict[str, LabelRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        columns = label_csv_columns_for_path(path)
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for record in labels.values():
            writer.writerow(record.as_row())


def update_index(raw_dir: Path, labels_path: Path, prune_missing: bool = False) -> None:
    existing = load_labels(labels_path)
    seen = set()
    added = 0
    for file in find_tiffs(raw_dir):
        rel = file.relative_to(raw_dir)
        key = str(rel)
        seen.add(key)
        if key not in existing:
            record = default_label_row(rel)
            existing[key] = record
            added += 1
    removed = 0
    if prune_missing:
        to_remove = [k for k in existing if k not in seen]
        for key in to_remove:
            existing.pop(key)
        removed = len(to_remove)
    save_labels(labels_path, existing)
    logger.info("Index updated. Added %d new files, removed %d. Total %d.", added, removed, len(existing))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan raw data folder for TIFF files and maintain labels.csv")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir(), help="Directory containing raw TIFF files")
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=labels_dir() / "labels.csv",
        help="Path to labels CSV (created if missing)",
    )
    parser.add_argument("--prune-missing", action="store_true", help="Remove entries for files no longer present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    update_index(args.raw_dir, args.labels_path, prune_missing=args.prune_missing)


if __name__ == "__main__":
    main()
