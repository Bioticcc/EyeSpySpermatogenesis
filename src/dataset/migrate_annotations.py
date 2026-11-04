from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

from src.dataset.schema import Orientation
from src.utils.logging import get_logger
from src.utils.paths import annotations_dir

logger = get_logger(__name__)


def migrate_annotation_data(data: Dict[str, Any]) -> int:
    tubules = data.get("tubules", [])
    updated = 0
    if isinstance(tubules, list):
        for tubule in tubules:
            if not isinstance(tubule, dict):
                continue
            changed = False
            if "orientation" not in tubule:
                tubule["orientation"] = Orientation.UNKNOWN.value
                changed = True
            if "partial" not in tubule:
                tubule["partial"] = False
                changed = True
            if changed:
                updated += 1
    return updated


def migrate_annotation_file(path: Path, in_place: bool) -> int:
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except Exception as exc:
        logger.warning("Skipping %s (failed to parse JSON): %s", path.name, exc)
        return 0
    if not isinstance(data, dict):
        logger.warning("Skipping %s (unexpected JSON structure).", path.name)
        return 0
    updated = migrate_annotation_data(data)
    if updated and in_place:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return updated


def migrate_annotations(annotations_path: Path, in_place: bool) -> Tuple[int, int]:
    updated_files = 0
    updated_tubules = 0
    if not annotations_path.exists():
        logger.warning("Annotations directory does not exist: %s", annotations_path)
        return 0, 0
    for path in sorted(annotations_path.glob("*.json")):
        updated = migrate_annotation_file(path, in_place=in_place)
        if updated:
            updated_files += 1
            updated_tubules += updated
    return updated_files, updated_tubules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill missing tubule orientation/partial fields.")
    parser.add_argument("--annotations-dir", type=Path, default=annotations_dir())
    parser.add_argument("--in-place", action="store_true", help="Write updates back to JSON files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    updated_files, updated_tubules = migrate_annotations(args.annotations_dir, in_place=args.in_place)
    if args.in_place:
        logger.info("Updated %d file(s); filled %d tubule(s).", updated_files, updated_tubules)
    else:
        logger.info(
            "Dry run: %d file(s) would be updated; %d tubule(s) missing orientation.",
            updated_files,
            updated_tubules,
        )


if __name__ == "__main__":
    main()
