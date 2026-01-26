from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from src.dataset.annotations import annotation_path, load_polygon_annotation
from src.dataset.indexing import load_labels
from src.dataset.schema import IncludeFlag, Orientation, Quality
from src.utils.logging import get_logger
from src.utils.paths import annotations_dir, labels_dir

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report tubule counts by orientation.")
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--annotations-dir", type=Path, default=annotations_dir())
    parser.add_argument(
        "--include",
        choices=[flag.value for flag in IncludeFlag] + ["all"],
        default="all",
        help="Filter records by include flag",
    )
    parser.add_argument(
        "--quality",
        choices=[flag.value for flag in Quality] + ["all"],
        default="all",
        help="Filter records by quality flag",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = load_labels(args.labels_path)
    records = list(labels.values())

    filtered = []
    for record in records:
        if args.include != "all" and record.include.value != args.include:
            continue
        if args.quality != "all" and record.quality.value != args.quality:
            continue
        filtered.append(record)

    counts = Counter({orientation.value: 0 for orientation in Orientation})
    missing = 0
    for record in filtered:
        ann = load_polygon_annotation(annotation_path(args.annotations_dir, record.image_id))
        if ann is None:
            missing += 1
            continue
        for tubule in ann.tubules:
            counts[tubule.orientation.value] += 1

    total_tubules = sum(counts.values())
    logger.info(
        "Records: %d (missing annotations: %d) | filters include=%s quality=%s",
        len(filtered),
        missing,
        args.include,
        args.quality,
    )
    logger.info("Tubules: %d", total_tubules)
    for orientation in Orientation:
        logger.info("  %s: %d", orientation.value, counts[orientation.value])


if __name__ == "__main__":
    main()
