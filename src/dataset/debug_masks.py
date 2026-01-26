from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from src.dataset.annotations import annotation_path, load_polygon_annotation
from src.dataset.schema import Orientation
from src.dataset.seg_dataset import TubuleSegmentationDataset, filter_records, load_label_records
from src.training.seg_config import DEFAULT_SEG_CONFIG
from src.utils.logging import get_logger
from src.utils.paths import annotations_dir, labels_dir, raw_data_dir

logger = get_logger(__name__)


def _blend_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int], alpha: float) -> np.ndarray:
    if mask.size == 0 or not mask.any():
        return image.copy()
    overlay = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + color_arr * alpha
    return overlay.clip(0, 255).astype(np.uint8)


def _filter_for_nonempty_annotations(records: List, annotations_dir_path: Path) -> List:
    filtered = []
    missing = 0
    empty = 0
    for record in records:
        ann_path = annotation_path(annotations_dir_path, record.image_id)
        if not ann_path.exists():
            missing += 1
            continue
        ann = load_polygon_annotation(ann_path)
        if ann is None:
            missing += 1
            continue
        if not ann.tubules:
            empty += 1
            continue
        filtered.append(record)
    if missing or empty:
        logger.warning(
            "Skipped %d record(s) without polygon annotations and %d with empty annotations.",
            missing,
            empty,
        )
    return filtered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug segmentation GT mask generation")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir())
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--annotations-dir", type=Path, default=annotations_dir())
    parser.add_argument("--n", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--out", type=Path, default=Path("outputs/mask_debug"))
    parser.add_argument("--max-size", type=int, default=DEFAULT_SEG_CONFIG.max_size)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--allowed-orientations",
        nargs="*",
        choices=[o.value for o in Orientation],
        default=DEFAULT_SEG_CONFIG.allowed_orientations,
        help="Limit polygons to these orientations (default: all)",
    )
    parser.add_argument(
        "--only-high-quality",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Filter to quality=high only",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_label_records(args.labels_path)
    records = filter_records(records, include_yes_only=True, only_high_quality=args.only_high_quality)
    records = _filter_for_nonempty_annotations(records, args.annotations_dir)
    if not records:
        logger.warning("No annotated records available after filtering.")
        return

    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(records)
    records = records[: max(args.n, 0)]
    if not records:
        logger.warning("No records selected for debug visualization.")
        return

    dataset = TubuleSegmentationDataset(
        records,
        args.raw_dir,
        args.annotations_dir,
        max_size=args.max_size,
        augment=False,
        require_annotations=True,
        instance_mode=False,
        allowed_orientations=args.allowed_orientations,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    for idx in range(len(dataset)):
        record = dataset.records[idx]
        image, mask = dataset[idx]
        gt_mask = mask.squeeze(0).numpy().astype(bool)
        gt_pos_frac = float(gt_mask.mean()) if gt_mask.size else 0.0
        logger.info(
            "debug_masks image_id=%s file=%s gt_pos_frac=%.4f",
            record.image_id,
            record.rel_path,
            gt_pos_frac,
        )

        image_np = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        overlay = _blend_mask(image_np, gt_mask, (0, 255, 0), alpha=0.35)
        out_path = args.out / f"{record.image_id}_gt.png"
        Image.fromarray(overlay).save(out_path)


if __name__ == "__main__":
    main()
