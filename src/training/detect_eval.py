from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader
from torchvision.ops import nms

from src.dataset.detect_dataset import TubuleDetectDataset, detection_collate_fn, filter_records, load_label_records
from src.dataset.indexing import load_labels
from src.dataset.schema import LabelRecord
from src.training.detect_metrics import summarize_metrics
from src.training.detect_model import create_detector
from src.utils.logging import get_logger
from src.utils.paths import annotations_dir, exported_models_dir, labels_dir, raw_data_dir
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tubule detector and compute metrics")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir())
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--annotations-dir", type=Path, default=annotations_dir())
    parser.add_argument("--weights", type=Path, default=exported_models_dir() / "detector.pt")
    parser.add_argument("--split-path", type=Path, help="Optional CSV with rel_path column for evaluation")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument(
        "--include-yes-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter to include=yes only",
    )
    parser.add_argument("--only-high-quality", action="store_true")
    parser.add_argument(
        "--require-annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require annotations for all records",
    )
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--save-viz", action="store_true", help="Save overlay images to outputs/")
    parser.add_argument("--viz-dir", type=Path, default=Path("outputs/detect_viz"))
    return parser.parse_args()


def load_split_records(split_path: Path, labels_path: Path) -> List[LabelRecord]:
    labels = load_labels(labels_path)
    records: List[LabelRecord] = []
    if not split_path.exists():
        raise SystemExit(f"Split file not found at {split_path}")
    with split_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel_path = row.get("rel_path", "")
            if rel_path in labels:
                records.append(labels[rel_path])
            else:
                image_id = Path(rel_path).stem
                records.append(LabelRecord(image_id=image_id, rel_path=rel_path))
    return records


def draw_overlay(image: Image.Image, gt_boxes: List[List[float]], pred_boxes: List[List[float]], scores: List[float]) -> Image.Image:
    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        draw.rectangle(box, outline="lime", width=2)
    for box, score in zip(pred_boxes, scores):
        draw.rectangle(box, outline="red", width=2)
        draw.text((box[0], max(0, box[1] - 10)), f"{score:.2f}", fill="red")
    return image


def resize_for_viz(array: np.ndarray, max_size: int | None) -> Image.Image:
    image = Image.fromarray((array * 255).clip(0, 255).astype("uint8"))
    if max_size and max(image.size) > max_size:
        image.thumbnail((max_size, max_size))
    return image


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if not args.weights.exists():
        raise SystemExit(f"Weights not found at {args.weights}")
    if args.split_path:
        records = load_split_records(args.split_path, args.labels_path)
    else:
        records = load_label_records(args.labels_path)
        records = filter_records(records, include_yes_only=args.include_yes_only, only_high_quality=args.only_high_quality)
    if not records:
        raise SystemExit("No records available for evaluation.")

    dataset = TubuleDetectDataset(
        records,
        args.raw_dir,
        args.annotations_dir,
        max_size=args.max_size,
        augment=False,
        require_annotations=args.require_annotations,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    model = create_detector(num_classes=2)
    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    gt_boxes_list: List[List[List[float]]] = []
    pred_boxes_list: List[List[List[float]]] = []

    if args.save_viz:
        args.viz_dir.mkdir(parents=True, exist_ok=True)

    idx = 0
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].cpu().numpy().tolist()
                scores = output["scores"].cpu()
                boxes = output["boxes"].cpu()
                keep = scores >= args.score_threshold
                scores = scores[keep]
                boxes = boxes[keep]
                if boxes.numel() > 0 and args.nms_iou > 0:
                    keep_idx = nms(boxes, scores, args.nms_iou)
                    boxes = boxes[keep_idx]
                    scores = scores[keep_idx]
                pred_boxes = boxes.numpy().tolist()
                gt_boxes_list.append(gt_boxes)
                pred_boxes_list.append(pred_boxes)

                if args.save_viz:
                    record = records[idx]
                    array = read_tiff(args.raw_dir / record.rel_path)
                    image = resize_for_viz(array, args.max_size)
                    image = draw_overlay(image, gt_boxes, pred_boxes, scores.numpy().tolist())
                    out_path = args.viz_dir / f"{record.image_id}_overlay.png"
                    image.save(out_path)
                idx += 1

    precision, recall, mean_iou, count_error = summarize_metrics(gt_boxes_list, pred_boxes_list, iou_threshold=0.5)
    logger.info("Precision@0.5=%.3f Recall@0.5=%.3f MeanIoU=%.3f CountErr=%.3f", precision, recall, mean_iou, count_error)


if __name__ == "__main__":
    main()
