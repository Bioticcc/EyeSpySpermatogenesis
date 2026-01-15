from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.ops import nms

from src.dataset.annotations import annotation_path
from src.dataset.detect_dataset import TubuleDetectDataset, detection_collate_fn, filter_records, load_label_records
from src.training.detect_metrics import summarize_metrics
from src.training.detect_model import create_detector
from src.utils.logging import get_logger
from src.utils.paths import annotations_dir, exported_models_dir, labels_dir, raw_data_dir
from src.utils.seed import set_seed, worker_init_fn

logger = get_logger(__name__)


def split_records(records: List, val_ratio: float, seed: int) -> Tuple[List, List]:
    if not records or val_ratio <= 0:
        return records, []
    if len(records) < 2:
        return records, []
    train_records, val_records = train_test_split(records, test_size=val_ratio, random_state=seed, shuffle=True)
    return list(train_records), list(val_records)


def filter_for_annotations(records: List, annotations_dir: Path) -> List:
    filtered = []
    missing = 0
    for record in records:
        if annotation_path(annotations_dir, record.image_id).exists():
            filtered.append(record)
        else:
            missing += 1
    if missing:
        logger.warning("Skipped %d record(s) without annotations.", missing)
    return filtered


def evaluate_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    score_threshold: float,
    nms_iou: float,
) -> Tuple[float, float, float, float]:
    model.eval()
    gt_boxes_list: List[List[List[float]]] = []
    pred_boxes_list: List[List[List[float]]] = []
    with torch.no_grad():
        for images, targets in loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].cpu().numpy().tolist()
                scores = output["scores"].cpu()
                boxes = output["boxes"].cpu()
                keep = scores >= score_threshold
                scores = scores[keep]
                boxes = boxes[keep]
                if boxes.numel() > 0 and nms_iou > 0:
                    keep_idx = nms(boxes, scores, nms_iou)
                    boxes = boxes[keep_idx]
                pred_boxes = boxes.numpy().tolist()
                gt_boxes_list.append(gt_boxes)
                pred_boxes_list.append(pred_boxes)
    return summarize_metrics(gt_boxes_list, pred_boxes_list, iou_threshold=0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tubule detector (Faster R-CNN baseline)")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir())
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--annotations-dir", type=Path, default=annotations_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--export-path", type=Path, default=exported_models_dir() / "detector.pt")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-size", type=int, default=1024, help="Downscale so max side <= this value")
    parser.add_argument("--only-high-quality", action="store_true", help="Filter to quality=high only")
    parser.add_argument(
        "--include-yes-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Filter to include=yes only",
    )
    parser.add_argument(
        "--require-annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require annotations for all records",
    )
    parser.add_argument("--augment", action="store_true", help="Use random flips during training")
    parser.add_argument("--score-threshold", type=float, default=0.25, help="Score threshold for val metrics")
    parser.add_argument("--nms-iou", type=float, default=0.5, help="NMS IoU for val metrics")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    records = load_label_records(args.labels_path)
    records = filter_records(records, include_yes_only=args.include_yes_only, only_high_quality=args.only_high_quality)
    if args.require_annotations:
        records = filter_for_annotations(records, args.annotations_dir)
    if not records:
        raise SystemExit("No records available after filtering. Check labels.csv and annotations.")

    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    if not train_records:
        raise SystemExit("Training split is empty after filtering.")

    train_ds = TubuleDetectDataset(
        train_records,
        args.raw_dir,
        args.annotations_dir,
        max_size=args.max_size,
        augment=args.augment,
        require_annotations=args.require_annotations,
    )
    val_ds = TubuleDetectDataset(
        val_records,
        args.raw_dir,
        args.annotations_dir,
        max_size=args.max_size,
        augment=False,
        require_annotations=args.require_annotations,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
        worker_init_fn=worker_init_fn,
    )

    model = create_detector(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_path = args.output_dir / "detector_best.pt"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses)) if losses else 0.0

        if val_records:
            precision, recall, mean_iou, count_error = evaluate_on_loader(
                model,
                val_loader,
                device=device,
                score_threshold=args.score_threshold,
                nms_iou=args.nms_iou,
            )
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            logger.info(
                "Epoch %d/%d | loss=%.4f | val_precision=%.3f val_recall=%.3f mean_iou=%.3f count_err=%.3f",
                epoch + 1,
                args.epochs,
                avg_loss,
                precision,
                recall,
                mean_iou,
                count_error,
            )
        else:
            precision = recall = mean_iou = count_error = 0.0
            f1 = 0.0
            logger.info("Epoch %d/%d | loss=%.4f | no val split", epoch + 1, args.epochs, avg_loss)

        if f1 >= best_f1:
            best_f1 = f1
            args.output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "f1": f1}, best_path)
            logger.info("Saved best checkpoint to %s", best_path)

    if best_path.exists():
        export_path = args.export_path
        export_path.parent.mkdir(parents=True, exist_ok=True)
        state = torch.load(best_path, map_location="cpu")
        model.load_state_dict(state["state_dict"])
        torch.save(model.state_dict(), export_path)
        logger.info("Exported detector weights to %s", export_path)
    else:
        logger.warning("No checkpoint saved; export skipped.")


if __name__ == "__main__":
    main()
