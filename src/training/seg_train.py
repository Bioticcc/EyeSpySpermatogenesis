from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.dataset.annotations import annotation_path, load_polygon_annotation
from src.dataset.schema import NegativeTrainingFlag, Orientation
from src.dataset.seg_dataset import TubuleSegmentationDataset, filter_records, load_label_records
from src.training.seg_config import DEFAULT_SEG_CONFIG
from src.training.seg_metrics import dice_coefficient, iou_score, precision_recall
from src.training.seg_model import create_segmentation_model
from src.utils.logging import get_logger
from src.utils.paths import annotations_dir, labels_dir, raw_data_dir
from src.utils.seed import set_seed, worker_init_fn

logger = get_logger(__name__)


def split_records(records: List, val_ratio: float, seed: int) -> Tuple[List, List]:
    if not records or val_ratio <= 0:
        return records, []
    if len(records) < 2:
        return records, []
    train_records, val_records = train_test_split(records, test_size=val_ratio, random_state=seed, shuffle=True)
    return list(train_records), list(val_records)


def filter_for_annotations(
    records: List,
    annotations_dir_path: Path,
    require_nonempty: bool,
    allow_negative_training: bool,
) -> List:
    filtered = []
    missing = 0
    empty = 0
    empty_negative = 0
    for record in records:
        ann_path = annotation_path(annotations_dir_path, record.image_id)
        if not ann_path.exists():
            missing += 1
            continue
        if require_nonempty or allow_negative_training:
            ann = load_polygon_annotation(ann_path)
            if ann is None:
                missing += 1
                continue
            if not ann.tubules:
                if allow_negative_training and record.negative_training.value == "yes":
                    empty_negative += 1
                else:
                    empty += 1
                    continue
        filtered.append(record)
    if missing or empty:
        if require_nonempty or allow_negative_training:
            logger.warning(
                "Skipped %d record(s) without polygon annotations and %d with empty annotations.",
                missing,
                empty,
            )
        else:
            logger.warning("Skipped %d record(s) without polygon annotations.", missing)
    if allow_negative_training and empty_negative:
        logger.info("Included %d negative-training record(s) with empty annotations.", empty_negative)
    return filtered


def split_ignore_targets(targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_mask = targets != -1
    targets_bin = (targets > 0.5).float()
    return targets_bin, valid_mask


def dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = targets.float()
    if valid_mask is not None:
        valid = valid_mask.float()
        probs = probs * valid
        targets = targets * valid
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dims)
    total = probs.sum(dims) + targets.sum(dims)
    dice = (2 * intersection + eps) / (total + eps)
    return 1.0 - dice.mean()


def masked_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    targets = targets.float()
    if valid_mask is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    valid = valid_mask.float()
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    loss = loss * valid
    denom = valid.sum()
    if denom.item() <= 0:
        return torch.tensor(0.0, device=logits.device)
    return loss.sum() / denom


def balanced_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-6,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    targets = targets.float()
    if valid_mask is None:
        pos = targets.sum()
        neg = targets.numel() - pos
        if pos.item() <= 0 or neg.item() <= 0:
            return F.binary_cross_entropy_with_logits(logits, targets)
        pos_weight = neg / (pos + eps)
        neg_weight = pos / (neg + eps)
        weights = torch.where(targets > 0.5, pos_weight, neg_weight)
        return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
    valid = valid_mask.float()
    pos = (targets * valid).sum()
    neg = valid.sum() - pos
    if pos.item() <= 0 or neg.item() <= 0:
        return masked_bce_with_logits(logits, targets, valid_mask=valid_mask)
    pos_weight = neg / (pos + eps)
    neg_weight = pos / (neg + eps)
    weights = torch.where(targets > 0.5, pos_weight, neg_weight) * valid
    loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights, reduction="sum")
    return loss / valid.sum().clamp_min(1.0)
    pos = targets.sum()
    neg = targets.numel() - pos
    if pos.item() <= 0 or neg.item() <= 0:
        return F.binary_cross_entropy_with_logits(logits, targets)
    pos_weight = neg / (pos + eps)
    neg_weight = pos / (neg + eps)
    weights = torch.where(targets > 0.5, pos_weight, neg_weight)
    return F.binary_cross_entropy_with_logits(logits, targets, weight=weights)


def area_penalty(
    probs: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if valid_mask is None:
        pred_mean = probs.mean(dim=(1, 2, 3))
        target_mean = targets.float().mean(dim=(1, 2, 3))
        return (pred_mean - target_mean).abs().mean()
    valid = valid_mask.float()
    dims = (1, 2, 3)
    valid_sum = valid.sum(dims).clamp_min(1.0)
    pred_mean = (probs * valid).sum(dims) / valid_sum
    target_mean = (targets.float() * valid).sum(dims) / valid_sum
    return (pred_mean - target_mean).abs().mean()


def evaluate_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    metrics_ignore: bool,
) -> Tuple[float, float, float, float]:
    model.eval()
    dice_scores: List[float] = []
    iou_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            preds = (torch.sigmoid(logits) >= threshold).cpu()
            targets = masks.cpu()
            targets_bin = (targets > 0.5).float()
            valid_mask = targets != -1 if metrics_ignore else None
            if valid_mask is not None and valid_mask.sum().item() == 0:
                continue
            dice_scores.append(dice_coefficient(preds, targets_bin, valid_mask=valid_mask))
            iou_scores.append(iou_score(preds, targets_bin, valid_mask=valid_mask))
            precision, recall = precision_recall(preds, targets_bin, valid_mask=valid_mask)
            precision_scores.append(precision)
            recall_scores.append(recall)
    if not dice_scores:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(np.mean(dice_scores)),
        float(np.mean(iou_scores)),
        float(np.mean(precision_scores)),
        float(np.mean(recall_scores)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train tubule segmentation model (U-Net baseline)")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir())
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--annotations-dir", type=Path, default=annotations_dir())
    parser.add_argument("--output-dir", type=Path, default=Path("models/checkpoints"))
    parser.add_argument("--export-path", type=Path, default=DEFAULT_SEG_CONFIG.export_path)
    parser.add_argument("--epochs", type=int, default=DEFAULT_SEG_CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SEG_CONFIG.batch_size)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_SEG_CONFIG.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_SEG_CONFIG.weight_decay)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_SEG_CONFIG.val_ratio)
    parser.add_argument("--max-size", type=int, default=DEFAULT_SEG_CONFIG.max_size)
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
        help="Require polygon annotations for all records",
    )
    parser.add_argument(
        "--require-nonempty-annotations",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Require at least one tubule in each annotation",
    )
    parser.add_argument(
        "--use-negative-training",
        action="store_true",
        help="Include empty annotations only when negative_training=yes",
    )
    parser.add_argument("--augment", action="store_true", help="Use random flips during training")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SEG_CONFIG.threshold)
    parser.add_argument(
        "--balanced-loss",
        action="store_true",
        help="Reweight BCE to penalize dominant class (useful when masks cover most of the image)",
    )
    parser.add_argument(
        "--ignore-band-px",
        type=int,
        default=32,
        help="Ignore-band dilation radius in pixels (0 disables; overridden by --ignore-band-frac)",
    )
    parser.add_argument(
        "--ignore-band-frac",
        type=float,
        nargs="?",
        const=0.02,
        default=None,
        help="Ignore-band radius as fraction of min(H,W) (default: 0.02 when flag is set)",
    )
    parser.add_argument(
        "--area-penalty",
        type=float,
        default=0.0,
        help="Add L1 penalty on predicted mask area vs target area (0 disables)",
    )
    parser.add_argument(
        "--metrics-ignore",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute metrics on valid pixels only (ignore target == -1)",
    )
    parser.add_argument(
        "--negative-sample-weight",
        type=float,
        default=1.0,
        help="Upsample negative_training=yes records in training (1 disables)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    records = load_label_records(args.labels_path)
    records = filter_records(records, include_yes_only=args.include_yes_only, only_high_quality=args.only_high_quality)
    if args.require_annotations or args.require_nonempty_annotations or args.use_negative_training:
        records = filter_for_annotations(
            records,
            args.annotations_dir,
            require_nonempty=args.require_nonempty_annotations,
            allow_negative_training=args.use_negative_training,
        )
    if not records:
        logger.warning("No records available after filtering. Check labels.csv and annotations.")
        return

    train_records, val_records = split_records(records, args.val_ratio, args.seed)
    if not train_records:
        logger.warning("Training split is empty after filtering.")
        return

    train_ds = TubuleSegmentationDataset(
        train_records,
        args.raw_dir,
        args.annotations_dir,
        max_size=args.max_size,
        augment=args.augment,
        require_annotations=args.require_annotations,
        instance_mode=False,
        allowed_orientations=args.allowed_orientations,
        ignore_band_px=args.ignore_band_px,
        ignore_band_frac=args.ignore_band_frac,
    )
    val_ds = TubuleSegmentationDataset(
        val_records,
        args.raw_dir,
        args.annotations_dir,
        max_size=args.max_size,
        augment=False,
        require_annotations=args.require_annotations,
        instance_mode=False,
        allowed_orientations=args.allowed_orientations,
        ignore_band_px=args.ignore_band_px,
        ignore_band_frac=args.ignore_band_frac,
    )

    sampler = None
    if args.negative_sample_weight > 1.0:
        weights = []
        negative_count = 0
        for record in train_records:
            if record.negative_training == NegativeTrainingFlag.YES:
                weights.append(args.negative_sample_weight)
                negative_count += 1
            else:
                weights.append(1.0)
        if negative_count > 0:
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            logger.info(
                "Using negative sample weight %.2f for %d record(s).",
                args.negative_sample_weight,
                negative_count,
            )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    model = create_segmentation_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_dice = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "seg_best.pt"
    last_path = args.output_dir / "seg_last.pt"
    high_gt_streak = 0
    high_gt_warned = False

    train_loader_len = len(train_loader)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            pos_frac = float((masks == 1).float().mean().item()) if masks.numel() else 0.0
            ignore_frac = float((masks == -1).float().mean().item()) if masks.numel() else 0.0
            neg_frac = float((masks == 0).float().mean().item()) if masks.numel() else 0.0
            logger.info(
                "batch_mask_fracs pos=%.4f ignore=%.4f neg=%.4f",
                pos_frac,
                ignore_frac,
                neg_frac,
            )
            if pos_frac > 0.95:
                high_gt_streak += 1
                if (high_gt_streak >= 2 or train_loader_len == 1) and not high_gt_warned:
                    logger.warning(
                        "GT masks are almost full-image; check polygon rasterization / coordinate scaling."
                    )
                    high_gt_warned = True
            else:
                high_gt_streak = 0
            logits = model(images)
            targets, valid_mask = split_ignore_targets(masks)
            if valid_mask.sum().item() == 0:
                logger.warning("Skipping batch with no valid pixels (all ignored).")
                continue
            valid_frac = float(valid_mask.float().mean().item()) if masks.numel() else 0.0
            if valid_frac < 0.01:
                logger.warning("Low valid pixel fraction (%.4f); loss may be unstable.", valid_frac)
            if args.balanced_loss:
                bce = balanced_bce_with_logits(logits, targets, valid_mask=valid_mask)
            else:
                bce = masked_bce_with_logits(logits, targets, valid_mask=valid_mask)
            dloss = dice_loss(logits, targets, valid_mask=valid_mask)
            loss = bce + dloss
            if args.area_penalty > 0:
                probs = torch.sigmoid(logits)
                loss = loss + args.area_penalty * area_penalty(probs, targets, valid_mask=valid_mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses)) if losses else 0.0

        if val_records:
            dice, iou, precision, recall = evaluate_on_loader(
                model,
                val_loader,
                device=device,
                threshold=args.threshold,
                metrics_ignore=args.metrics_ignore,
            )
            logger.info(
                "Epoch %d/%d | loss=%.4f | val_dice=%.3f val_iou=%.3f val_prec=%.3f val_rec=%.3f",
                epoch + 1,
                args.epochs,
                avg_loss,
                dice,
                iou,
                precision,
                recall,
            )
            if dice > best_dice:
                best_dice = dice
                torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, best_path)
        else:
            logger.info("Epoch %d/%d | loss=%.4f", epoch + 1, args.epochs, avg_loss)

        torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, last_path)

    if best_path.exists():
        args.export_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.load(best_path, map_location="cpu")["state_dict"], args.export_path)
        logger.info("Exported best model to %s", args.export_path)
    else:
        logger.info("Training finished; no best checkpoint saved.")


if __name__ == "__main__":
    main()
