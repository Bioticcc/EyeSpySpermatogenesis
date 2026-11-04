from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.dataset.annotations import annotation_path, load_polygon_annotation
from src.dataset.schema import Orientation
from src.dataset.masks import split_tubule_polygons
from src.dataset.seg_dataset import TubuleSegmentationDataset, filter_records, load_label_records
from src.training.seg_config import DEFAULT_SEG_CONFIG
from src.training.seg_metrics import dice_coefficient, iou_score, precision_recall
from src.training.seg_model import create_segmentation_model
from src.utils.logging import get_logger
from src.utils.paths import annotations_dir, labels_dir, raw_data_dir
from src.utils.postprocess import postprocess_mask

logger = get_logger(__name__)


def filter_for_annotations(
    records: List,
    annotations_dir_path: Path,
    require_positive: bool,
    require_any: bool,
    allow_negative_training: bool,
    allowed_orientations: Optional[set[Orientation]] = None,
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
        if require_positive or require_any or allow_negative_training:
            ann = load_polygon_annotation(ann_path)
            if ann is None:
                missing += 1
                continue
            if require_any and not ann.tubules:
                empty += 1
                continue
            _, _, counts = split_tubule_polygons(
                ann.tubules,
                allowed_orientations=allowed_orientations,
            )
            has_positive = counts["pos_full_cross_section"] > 0
            if not has_positive:
                if allow_negative_training and record.negative_training.value == "yes":
                    empty_negative += 1
                else:
                    empty += 1
                    continue
        filtered.append(record)
    if missing or empty:
        if require_positive or require_any or allow_negative_training:
            reason = "without required annotations"
            if require_positive:
                reason = "without positive annotations"
            elif require_any:
                reason = "without any tubule annotations"
            logger.warning(
                "Skipped %d record(s) without polygon annotations and %d %s.",
                missing,
                empty,
                reason,
            )
        else:
            logger.warning("Skipped %d record(s) without polygon annotations.", missing)
    if allow_negative_training and empty_negative:
        logger.info("Included %d negative-training record(s) with empty annotations.", empty_negative)
    return filtered


def _blend_mask(image: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int], alpha: float) -> np.ndarray:
    if mask.size == 0 or not mask.any():
        return image.copy()
    overlay = image.astype(np.float32).copy()
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + color_arr * alpha
    return overlay.clip(0, 255).astype(np.uint8)


def _draw_component_boxes(image: np.ndarray, components: Optional[List[dict]]) -> np.ndarray:
    if not components:
        return image
    from PIL import Image, ImageDraw, ImageFont

    overlay = Image.fromarray(image)
    draw = ImageDraw.Draw(overlay)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for idx, comp in enumerate(components):
        x1, y1, x2, y2 = comp["bbox"]
        draw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)
        label = f"{idx}:{comp['area']}"
        if font is None:
            draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 0))
        else:
            draw.text((x1 + 2, y1 + 2), label, fill=(255, 255, 0), font=font)
    return np.array(overlay)


def save_mask_overlay(
    image: np.ndarray,
    gt_mask: np.ndarray,
    gt_neg_mask: Optional[np.ndarray],
    pred_mask: np.ndarray,
    path: Path,
    components: Optional[List[dict]] = None,
    alpha: float = 0.35,
) -> None:
    overlay = _blend_mask(image, gt_mask, (0, 255, 0), alpha)
    if gt_neg_mask is not None:
        overlay = _blend_mask(overlay, gt_neg_mask, (0, 0, 255), alpha)
    overlay = _blend_mask(overlay, pred_mask, (255, 0, 0), alpha)
    overlay = _draw_component_boxes(overlay, components)
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(overlay).save(path)


def evaluate_on_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
    save_viz: bool,
    viz_dir: Path,
    max_viz: int,
    records: Optional[List],
    postprocess: bool,
    pp_threshold: Optional[float],
    pp_min_area_frac: float,
    pp_max_components: int,
    pp_fill_holes: bool,
    pp_max_hole_frac: float,
    metrics_ignore: bool,
) -> Tuple[float, float, float, float]:
    model.eval()
    dice_scores: List[float] = []
    iou_scores: List[float] = []
    precision_scores: List[float] = []
    recall_scores: List[float] = []
    neg_fp_rates: List[float] = []
    gt_pos_fracs: List[float] = []
    pred_pos_fracs: List[float] = []
    pred_pos_fracs_pre: List[float] = []
    postprocess_enabled = postprocess
    threshold_for_post = threshold if pp_threshold is None else float(pp_threshold)
    saved = 0
    dataset_idx = 0
    with torch.no_grad():
        for images, masks, neg_masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            neg_masks = neg_masks.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            preds = probs >= (threshold_for_post if postprocess_enabled else threshold)
            targets = masks.cpu()
            neg_targets = neg_masks.cpu()
            targets_bin = (targets > 0.5).float()
            valid_mask = targets != -1 if metrics_ignore else None
            if postprocess_enabled:
                preds_post = torch.zeros_like(preds, dtype=torch.bool)
                components_batch: List[Optional[List[dict]]] = []
                for idx in range(preds.shape[0]):
                    prob_map = probs[idx].squeeze(0).numpy()
                    pred_mask = preds[idx].squeeze(0).numpy().astype(bool)
                    height, width = pred_mask.shape
                    min_area_px = 0
                    if pp_min_area_frac and pp_min_area_frac > 0:
                        min_area_px = int(round(pp_min_area_frac * height * width))
                        min_area_px = max(min_area_px, 1)
                    max_hole_px = 0
                    if pp_max_hole_frac and pp_max_hole_frac > 0:
                        max_hole_px = int(round(pp_max_hole_frac * height * width))
                        max_hole_px = max(max_hole_px, 1)
                    pred_mask_post, components = postprocess_mask(
                        pred_mask,
                        min_area_px=min_area_px,
                        max_components=pp_max_components,
                        fill_holes=pp_fill_holes,
                        max_hole_area_px=max_hole_px,
                        probs=prob_map,
                    )
                    preds_post[idx, 0] = torch.from_numpy(pred_mask_post.astype(bool))
                    components_batch.append(components)
            else:
                preds_post = preds
                components_batch = []

            if valid_mask is not None and valid_mask.sum().item() == 0:
                continue
            dice_scores.append(dice_coefficient(preds_post, targets_bin, valid_mask=valid_mask))
            iou_scores.append(iou_score(preds_post, targets_bin, valid_mask=valid_mask))
            precision, recall = precision_recall(preds_post, targets_bin, valid_mask=valid_mask)
            precision_scores.append(precision)
            recall_scores.append(recall)

            for idx in range(images.shape[0]):
                record = records[dataset_idx] if records and dataset_idx < len(records) else None
                dataset_idx += 1
                image_id = record.image_id if record else f"idx_{dataset_idx - 1}"
                filename = record.rel_path if record else "unknown"

                gt_mask = targets_bin[idx].squeeze(0).numpy().astype(bool)
                gt_neg_mask = neg_targets[idx].squeeze(0).numpy().astype(bool)
                pred_mask_pre = preds[idx].squeeze(0).numpy().astype(bool)
                pred_mask_post = preds_post[idx].squeeze(0).numpy().astype(bool)
                prob_map = probs[idx].squeeze(0).numpy()
                components = components_batch[idx] if postprocess_enabled else None

                gt_pos_frac = float(gt_mask.mean()) if gt_mask.size else 0.0
                pred_pos_frac_pre = float(pred_mask_pre.mean()) if pred_mask_pre.size else 0.0
                pred_pos_frac_post = float(pred_mask_post.mean()) if pred_mask_post.size else 0.0
                prob_mean = float(prob_map.mean()) if prob_map.size else 0.0
                prob_min = float(prob_map.min()) if prob_map.size else 0.0
                prob_max = float(prob_map.max()) if prob_map.size else 0.0

                gt_pos_fracs.append(gt_pos_frac)
                pred_pos_fracs_pre.append(pred_pos_frac_pre)
                pred_pos_fracs.append(pred_pos_frac_post)
                if gt_neg_mask.any():
                    neg_fp_rate = float((pred_mask_post & gt_neg_mask).sum() / max(gt_neg_mask.sum(), 1))
                    neg_fp_rates.append(neg_fp_rate)

                if postprocess_enabled:
                    logger.info(
                        "seg_eval image_id=%s file=%s gt_pos_frac=%.4f pred_pos_frac=%.4f pred_pos_frac_post=%.4f "
                        "prob_mean=%.4f prob_min=%.4f prob_max=%.4f threshold=%.3f",
                        image_id,
                        filename,
                        gt_pos_frac,
                        pred_pos_frac_pre,
                        pred_pos_frac_post,
                        prob_mean,
                        prob_min,
                        prob_max,
                        threshold_for_post,
                    )
                else:
                    logger.info(
                        "seg_eval image_id=%s file=%s gt_pos_frac=%.4f pred_pos_frac=%.4f "
                        "prob_mean=%.4f prob_min=%.4f prob_max=%.4f threshold=%.3f",
                        image_id,
                        filename,
                        gt_pos_frac,
                        pred_pos_frac_pre,
                        prob_mean,
                        prob_min,
                        prob_max,
                        threshold,
                    )

                if save_viz and saved < max_viz:
                    image_np = (images[idx].cpu().permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                    base_path = viz_dir / f"seg_viz_{saved:03d}_{image_id}"
                    raw_path = base_path.with_name(f"{base_path.name}_raw.png")
                    save_mask_overlay(image_np, gt_mask, gt_neg_mask, pred_mask_pre, raw_path)
                    if postprocess_enabled:
                        post_path = base_path.with_name(f"{base_path.name}_post.png")
                        save_mask_overlay(
                            image_np,
                            gt_mask,
                            gt_neg_mask,
                            pred_mask_post,
                            post_path,
                            components=components,
                        )
                    saved += 1

    if not gt_pos_fracs:
        logger.warning("No images evaluated; skipping segmentation diagnostics summary.")
        return 0.0, 0.0, 0.0, 0.0

    gt_array = np.array(gt_pos_fracs)
    pred_array = np.array(pred_pos_fracs)
    pred_pre_array = np.array(pred_pos_fracs_pre)
    logger.info(
        "seg_eval summary gt_pos_frac mean=%.4f median=%.4f",
        float(gt_array.mean()),
        float(np.median(gt_array)),
    )
    logger.info(
        "seg_eval summary pred_pos_frac mean=%.4f median=%.4f",
        float(pred_pre_array.mean()),
        float(np.median(pred_pre_array)),
    )
    if postprocess_enabled:
        logger.info(
            "seg_eval summary pred_pos_frac_post mean=%.4f median=%.4f",
            float(pred_array.mean()),
            float(np.median(pred_array)),
        )
    logger.info(
        "seg_eval summary pred_pos_frac>0.95 count=%d",
        int((pred_pre_array > 0.95).sum()),
    )
    if postprocess_enabled:
        logger.info(
            "seg_eval summary pred_pos_frac_post>0.95 count=%d",
            int((pred_array > 0.95).sum()),
        )
    logger.info(
        "seg_eval summary gt_pos_frac>0.95 count=%d",
        int((gt_array > 0.95).sum()),
    )
    if neg_fp_rates:
        neg_fp_arr = np.array(neg_fp_rates)
        logger.info(
            "seg_eval summary neg_tubule_fp_rate mean=%.4f median=%.4f",
            float(neg_fp_arr.mean()),
            float(np.median(neg_fp_arr)),
        )

    return (
        float(np.mean(dice_scores)),
        float(np.mean(iou_scores)),
        float(np.mean(precision_scores)),
        float(np.mean(recall_scores)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate tubule segmentation model")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir())
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--annotations-dir", type=Path, default=annotations_dir())
    parser.add_argument("--weights", type=Path, default=DEFAULT_SEG_CONFIG.export_path)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
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
        help="Require at least one positive (full cross-section) tubule in each annotation",
    )
    parser.add_argument(
        "--require-any-annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require at least one tubule polygon (positive or negative) in each annotation",
    )
    parser.add_argument(
        "--use-negative-training",
        action="store_true",
        help="Include empty annotations only when negative_training=yes",
    )
    parser.add_argument("--threshold", type=float, default=DEFAULT_SEG_CONFIG.threshold)
    parser.add_argument(
        "--metrics-ignore",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute metrics on valid pixels only (ignore target == -1)",
    )
    parser.add_argument(
        "--ignore-band-px",
        type=int,
        default=32,
        help="Ignore-band dilation radius in pixels (0 disables; only used with --metrics-ignore)",
    )
    parser.add_argument(
        "--ignore-band-frac",
        type=float,
        nargs="?",
        const=0.02,
        default=None,
        help="Ignore-band radius as fraction of min(H,W) (default: 0.02 when flag is set)",
    )
    parser.add_argument("--save-viz", action="store_true", help="Save overlay visualizations")
    parser.add_argument("--viz-dir", type=Path, default=Path("outputs/seg_viz"))
    parser.add_argument("--max-viz", type=int, default=8)
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Enable postprocessing (threshold -> clean -> components)",
    )
    parser.add_argument(
        "--pp-threshold",
        type=float,
        default=None,
        help="Postprocess threshold (defaults to --threshold when unset)",
    )
    parser.add_argument(
        "--pp-min-area-frac",
        type=float,
        default=0.01,
        help="Remove predicted components smaller than frac*H*W",
    )
    parser.add_argument(
        "--pp-max-components",
        type=int,
        default=4,
        help="Keep largest K predicted components",
    )
    parser.add_argument(
        "--pp-fill-holes",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fill small holes inside predicted components",
    )
    parser.add_argument(
        "--pp-max-hole-frac",
        type=float,
        default=0.005,
        help="Fill holes smaller than frac*H*W",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    records = load_label_records(args.labels_path)
    records = filter_records(records, include_yes_only=args.include_yes_only, only_high_quality=args.only_high_quality)
    allowed_orientations = TubuleSegmentationDataset._normalize_orientations(args.allowed_orientations)
    require_positive = args.require_nonempty_annotations
    require_any = args.require_any_annotations
    if args.require_annotations or require_positive or require_any or args.use_negative_training:
        records = filter_for_annotations(
            records,
            args.annotations_dir,
            require_positive=require_positive,
            require_any=require_any,
            allow_negative_training=args.use_negative_training,
            allowed_orientations=allowed_orientations,
        )
    if not records:
        logger.warning("No records available after filtering. Check labels.csv and annotations.")
        return

    if not args.weights.exists():
        raise FileNotFoundError(f"Segmentation weights not found at {args.weights}.")

    dataset = TubuleSegmentationDataset(
        records,
        args.raw_dir,
        args.annotations_dir,
        max_size=args.max_size,
        augment=False,
        require_annotations=args.require_annotations,
        instance_mode=False,
        allowed_orientations=args.allowed_orientations,
        ignore_band_px=args.ignore_band_px if args.metrics_ignore else 0,
        ignore_band_frac=args.ignore_band_frac if args.metrics_ignore else None,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = create_segmentation_model()
    state = torch.load(args.weights, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)

    dice, iou, precision, recall = evaluate_on_loader(
        model,
        loader,
        device=device,
        threshold=args.threshold,
        save_viz=args.save_viz,
        viz_dir=args.viz_dir,
        max_viz=args.max_viz,
        records=dataset.records,
        postprocess=args.postprocess,
        pp_threshold=args.pp_threshold,
        pp_min_area_frac=args.pp_min_area_frac,
        pp_max_components=args.pp_max_components,
        pp_fill_holes=args.pp_fill_holes,
        pp_max_hole_frac=args.pp_max_hole_frac,
        metrics_ignore=args.metrics_ignore,
    )
    logger.info("Segmentation metrics | dice=%.3f iou=%.3f precision=%.3f recall=%.3f", dice, iou, precision, recall)


if __name__ == "__main__":
    main()
