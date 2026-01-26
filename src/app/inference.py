from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.ops import nms

from src.dataset.masks import boxes_from_instance_map, split_mask_instances
from src.training.detect_model import create_detector
from src.training.seg_model import create_segmentation_model
from src.utils.logging import get_logger
from src.utils.postprocess import extract_components, postprocess_mask, threshold_probs
from src.utils.paths import exported_models_dir
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


def load_detector(model_path: Path, device: Optional[torch.device] = None) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Detector weights not found at {model_path}. Train and export first.")
    model = create_detector(num_classes=2)
    state = torch.load(model_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    device = device or torch.device("cpu")
    model.to(device)
    model.eval()
    return model


def load_segmenter(model_path: Path, device: Optional[torch.device] = None) -> torch.nn.Module:
    if not model_path.exists():
        raise FileNotFoundError(f"Segmentation weights not found at {model_path}. Train and export first.")
    model = create_segmentation_model()
    state = torch.load(model_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    device = device or torch.device("cpu")
    model.to(device)
    model.eval()
    return model


def _prepare_image(array: np.ndarray, max_size: Optional[int]) -> tuple[torch.Tensor, float]:
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    _, height, width = tensor.shape
    scale = 1.0
    if max_size and max(height, width) > max_size:
        scale = max_size / float(max(height, width))
        new_h = int(round(height * scale))
        new_w = int(round(width * scale))
        tensor = TF.resize(tensor, [new_h, new_w], antialias=True)
    return tensor, scale


def predict_image(
    image_path: Path,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    score_threshold: float = 0.25,
    nms_iou: float = 0.5,
    max_size: Optional[int] = 1024,
) -> Dict:
    array = read_tiff(image_path)
    tensor, scale = _prepare_image(array, max_size=max_size)
    device = device or torch.device("cpu")
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model([tensor])[0]
    scores = output["scores"].cpu()
    boxes = output["boxes"].cpu()
    keep = scores >= score_threshold
    scores = scores[keep]
    boxes = boxes[keep]
    if boxes.numel() > 0 and nms_iou > 0:
        keep_idx = nms(boxes, scores, nms_iou)
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
    if scale != 1.0 and boxes.numel() > 0:
        boxes = boxes / scale
    boxes_xywh = _xyxy_to_xywh(boxes.numpy().tolist())
    detections = [
        {"bbox_xywh": box, "score": float(score), "label": "tubule"}
        for box, score in zip(boxes_xywh, scores.numpy().tolist())
    ]
    return {
        "image": image_path.name,
        "detections": detections,
    }


def _xyxy_to_xywh(boxes: List[List[float]]) -> List[List[float]]:
    converted: List[List[float]] = []
    for x1, y1, x2, y2 in boxes:
        converted.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])
    return converted


def mask_to_boxes(mask: np.ndarray, probs: Optional[np.ndarray] = None, min_area: int = 20) -> List[Dict]:
    height, width = mask.shape
    visited = np.zeros((height, width), dtype=bool)
    boxes: List[Dict] = []
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            prob_sum = 0.0
            while stack:
                cy, cx = stack.pop()
                area += 1
                if probs is not None:
                    prob_sum += float(probs[cy, cx])
                min_x = min(min_x, cx)
                max_x = max(max_x, cx)
                min_y = min(min_y, cy)
                max_y = max(max_y, cy)
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            if area < min_area:
                continue
            score = prob_sum / area if probs is not None else 1.0
            boxes.append(
                {
                    "bbox_xywh": [float(min_x), float(min_y), float(max_x - min_x + 1), float(max_y - min_y + 1)],
                    "score": float(score),
                    "label": "tubule",
                }
            )
    return boxes


def _components_to_detections(components: List[dict]) -> Tuple[List[Dict], List[Tuple[int, int, int, int]]]:
    detections: List[Dict] = []
    bboxes: List[Tuple[int, int, int, int]] = []
    for comp in components:
        x1, y1, x2, y2 = comp["bbox"]
        score = float(comp.get("mean_prob", 1.0))
        detections.append(
            {
                "bbox_xywh": [float(x1), float(y1), float(x2 - x1 + 1), float(y2 - y1 + 1)],
                "score": score,
                "label": "tubule",
            }
        )
        bboxes.append((int(x1), int(y1), int(x2), int(y2)))
    return detections, bboxes


def predict_segmentation(
    image_path: Path,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    threshold: float = 0.5,
    max_size: Optional[int] = 1024,
    split_instances: bool = False,
    instance_erosion_iters: int = 2,
    instance_min_area: int = 20,
    instance_seed_min_area: int = 5,
    postprocess: bool = False,
    pp_threshold: Optional[float] = None,
    pp_min_area_frac: float = 0.01,
    pp_max_components: int = 4,
    pp_fill_holes: bool = True,
    pp_max_hole_frac: float = 0.005,
) -> Dict:
    array = read_tiff(image_path)
    tensor, scale = _prepare_image(array, max_size=max_size)
    device = device or torch.device("cpu")
    tensor = tensor.to(device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0, 0]
    if scale != 1.0:
        probs = F.interpolate(probs.unsqueeze(0).unsqueeze(0), size=array.shape[:2], mode="bilinear", align_corners=False)[
            0, 0
        ]
    probs_np = probs.cpu().numpy()
    threshold_for_post = threshold if pp_threshold is None else float(pp_threshold)
    raw_mask = threshold_probs(probs_np, threshold_for_post if postprocess else threshold)
    final_mask = raw_mask
    components: Optional[List[dict]] = None
    if postprocess:
        height, width = raw_mask.shape
        min_area_px = 0
        if pp_min_area_frac and pp_min_area_frac > 0:
            min_area_px = int(round(pp_min_area_frac * height * width))
            min_area_px = max(min_area_px, 1)
        max_hole_px = 0
        if pp_max_hole_frac and pp_max_hole_frac > 0:
            max_hole_px = int(round(pp_max_hole_frac * height * width))
            max_hole_px = max(max_hole_px, 1)
        final_mask, components = postprocess_mask(
            raw_mask,
            min_area_px=min_area_px,
            max_components=pp_max_components,
            fill_holes=pp_fill_holes,
            max_hole_area_px=max_hole_px,
            probs=probs_np,
        )
    instance_map = None
    if split_instances:
        instance_map = split_mask_instances(
            final_mask,
            erosion_iters=instance_erosion_iters,
            min_area=instance_min_area,
            seed_min_area=instance_seed_min_area,
        )
        boxes = boxes_from_instance_map(instance_map, probs=probs_np, min_area=instance_min_area)
        tubule_bboxes = []
        for det in boxes:
            x, y, w, h = det["bbox_xywh"]
            tubule_bboxes.append((int(x), int(y), int(x + w - 1), int(y + h - 1)))
    else:
        if components is None:
            components = extract_components(final_mask, probs=probs_np)
        if instance_min_area and instance_min_area > 0:
            components = [comp for comp in components if comp["area"] >= int(instance_min_area)]
        boxes, tubule_bboxes = _components_to_detections(components)
    return {
        "image": image_path.name,
        "mask": final_mask,
        "raw_mask": raw_mask,
        "prob": probs_np,
        "instance_map": instance_map,
        "detections": boxes,
        "tubule_bboxes": tubule_bboxes,
        "threshold": float(threshold),
        "postprocess": bool(postprocess),
    }


def save_results_json(results: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_results_csv(results: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "x", "y", "w", "h", "score"])
        for result in results:
            image_name = result["image"]
            for det in result.get("detections", []):
                x, y, w, h = det["bbox_xywh"]
                writer.writerow([image_name, f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}", f"{det['score']:.4f}"])


def save_mask_png(mask: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_instance_map_png(instance_map: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(instance_map.astype(np.uint16), mode="I;16").save(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tubule detection or segmentation on TIFF images")
    parser.add_argument("images", type=Path, nargs="+", help="Path(s) to TIFF images")
    parser.add_argument("--task", type=str, choices=["seg", "detect"], default="seg")
    parser.add_argument("--weights", type=Path, help="Path to exported weights")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--split-instances", action="store_true", help="Split segmentation mask into instances")
    parser.add_argument(
        "--instance-erosion-iters",
        type=int,
        default=2,
        help="Erosion iterations used to generate instance seeds",
    )
    parser.add_argument("--instance-min-area", type=int, default=20, help="Minimum pixels per instance")
    parser.add_argument("--instance-seed-min-area", type=int, default=5, help="Minimum seed component size")
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV output path")
    parser.add_argument("--output-mask-dir", type=Path, help="Directory to save segmentation masks as PNGs")
    parser.add_argument("--output-instance-mask-dir", type=Path, help="Directory to save instance maps as PNGs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    if args.task == "detect":
        weights = args.weights or (exported_models_dir() / "detector.pt")
        model = load_detector(weights, device=device)
        results = []
        for image in args.images:
            result = predict_image(
                image,
                model,
                device=device,
                score_threshold=args.score_threshold,
                nms_iou=args.nms_iou,
                max_size=args.max_size,
            )
            results.append(result)
            logger.info("%s -> %d detections", image, len(result["detections"]))
        if args.output_json:
            save_results_json(results, args.output_json)
            logger.info("Saved JSON to %s", args.output_json)
        if args.output_csv:
            save_results_csv(results, args.output_csv)
            logger.info("Saved CSV to %s", args.output_csv)
        return

    weights = args.weights or (exported_models_dir() / "seg_model.pt")
    model = load_segmenter(weights, device=device)
    results = []
    for image in args.images:
        result = predict_segmentation(
            image,
            model,
            device=device,
            threshold=args.threshold,
            max_size=args.max_size,
            split_instances=args.split_instances,
            instance_erosion_iters=args.instance_erosion_iters,
            instance_min_area=args.instance_min_area,
            instance_seed_min_area=args.instance_seed_min_area,
        )
        results.append({
            "image": result["image"],
            "detections": result["detections"],
            "threshold": result["threshold"],
        })
        logger.info("%s -> %d instances", image, len(result["detections"]))
        if args.output_mask_dir:
            mask_path = args.output_mask_dir / f"{image.stem}_mask.png"
            save_mask_png(result["mask"], mask_path)
        if args.output_instance_mask_dir and result.get("instance_map") is not None:
            mask_path = args.output_instance_mask_dir / f"{image.stem}_instances.png"
            save_instance_map_png(result["instance_map"], mask_path)
    if args.output_json:
        save_results_json(results, args.output_json)
        logger.info("Saved JSON to %s", args.output_json)
    if args.output_csv:
        save_results_csv(results, args.output_csv)
        logger.info("Saved CSV to %s", args.output_csv)


if __name__ == "__main__":
    main()


# Phase 2 / future: legacy stage classification inference (kept for reference).
def phase2_load_stage_model(model_path: Path, device: Optional[torch.device] = None) -> torch.nn.Module:
    from src.dataset.schema import StageCluster
    from src.training.model import create_model

    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found at {model_path}. Train and export first.")
    num_classes = len(StageCluster.labeled_values())
    model = create_model(num_classes=num_classes)
    state = torch.load(model_path, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    device = device or torch.device("cpu")
    model.to(device)
    model.eval()
    return model


def phase2_predict_stage(
    image_path: Path,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    image_size: int = 224,
) -> Dict:
    from src.dataset.schema import StageCluster
    from src.dataset.transforms import build_transforms

    array = read_tiff(image_path)
    transform = build_transforms(train=False, image_size=image_size)
    tensor = transform(array).unsqueeze(0)
    device = device or torch.device("cpu")
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
    probs_np = probs.cpu().numpy()
    labels = StageCluster.labeled_values()
    top_indices = probs_np.argsort()[::-1][:3]
    top3 = [(labels[i], float(probs_np[i])) for i in top_indices]
    return {
        "image": image_path.name,
        "predicted": labels[int(top_indices[0])],
        "confidence": float(probs_np[top_indices[0]]),
        "top3": top3,
    }
