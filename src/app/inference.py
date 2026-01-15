from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torchvision.transforms.functional as F
from torchvision.ops import nms

from src.training.detect_model import create_detector
from src.utils.logging import get_logger
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


def _prepare_image(array, max_size: Optional[int]) -> tuple[torch.Tensor, float]:
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    _, height, width = tensor.shape
    scale = 1.0
    if max_size and max(height, width) > max_size:
        scale = max_size / float(max(height, width))
        new_h = int(round(height * scale))
        new_w = int(round(width * scale))
        tensor = F.resize(tensor, [new_h, new_w], antialias=True)
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
        {"bbox_xywh": box, "score": float(score), "label": "tubule"} for box, score in zip(boxes_xywh, scores.numpy().tolist())
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
            for det in result["detections"]:
                x, y, w, h = det["bbox_xywh"]
                writer.writerow([image_name, f"{x:.2f}", f"{y:.2f}", f"{w:.2f}", f"{h:.2f}", f"{det['score']:.4f}"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tubule detection on TIFF images")
    parser.add_argument("images", type=Path, nargs="+", help="Path(s) to TIFF images")
    parser.add_argument("--weights", type=Path, required=True, help="Path to exported detector weights")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--nms-iou", type=float, default=0.5)
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--output-json", type=Path, help="Optional JSON output path")
    parser.add_argument("--output-csv", type=Path, help="Optional CSV output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_detector(args.weights, device=device)
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
