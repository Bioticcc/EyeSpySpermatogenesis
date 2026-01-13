from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from src.dataset.schema import StageCluster
from src.dataset.transforms import build_transforms
from src.training.model import create_model
from src.utils.logging import get_logger
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


def load_model(model_path: Path, device: torch.device | None = None) -> torch.nn.Module:
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


def predict_image(
    image_path: Path,
    model: torch.nn.Module,
    device: torch.device | None = None,
    image_size: int = 224,
) -> Dict:
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on TIFF images")
    parser.add_argument("images", type=Path, nargs="+", help="Path(s) to TIFF images")
    parser.add_argument("--weights", type=Path, required=True, help="Path to exported model weights")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model = load_model(args.weights, device=device)
    for image in args.images:
        result = predict_image(image, model, device=device, image_size=args.image_size)
        logger.info("%s -> %s (%.3f)", image, result["predicted"], result["confidence"])
        for label, prob in result["top3"]:
            logger.info("  %s: %.3f", label, prob)


if __name__ == "__main__":
    main()
