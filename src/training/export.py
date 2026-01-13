from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.dataset.schema import StageCluster
from src.training.model import create_model
from src.utils.logging import get_logger
from src.utils.paths import exported_models_dir

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained weights for inference app")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to training checkpoint (.pt)")
    parser.add_argument("--output", type=Path, default=exported_models_dir() / "model.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_classes = len(StageCluster.labeled_values())
    model = create_model(num_classes=num_classes)
    state = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    logger.info("Exported model weights to %s", args.output)


if __name__ == "__main__":
    main()
