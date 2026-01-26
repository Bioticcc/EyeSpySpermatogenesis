from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.training.seg_model import create_segmentation_model
from src.utils.logging import get_logger
from src.utils.paths import exported_models_dir

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export segmentation weights for inference app")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to segmentation checkpoint (.pt)")
    parser.add_argument("--output", type=Path, default=exported_models_dir() / "seg_model.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = create_segmentation_model()
    state = torch.load(args.checkpoint, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    logger.info("Exported segmentation weights to %s", args.output)


if __name__ == "__main__":
    main()
