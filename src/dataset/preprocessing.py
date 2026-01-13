from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from src.dataset.schema import IncludeFlag, StageCluster
from src.dataset.transforms import build_transforms
from src.utils.logging import get_logger
from src.utils.paths import labels_dir, processed_data_dir, raw_data_dir
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


def preprocess(labels_path: Path, raw_dir: Path, output_dir: Path, image_size: int = 224) -> None:
    if not labels_path.exists():
        raise SystemExit(f"Labels file not found at {labels_path}")
    df = pd.read_csv(labels_path)
    df = df[(df["include"] == IncludeFlag.YES.value) & (df["stage_cluster"].isin(StageCluster.labeled_values()))]
    transform = build_transforms(train=False, image_size=image_size)
    for _, row in df.iterrows():
        rel_path = Path(row["rel_path"])
        src_path = raw_dir / rel_path
        if not src_path.exists():
            logger.warning("Skipping missing file %s", src_path)
            continue
        array = read_tiff(src_path)
        tensor = transform(array)
        out_path = output_dir / rel_path.with_suffix(".pt")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, out_path)
        logger.debug("Saved %s", out_path)
    logger.info("Preprocessing finished. Cached tensors in %s", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess TIFFs and cache tensors")
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir())
    parser.add_argument("--output-dir", type=Path, default=processed_data_dir())
    parser.add_argument("--image-size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preprocess(args.labels_path, args.raw_dir, args.output_dir, args.image_size)


if __name__ == "__main__":
    main()
