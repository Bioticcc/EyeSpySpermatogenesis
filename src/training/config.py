from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from src.utils.paths import labels_dir, raw_data_dir, splits_dir, models_dir


@dataclass
class TrainConfig:
    labels_path: Path = labels_dir() / "labels.csv"
    raw_dir: Path = raw_data_dir()
    splits_dir: Path = splits_dir()
    output_dir: Path = models_dir() / "checkpoints"
    image_size: int = 224
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    seed: int = 42
    include_low_quality: bool = False
    low_quality_weight: float = 0.5
    device: str = "cpu"

    @staticmethod
    def add_argparse_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--labels-path", type=Path, default=TrainConfig.labels_path)
        parser.add_argument("--raw-dir", type=Path, default=TrainConfig.raw_dir)
        parser.add_argument("--splits-dir", type=Path, default=TrainConfig.splits_dir)
        parser.add_argument("--output-dir", type=Path, default=TrainConfig.output_dir)
        parser.add_argument("--image-size", type=int, default=TrainConfig.image_size)
        parser.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
        parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
        parser.add_argument("--learning-rate", type=float, default=TrainConfig.learning_rate)
        parser.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
        parser.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
        parser.add_argument("--seed", type=int, default=TrainConfig.seed)
        parser.add_argument("--include-low-quality", action="store_true", help="Include low-quality images during training")
        parser.add_argument("--low-quality-weight", type=float, default=TrainConfig.low_quality_weight)
        parser.add_argument("--device", type=str, default=TrainConfig.device)

    @staticmethod
    def from_args(args: argparse.Namespace) -> "TrainConfig":
        return TrainConfig(
            labels_path=Path(args.labels_path),
            raw_dir=Path(args.raw_dir),
            splits_dir=Path(args.splits_dir),
            output_dir=Path(args.output_dir),
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed,
            include_low_quality=args.include_low_quality,
            low_quality_weight=args.low_quality_weight,
            device=args.device,
        )
