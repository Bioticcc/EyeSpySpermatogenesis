from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.dataset.schema import Quality, StageCluster
from src.dataset.transforms import build_transforms
from src.training.model import create_model
from src.utils.logging import get_logger
from src.utils.paths import exported_models_dir, raw_data_dir, splits_dir, labels_dir
from src.utils.seed import worker_init_fn
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


class EvalDataset(Dataset):
    def __init__(self, df: pd.DataFrame, raw_dir: Path, quality_map: Dict[str, str]):
        self.df = df.reset_index(drop=True)
        self.raw_dir = raw_dir
        self.quality_map = quality_map
        self.transform = build_transforms(train=False)
        self.stage_to_idx = {name: idx for idx, name in enumerate(StageCluster.labeled_values())}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = Path(row["rel_path"])
        path = self.raw_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        array = read_tiff(path)
        tensor = self.transform(array)
        return tensor, self.stage_to_idx[row["stage_cluster"]]


def load_quality_map(labels_path: Path) -> Dict[str, str]:
    df = pd.read_csv(labels_path)
    return {row["rel_path"]: row.get("quality", Quality.UNKNOWN.value) for _, row in df.iterrows()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--split-path", type=Path, default=splits_dir() / "test.csv")
    parser.add_argument("--raw-dir", type=Path, default=raw_data_dir())
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--checkpoint", type=Path, default=exported_models_dir() / "model.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--confusion-output", type=Path, default=Path("models/confusion_matrix.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.split_path.exists():
        raise SystemExit(f"Split file not found at {args.split_path}")
    df = pd.read_csv(args.split_path)
    quality_map = load_quality_map(args.labels_path)
    dataset = EvalDataset(df, args.raw_dir, quality_map)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    num_classes = len(StageCluster.labeled_values())
    model = create_model(num_classes=num_classes)
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found at {args.checkpoint}. Train or export first.")
    state = torch.load(args.checkpoint, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    all_preds: List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.numpy().tolist())
    acc = accuracy_score(all_targets, all_preds) if all_targets else 0.0
    f1 = f1_score(all_targets, all_preds, average="macro") if all_targets else 0.0
    cm = confusion_matrix(all_targets, all_preds, labels=list(range(num_classes)))
    logger.info("Eval accuracy=%.3f macro_f1=%.3f", acc, f1)
    args.confusion_output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.confusion_output, cm, delimiter=",", fmt="%d")
    logger.info("Saved confusion matrix to %s", args.confusion_output)


if __name__ == "__main__":
    main()
