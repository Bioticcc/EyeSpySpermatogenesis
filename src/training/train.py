from __future__ import annotations

# Phase 2 / future: stage classification training (legacy).

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score

from src.dataset.schema import Quality, StageCluster
from src.dataset.transforms import build_transforms
from src.training.config import TrainConfig
from src.training.model import create_model
from src.utils.logging import get_logger
from src.utils.paths import exported_models_dir
from src.utils.seed import set_seed, worker_init_fn
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Split file missing at {path}. Run src.dataset.splits first.")
    return pd.read_csv(path)


def load_quality_map(labels_path: Path) -> Dict[str, str]:
    df = pd.read_csv(labels_path)
    return {row["rel_path"]: row.get("quality", Quality.UNKNOWN.value) for _, row in df.iterrows()}


class TubuleDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        raw_dir: Path,
        transform,
        quality_map: Dict[str, str],
        low_quality_weight: float = 1.0,
    ):
        self.df = df.reset_index(drop=True)
        self.raw_dir = raw_dir
        self.transform = transform
        self.quality_map = quality_map
        self.low_quality_weight = low_quality_weight
        self.stage_to_idx = {name: idx for idx, name in enumerate(StageCluster.labeled_values())}

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        row = self.df.iloc[idx]
        rel_path = Path(row["rel_path"])
        stage = row["stage_cluster"]
        quality = self.quality_map.get(rel_path.as_posix(), Quality.UNKNOWN.value)
        path = self.raw_dir / rel_path
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")
        array = read_tiff(path)
        tensor = self.transform(array)
        weight = self.low_quality_weight if quality == Quality.LOW.value else 1.0
        return tensor, self.stage_to_idx[stage], torch.tensor(weight, dtype=torch.float32)


def compute_class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df["stage_cluster"].value_counts().reindex(StageCluster.labeled_values()).fillna(0)
    counts = counts.replace(0, 1)
    inv = 1.0 / counts
    weights = inv / inv.sum() * len(counts)
    return torch.tensor(weights.values, dtype=torch.float32)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    losses: List[float] = []
    all_preds: List[int] = []
    all_targets: List[int] = []
    for images, targets, sample_weights in loader:
        images = images.to(device)
        targets = targets.to(device)
        weights = sample_weights.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss = (loss * weights).mean()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = logits.argmax(dim=1).detach().cpu().numpy()
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.cpu().numpy().tolist())
    acc = accuracy_score(all_targets, all_preds)
    return float(np.mean(losses)), float(acc)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float, float]:
    model.eval()
    losses: List[float] = []
    all_preds: List[int] = []
    all_targets: List[int] = []
    with torch.no_grad():
        for images, targets, sample_weights in loader:
            images = images.to(device)
            targets = targets.to(device)
            weights = sample_weights.to(device)
            logits = model(images)
            loss = criterion(logits, targets)
            loss = (loss * weights).mean()
            losses.append(loss.item())
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.cpu().numpy().tolist())
    acc = accuracy_score(all_targets, all_preds) if all_targets else 0.0
    f1 = f1_score(all_targets, all_preds, average="macro") if all_targets else 0.0
    return float(np.mean(losses) if losses else 0.0), float(acc), float(f1)


def save_checkpoint(model: nn.Module, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, path)


def maybe_load_config(path: Path | None) -> Dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline CNN on tubule stages")
    parser.add_argument("--config", type=Path, help="Optional JSON config to override defaults")
    TrainConfig.add_argparse_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_dict = asdict(TrainConfig())
    cfg_dict.update(maybe_load_config(args.config))
    for key, value in vars(args).items():
        if key in cfg_dict and value is not None and key != "config":
            cfg_dict[key] = value
    cfg = TrainConfig.from_args(argparse.Namespace(**cfg_dict))

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    train_df = load_split(cfg.splits_dir / "train.csv")
    val_df = load_split(cfg.splits_dir / "val.csv")
    quality_map = load_quality_map(cfg.labels_path)

    train_df["quality"] = train_df["rel_path"].map(quality_map).fillna(Quality.UNKNOWN.value)
    if not cfg.include_low_quality:
        train_df = train_df[train_df["quality"] != Quality.LOW.value]
    if train_df.empty:
        raise SystemExit("No training data after filtering. Check labels and splits.")

    transform_train = build_transforms(train=True, image_size=cfg.image_size)
    transform_eval = build_transforms(train=False, image_size=cfg.image_size)
    train_ds = TubuleDataset(train_df, cfg.raw_dir, transform_train, quality_map, low_quality_weight=cfg.low_quality_weight)
    val_ds = TubuleDataset(val_df, cfg.raw_dir, transform_eval, quality_map, low_quality_weight=1.0)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
    )

    num_classes = len(StageCluster.labeled_values())
    model = create_model(num_classes=num_classes).to(device)

    class_weights = compute_class_weights(train_df).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)

    best_f1 = -1.0
    best_path = cfg.output_dir / "best.pt"

    for epoch in range(cfg.epochs):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.3f | val_loss=%.4f val_acc=%.3f val_f1=%.3f",
            epoch + 1,
            cfg.epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            val_f1,
        )
        if val_f1 > best_f1:
            best_f1 = val_f1
            save_checkpoint(model, best_path)
            logger.info("New best model saved to %s", best_path)

    if best_path.exists():
        export_path = exported_models_dir() / "model.pt"
        export_path.parent.mkdir(parents=True, exist_ok=True)
        model.load_state_dict(torch.load(best_path)["state_dict"])
        torch.save(model.state_dict(), export_path)
        logger.info("Exported best weights to %s", export_path)
    else:
        logger.warning("Training finished without a saved model. Check logs.")


if __name__ == "__main__":
    main()
