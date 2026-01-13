from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.dataset.schema import IncludeFlag, Quality, StageCluster
from src.utils.logging import get_logger
from src.utils.paths import labels_dir, splits_dir

logger = get_logger(__name__)


def load_labels_df(labels_path: Path) -> pd.DataFrame:
    if not labels_path.exists():
        raise SystemExit(f"Labels file not found at {labels_path}. Run indexing first.")
    df = pd.read_csv(labels_path)
    return df


def filter_df(df: pd.DataFrame, include_low_quality: bool = False) -> pd.DataFrame:
    df = df[df["include"] == IncludeFlag.YES.value]
    if include_low_quality:
        df = df[df["quality"].isin([Quality.HIGH.value, Quality.LOW.value])]
    else:
        df = df[df["quality"] == Quality.HIGH.value]
    df = df[df["stage_cluster"].isin(StageCluster.labeled_values())]
    return df.reset_index(drop=True)


def stratified_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if df.empty:
        raise SystemExit("No labeled data available after filtering. Check labels.csv.")
    try:
        train_df, temp_df = train_test_split(
            df,
            test_size=1 - train_ratio,
            stratify=df["stage_cluster"],
            random_state=seed,
            shuffle=True,
        )
        relative_val = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=1 - relative_val,
            stratify=temp_df["stage_cluster"],
            random_state=seed,
            shuffle=True,
        )
    except ValueError as exc:
        logger.warning("Stratified split failed (%s). Falling back to random split.", exc)
        train_df, temp_df = train_test_split(df, test_size=1 - train_ratio, random_state=seed, shuffle=True)
        relative_val = val_ratio / (1 - train_ratio)
        val_df, test_df = train_test_split(temp_df, test_size=1 - relative_val, random_state=seed, shuffle=True)
    return train_df, val_df, test_df


def save_split(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df[["rel_path", "stage_cluster"]].to_csv(path, index=False)
    logger.info("Saved %d entries to %s", len(df), path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified train/val/test splits")
    parser.add_argument("--labels-path", type=Path, default=labels_dir() / "labels.csv")
    parser.add_argument("--output-dir", type=Path, default=splits_dir())
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-low-quality", action="store_true", help="Include low-quality images in the pool")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_labels_df(args.labels_path)
    df = filter_df(df, include_low_quality=args.include_low_quality)
    train_df, val_df, test_df = stratified_split(df, args.train_ratio, args.val_ratio, args.seed)
    save_split(train_df, args.output_dir / "train.csv")
    save_split(val_df, args.output_dir / "val.csv")
    save_split(test_df, args.output_dir / "test.csv")


if __name__ == "__main__":
    main()
