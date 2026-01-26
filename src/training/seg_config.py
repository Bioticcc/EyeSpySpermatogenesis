from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from src.utils.paths import exported_models_dir


@dataclass
class SegConfig:
    epochs: int = 10
    batch_size: int = 2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_ratio: float = 0.2
    max_size: int = 1024
    threshold: float = 0.5
    export_path: Path = exported_models_dir() / "seg_model.pt"
    allowed_orientations: Optional[List[str]] = None


DEFAULT_SEG_CONFIG = SegConfig()
