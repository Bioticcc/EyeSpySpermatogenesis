from __future__ import annotations

import sys
from pathlib import Path


def base_path() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


def bundled_detector_path() -> Path:
    # In dev: project_root/models/exported/detector.pt
    # In PyInstaller bundle: packaged alongside executable under models/exported
    return base_path() / "models" / "exported" / "detector.pt"


def bundled_stage_model_path() -> Path:
    # Phase 2 / legacy classifier weights
    return base_path() / "models" / "exported" / "model.pt"


def bundled_seg_model_path() -> Path:
    # Phase 1 segmentation weights
    return base_path() / "models" / "exported" / "seg_model.pt"
