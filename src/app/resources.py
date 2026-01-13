from __future__ import annotations

import sys
from pathlib import Path


def base_path() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[2]


def bundled_model_path() -> Path:
    # In dev: project_root/models/exported/model.pt
    # In PyInstaller bundle: packaged alongside executable under models/exported
    return base_path() / "models" / "exported" / "model.pt"
