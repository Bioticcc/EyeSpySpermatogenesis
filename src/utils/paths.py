from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    path = repo_root() / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path


def raw_data_dir() -> Path:
    path = data_dir() / "raw"
    path.mkdir(parents=True, exist_ok=True)
    return path


def processed_data_dir() -> Path:
    path = data_dir() / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def splits_dir() -> Path:
    path = data_dir() / "splits"
    path.mkdir(parents=True, exist_ok=True)
    return path


def labels_dir() -> Path:
    path = data_dir() / "labels"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir() -> Path:
    path = repo_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def exported_models_dir() -> Path:
    path = models_dir() / "exported"
    path.mkdir(parents=True, exist_ok=True)
    return path
