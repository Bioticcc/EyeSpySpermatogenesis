from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.dataset.schema import now_iso


@dataclass
class BoxAnnotation:
    box_id: str
    bbox_xywh: List[float]
    label: str = "tubule"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box_id": self.box_id,
            "bbox_xywh": [float(v) for v in self.bbox_xywh],
            "label": self.label,
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "BoxAnnotation":
        return BoxAnnotation(
            box_id=str(data.get("box_id", "")),
            bbox_xywh=[float(v) for v in data.get("bbox_xywh", [0, 0, 0, 0])],
            label=str(data.get("label", "tubule")),
            notes=str(data.get("notes", "")),
        )


@dataclass
class ImageAnnotation:
    image_id: str
    rel_path: str
    tissue: Optional[str] = None
    boxes: List[BoxAnnotation] = field(default_factory=list)
    labeled_by: str = ""
    labeled_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "rel_path": self.rel_path,
            "tissue": self.tissue,
            "boxes": [box.to_dict() for box in self.boxes],
            "labeled_by": self.labeled_by,
            "labeled_at": self.labeled_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ImageAnnotation":
        return ImageAnnotation(
            image_id=str(data.get("image_id", "")),
            rel_path=str(data.get("rel_path", "")),
            tissue=data.get("tissue"),
            boxes=[BoxAnnotation.from_dict(b) for b in data.get("boxes", [])],
            labeled_by=str(data.get("labeled_by", "")),
            labeled_at=str(data.get("labeled_at", "")),
        )


def new_box_id() -> str:
    return uuid.uuid4().hex


def annotation_path(annotations_dir: Path, image_id: str) -> Path:
    return annotations_dir / f"{image_id}.json"


def load_annotation(path: Path) -> Optional[ImageAnnotation]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return ImageAnnotation.from_dict(data)


def save_annotation(annotation: ImageAnnotation, path: Path) -> None:
    if not annotation.labeled_at:
        annotation.labeled_at = now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(annotation.to_dict(), f, indent=2)
