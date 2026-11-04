from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.dataset.schema import IncludeFlag, Orientation, Quality, now_iso


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


@dataclass
class TubulePolygon:
    tubule_id: str
    polygon_xy: List[List[float]]
    orientation: Orientation = Orientation.UNKNOWN
    partial: bool = False
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tubule_id": self.tubule_id,
            "polygon_xy": [[float(x), float(y)] for x, y in self.polygon_xy],
            "orientation": self.orientation.value,
            "partial": bool(self.partial),
            "notes": self.notes,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "TubulePolygon":
        points = data.get("polygon_xy", [])
        polygon_xy = [[float(x), float(y)] for x, y in points] if isinstance(points, list) else []
        tubule_id = str(data.get("tubule_id", ""))
        orientation_value = data.get("orientation", Orientation.UNKNOWN.value)
        try:
            orientation = Orientation(str(orientation_value))
        except Exception as exc:
            allowed = ", ".join([o.value for o in Orientation])
            raise ValueError(
                f"Invalid orientation '{orientation_value}' for tubule '{tubule_id}'. Allowed: {allowed}."
            ) from exc
        return TubulePolygon(
            tubule_id=tubule_id,
            polygon_xy=polygon_xy,
            orientation=orientation,
            partial=bool(data.get("partial", False)),
            notes=str(data.get("notes", "")),
        )


@dataclass
class PolygonAnnotation:
    image_id: str
    rel_path: str
    include: IncludeFlag = IncludeFlag.UNKNOWN
    quality: Quality = Quality.UNKNOWN
    tubules: List[TubulePolygon] = field(default_factory=list)
    labeled_by: str = ""
    labeled_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "rel_path": self.rel_path,
            "include": self.include.value,
            "quality": self.quality.value,
            "tubules": [tubule.to_dict() for tubule in self.tubules],
            "labeled_by": self.labeled_by,
            "labeled_at": self.labeled_at,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "PolygonAnnotation":
        def safe_enum(value: str, enum_cls, default):
            try:
                return enum_cls(value)
            except Exception:
                return default

        return PolygonAnnotation(
            image_id=str(data.get("image_id", "")),
            rel_path=str(data.get("rel_path", "")),
            include=safe_enum(data.get("include", IncludeFlag.UNKNOWN.value), IncludeFlag, IncludeFlag.UNKNOWN),
            quality=safe_enum(data.get("quality", Quality.UNKNOWN.value), Quality, Quality.UNKNOWN),
            tubules=[TubulePolygon.from_dict(t) for t in data.get("tubules", []) if isinstance(t, dict)],
            labeled_by=str(data.get("labeled_by", "")),
            labeled_at=str(data.get("labeled_at", "")),
        )


def new_tubule_id() -> str:
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


def load_polygon_annotation(path: Path) -> Optional[PolygonAnnotation]:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return None
    return PolygonAnnotation.from_dict(data)


def save_polygon_annotation(annotation: PolygonAnnotation, path: Path) -> None:
    if not annotation.labeled_at:
        annotation.labeled_at = now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(annotation.to_dict(), f, indent=2)
