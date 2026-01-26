from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Optional


class StageCluster(str, Enum):
    XII_I = "XII_I"
    II_IV = "II_IV"
    V_VI = "V_VI"
    VII_VIII = "VII_VIII"
    IX_X = "IX_X"
    XI = "XI"
    UNKNOWN = "unknown"

    @classmethod
    def labeled_values(cls) -> List[str]:
        return [cls.XII_I, cls.II_IV, cls.V_VI, cls.VII_VIII, cls.IX_X, cls.XI]


class Quality(str, Enum):
    HIGH = "high"
    LOW = "low"
    UNKNOWN = "unknown"


class IncludeFlag(str, Enum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class NegativeTrainingFlag(str, Enum):
    YES = "yes"
    NO = "no"


class Orientation(str, Enum):
    CROSS_SECTION = "cross_section"
    LONGITUDINAL = "longitudinal"
    OBLIQUE = "oblique"
    UNKNOWN = "unknown"

    @classmethod
    def labeled_values(cls) -> List[str]:
        return [cls.CROSS_SECTION, cls.LONGITUDINAL, cls.OBLIQUE]


DEFAULT_LABEL_CSV_COLUMNS = [
    "image_id",
    "rel_path",
    "quality",
    "include",
    "negative_training",
    "notes",
    "labeled_by",
    "labeled_at",
]

LEGACY_LABEL_CSV_COLUMNS = [
    "image_id",
    "rel_path",
    "stage_cluster",
    "quality",
    "include",
    "negative_training",
    "notes",
    "labeled_by",
    "labeled_at",
]

LABEL_CSV_COLUMNS = DEFAULT_LABEL_CSV_COLUMNS


@dataclass
class LabelRecord:
    image_id: str
    rel_path: str
    stage_cluster: StageCluster = StageCluster.UNKNOWN
    quality: Quality = Quality.UNKNOWN
    include: IncludeFlag = IncludeFlag.UNKNOWN
    negative_training: NegativeTrainingFlag = NegativeTrainingFlag.NO
    notes: str = ""
    labeled_by: str = ""
    labeled_at: str = ""

    def as_row(self) -> Dict[str, str]:
        row = asdict(self)
        row["stage_cluster"] = self.stage_cluster.value
        row["quality"] = self.quality.value
        row["include"] = self.include.value
        row["negative_training"] = self.negative_training.value
        return row

    @staticmethod
    def from_row(row: Dict[str, str]) -> "LabelRecord":
        def safe_enum(value: str, enum_cls, default):
            try:
                return enum_cls(value)
            except Exception:
                return default

        return LabelRecord(
            image_id=row.get("image_id", ""),
            rel_path=row.get("rel_path", ""),
            stage_cluster=safe_enum(row.get("stage_cluster", StageCluster.UNKNOWN), StageCluster, StageCluster.UNKNOWN),
            quality=safe_enum(row.get("quality", Quality.UNKNOWN), Quality, Quality.UNKNOWN),
            include=safe_enum(row.get("include", IncludeFlag.UNKNOWN), IncludeFlag, IncludeFlag.UNKNOWN),
            negative_training=safe_enum(
                row.get("negative_training", NegativeTrainingFlag.NO),
                NegativeTrainingFlag,
                NegativeTrainingFlag.NO,
            ),
            notes=row.get("notes", ""),
            labeled_by=row.get("labeled_by", ""),
            labeled_at=row.get("labeled_at", ""),
        )


def default_label_row(image_path: Path) -> LabelRecord:
    image_id = image_path.stem
    rel_path = str(image_path)
    return LabelRecord(image_id=image_id, rel_path=rel_path)


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def label_csv_columns_for_path(path: Path) -> List[str]:
    if not path.exists():
        return DEFAULT_LABEL_CSV_COLUMNS
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader, [])
    except Exception:
        return DEFAULT_LABEL_CSV_COLUMNS
    if "stage_cluster" in header:
        return LEGACY_LABEL_CSV_COLUMNS
    return DEFAULT_LABEL_CSV_COLUMNS
