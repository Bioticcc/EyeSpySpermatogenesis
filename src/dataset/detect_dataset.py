from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F

from src.dataset.annotations import annotation_path, load_annotation
from src.dataset.indexing import load_labels
from src.dataset.schema import IncludeFlag, LabelRecord, Quality
from src.utils.logging import get_logger
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


def filter_records(
    records: Iterable[LabelRecord],
    include_yes_only: bool = True,
    only_high_quality: bool = False,
) -> List[LabelRecord]:
    filtered: List[LabelRecord] = []
    for record in records:
        if include_yes_only and record.include != IncludeFlag.YES:
            continue
        if only_high_quality and record.quality != Quality.HIGH:
            continue
        filtered.append(record)
    filtered.sort(key=lambda r: r.rel_path)
    return filtered


def load_label_records(labels_path: Path) -> List[LabelRecord]:
    labels = load_labels(labels_path)
    return list(labels.values())


class TubuleDetectDataset(Dataset):
    def __init__(
        self,
        records: List[LabelRecord],
        raw_dir: Path,
        annotations_dir: Path,
        max_size: Optional[int] = None,
        augment: bool = False,
        require_annotations: bool = True,
    ) -> None:
        self.records = records
        self.raw_dir = raw_dir
        self.annotations_dir = annotations_dir
        self.max_size = max_size
        self.augment = augment
        self.require_annotations = require_annotations
        self._missing_warned = set()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        image_path = self.raw_dir / record.rel_path
        if not image_path.exists():
            raise FileNotFoundError(f"Missing file: {image_path}")
        array = read_tiff(image_path)
        image = torch.from_numpy(array).permute(2, 0, 1)

        ann_path = annotation_path(self.annotations_dir, record.image_id)
        annotation = load_annotation(ann_path)
        if annotation is None:
            if self.require_annotations:
                raise FileNotFoundError(f"Missing annotation: {ann_path}")
            if record.image_id not in self._missing_warned:
                logger.warning("Missing annotation for %s; treating as empty.", record.rel_path)
                self._missing_warned.add(record.image_id)
            boxes_xywh: List[List[float]] = []
        else:
            boxes_xywh = [box.bbox_xywh for box in annotation.boxes]

        boxes = self._xywh_to_xyxy(boxes_xywh)
        if self.augment:
            image, boxes = self._random_flips(image, boxes)
        image, boxes = self._maybe_resize(image, boxes)

        labels = torch.ones((boxes.shape[0],), dtype=torch.int64)
        area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64),
        }
        return image, target

    @staticmethod
    def _xywh_to_xyxy(boxes_xywh: List[List[float]]) -> torch.Tensor:
        if not boxes_xywh:
            return torch.zeros((0, 4), dtype=torch.float32)
        boxes = torch.tensor(boxes_xywh, dtype=torch.float32)
        x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        return torch.stack([x, y, x + w, y + h], dim=1)

    @staticmethod
    def _random_flips(image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, height, width = image.shape
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            if boxes.numel() > 0:
                x0 = width - boxes[:, 2]
                x1 = width - boxes[:, 0]
                boxes[:, 0], boxes[:, 2] = x0, x1
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            if boxes.numel() > 0:
                y0 = height - boxes[:, 3]
                y1 = height - boxes[:, 1]
                boxes[:, 1], boxes[:, 3] = y0, y1
        return image, boxes

    def _maybe_resize(self, image: torch.Tensor, boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.max_size:
            return image, boxes
        _, height, width = image.shape
        max_dim = max(height, width)
        if max_dim <= self.max_size:
            return image, boxes
        scale = self.max_size / float(max_dim)
        new_h = int(round(height * scale))
        new_w = int(round(width * scale))
        image = F.resize(image, [new_h, new_w], antialias=True)
        if boxes.numel() > 0:
            boxes = boxes * scale
        return image, boxes


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
