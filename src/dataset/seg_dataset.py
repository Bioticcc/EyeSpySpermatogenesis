from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as torchF
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode

from src.dataset.annotations import annotation_path, load_polygon_annotation
from src.dataset.masks import (
    build_ignore_target_mask,
    rasterize_instances,
    rasterize_tubule_masks,
    resolve_pos_neg_conflicts,
    split_tubule_polygons,
)
from src.dataset.indexing import load_labels
from src.dataset.schema import IncludeFlag, LabelRecord, Orientation, Quality
from src.utils.logging import get_logger
from src.utils.tiff_io import read_tiff

logger = get_logger(__name__)


def _polygon_bounds(polygons: List[List[List[float]]], height: int, width: int) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], bool]:
    xs: List[float] = []
    ys: List[float] = []
    out_of_bounds = False
    for polygon in polygons:
        for pair in polygon:
            try:
                x = float(pair[0])
                y = float(pair[1])
            except Exception:
                continue
            xs.append(x)
            ys.append(y)
            if x < 0 or x > width - 1 or y < 0 or y > height - 1:
                out_of_bounds = True
    if not xs or not ys:
        return None, None, None, None, out_of_bounds
    return min(xs), max(xs), min(ys), max(ys), out_of_bounds


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


class TubuleSegmentationDataset(Dataset):
    def __init__(
        self,
        records: List[LabelRecord],
        raw_dir: Path,
        annotations_dir: Path,
        max_size: Optional[int] = None,
        augment: bool = False,
        require_annotations: bool = True,
        instance_mode: bool = False,
        allowed_orientations: Optional[List[Orientation | str]] = None,
        pad_to_multiple: Optional[int] = 4,
        ignore_band_px: int = 0,
        ignore_band_frac: Optional[float] = None,
    ) -> None:
        self.records = records
        self.raw_dir = raw_dir
        self.annotations_dir = annotations_dir
        self.max_size = max_size
        self.augment = augment
        self.require_annotations = require_annotations
        self.instance_mode = instance_mode
        self.allowed_orientations = self._normalize_orientations(allowed_orientations)
        self.pad_to_multiple = pad_to_multiple
        self.ignore_band_px = ignore_band_px
        self.ignore_band_frac = ignore_band_frac
        self._missing_warned = set()
        self._stats_cache: Optional[dict] = None

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
        annotation = load_polygon_annotation(ann_path)
        if annotation is None:
            if self.require_annotations:
                raise FileNotFoundError(f"Missing annotation: {ann_path}")
            if record.image_id not in self._missing_warned:
                logger.warning("Missing segmentation annotation for %s; treating as empty.", record.rel_path)
                self._missing_warned.add(record.image_id)
            tubules = []
        else:
            tubules = annotation.tubules
            if self.allowed_orientations is not None:
                tubules = [t for t in tubules if t.orientation in self.allowed_orientations]

        height, width = array.shape[0], array.shape[1]
        pos_mask, neg_mask, _ = rasterize_tubule_masks(
            tubules,
            height,
            width,
            allowed_orientations=self.allowed_orientations,
        )
        neg_mask, overlap_px = resolve_pos_neg_conflicts(pos_mask, neg_mask)
        if overlap_px > 0:
            logger.warning(
                "Conflicting pos/neg tubule masks resolved (pos wins) image_id=%s overlap_px=%d",
                record.image_id,
                overlap_px,
            )
        gt_pos_frac = float(pos_mask.mean()) if pos_mask.size else 0.0
        if gt_pos_frac > 0.95:
            pos_polygons, _, _ = split_tubule_polygons(tubules, allowed_orientations=self.allowed_orientations)
            min_x, max_x, min_y, max_y, out_of_bounds = _polygon_bounds(pos_polygons, height, width)
            logger.warning(
                "Large GT mask after rasterization image_id=%s size=%dx%d gt_pos_frac=%.4f "
                "polygon_bounds=(%s,%s,%s,%s) any_points_outside=%s",
                record.image_id,
                height,
                width,
                gt_pos_frac,
                f"{min_x:.2f}" if min_x is not None else "NA",
                f"{max_x:.2f}" if max_x is not None else "NA",
                f"{min_y:.2f}" if min_y is not None else "NA",
                f"{max_y:.2f}" if max_y is not None else "NA",
                out_of_bounds,
            )
        if self.instance_mode:
            pos_polygons, _, _ = split_tubule_polygons(tubules, allowed_orientations=self.allowed_orientations)
            instance_mask = rasterize_instances(pos_polygons, height, width, return_list=False)
            mask = torch.from_numpy(instance_mask.astype(np.int64))
        else:
            ignore_radius = self._ignore_radius(height, width)
            if ignore_radius > 0:
                target = build_ignore_target_mask(pos_mask, ignore_radius)
            else:
                target = pos_mask.astype(np.int8)
            if neg_mask.any():
                target[(target == -1) & (neg_mask > 0)] = 0
            mask = torch.from_numpy(target.astype(np.float32))
            neg_mask_tensor = torch.from_numpy(neg_mask.astype(np.float32))
            mask = torch.stack([mask, neg_mask_tensor], dim=0)

        if self.augment:
            image, mask = self._random_flips(image, mask)
        image, mask = self._maybe_resize(image, mask)
        image, mask = self._pad_to_multiple(image, mask)

        if self.instance_mode:
            return image, mask
        target = mask[0]
        neg_mask_out = mask[1]
        return image, target.unsqueeze(0), neg_mask_out.unsqueeze(0)

    @staticmethod
    def _random_flips(image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if random.random() < 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1 if mask.ndim == 2 else 2])
        if random.random() < 0.5:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[0 if mask.ndim == 2 else 1])
        return image, mask

    def _maybe_resize(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.max_size:
            return image, mask
        _, height, width = image.shape
        max_dim = max(height, width)
        if max_dim <= self.max_size:
            return image, mask
        scale = self.max_size / float(max_dim)
        new_h = int(round(height * scale))
        new_w = int(round(width * scale))
        image = F.resize(image, [new_h, new_w], interpolation=InterpolationMode.BILINEAR, antialias=True)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
            mask = F.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
            mask = mask.squeeze(0)
        else:
            mask = F.resize(mask, [new_h, new_w], interpolation=InterpolationMode.NEAREST)
        return image, mask

    def _pad_to_multiple(self, image: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.pad_to_multiple or self.pad_to_multiple <= 1:
            return image, mask
        _, height, width = image.shape
        pad_h = (self.pad_to_multiple - (height % self.pad_to_multiple)) % self.pad_to_multiple
        pad_w = (self.pad_to_multiple - (width % self.pad_to_multiple)) % self.pad_to_multiple
        if pad_h == 0 and pad_w == 0:
            return image, mask
        pad = (0, pad_w, 0, pad_h)
        image = torchF.pad(image, pad, value=0.0)
        if mask.ndim == 2:
            mask = torchF.pad(mask, pad, value=0)
        else:
            mask = torchF.pad(mask, pad, value=0)
        return image, mask

    def _ignore_radius(self, height: int, width: int) -> int:
        if self.ignore_band_frac is not None:
            if self.ignore_band_frac <= 0:
                return 0
            radius = int(round(min(height, width) * self.ignore_band_frac))
            return max(radius, 1)
        return max(int(self.ignore_band_px), 0)

    def summarize(self, name: str = "dataset") -> dict:
        if self._stats_cache is not None:
            return self._stats_cache
        counts = {
            "pos_full_cross_section": 0,
            "partial_cross_section": 0,
            "neg_longitudinal": 0,
            "neg_oblique": 0,
        }
        pos_fracs: List[float] = []
        neg_fracs: List[float] = []
        bg_fracs: List[float] = []
        ignore_fracs: List[float] = []
        included = 0
        for record in self.records:
            image_path = self.raw_dir / record.rel_path
            if not image_path.exists():
                logger.warning("Missing image for %s; skipping stats.", record.rel_path)
                continue
            array = read_tiff(image_path)
            height, width = array.shape[0], array.shape[1]
            ann_path = annotation_path(self.annotations_dir, record.image_id)
            annotation = load_polygon_annotation(ann_path)
            if annotation is None:
                if self.require_annotations:
                    logger.warning("Missing annotation for %s; skipping stats.", record.rel_path)
                    continue
                tubules = []
            else:
                tubules = annotation.tubules
            pos_mask, neg_mask, mask_counts = rasterize_tubule_masks(
                tubules,
                height,
                width,
                allowed_orientations=self.allowed_orientations,
            )
            neg_mask, overlap_px = resolve_pos_neg_conflicts(pos_mask, neg_mask)
            if overlap_px > 0:
                logger.warning(
                    "Conflicting pos/neg tubule masks resolved (pos wins) image_id=%s overlap_px=%d",
                    record.image_id,
                    overlap_px,
                )
            for key, value in mask_counts.items():
                counts[key] += value
            ignore_radius = self._ignore_radius(height, width)
            if ignore_radius > 0:
                target = build_ignore_target_mask(pos_mask, ignore_radius)
                if neg_mask.any():
                    target[(target == -1) & (neg_mask > 0)] = 0
                ignore_frac = float((target == -1).mean())
            else:
                ignore_frac = 0.0
            pos_frac = float(pos_mask.mean()) if pos_mask.size else 0.0
            neg_frac = float(neg_mask.mean()) if neg_mask.size else 0.0
            bg_frac = max(0.0, 1.0 - pos_frac - neg_frac - ignore_frac)
            pos_fracs.append(pos_frac)
            neg_fracs.append(neg_frac)
            bg_fracs.append(bg_frac)
            ignore_fracs.append(ignore_frac)
            included += 1

        if included == 0:
            logger.warning("%s summary: no images included after filtering.", name)
            stats = {
                "num_images": 0,
                "counts": counts,
                "pos_fracs": pos_fracs,
                "neg_fracs": neg_fracs,
                "bg_fracs": bg_fracs,
                "ignore_fracs": ignore_fracs,
            }
            self._stats_cache = stats
            return stats

        pos_arr = np.array(pos_fracs)
        neg_arr = np.array(neg_fracs)
        bg_arr = np.array(bg_fracs)
        ignore_arr = np.array(ignore_fracs)
        logger.info(
            "%s summary images=%d",
            name,
            included,
        )
        logger.info(
            "%s polygons pos_full_cross_section=%d partial_cross_section=%d neg_longitudinal=%d neg_oblique=%d",
            name,
            counts["pos_full_cross_section"],
            counts["partial_cross_section"],
            counts["neg_longitudinal"],
            counts["neg_oblique"],
        )
        logger.info(
            "%s pixel_fracs mean pos=%.4f neg_tubule=%.4f bg=%.4f ignore=%.4f",
            name,
            float(pos_arr.mean()),
            float(neg_arr.mean()),
            float(bg_arr.mean()),
            float(ignore_arr.mean()),
        )
        logger.info(
            "%s pixel_fracs median pos=%.4f neg_tubule=%.4f bg=%.4f ignore=%.4f",
            name,
            float(np.median(pos_arr)),
            float(np.median(neg_arr)),
            float(np.median(bg_arr)),
            float(np.median(ignore_arr)),
        )
        near_zero = (pos_arr < 1e-4).sum()
        if near_zero >= max(3, int(0.3 * included)):
            logger.warning(
                "%s summary: %d/%d samples have near-zero pos_frac (<1e-4).",
                name,
                int(near_zero),
                included,
            )
        stats = {
            "num_images": included,
            "counts": counts,
            "pos_fracs": pos_fracs,
            "neg_fracs": neg_fracs,
            "bg_fracs": bg_fracs,
            "ignore_fracs": ignore_fracs,
        }
        self._stats_cache = stats
        return stats

    @staticmethod
    def _normalize_orientations(values: Optional[List[Orientation | str]]) -> Optional[set[Orientation]]:
        if not values:
            return None
        normalized: set[Orientation] = set()
        for value in values:
            if isinstance(value, Orientation):
                normalized.add(value)
            else:
                try:
                    normalized.add(Orientation(str(value)))
                except Exception as exc:
                    allowed = ", ".join([o.value for o in Orientation])
                    raise ValueError(
                        f"Invalid allowed orientation '{value}'. Allowed values: {allowed}."
                    ) from exc
        return normalized
