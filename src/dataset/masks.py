from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw

from src.dataset.annotations import TubulePolygon
from src.dataset.schema import Orientation


def _sanitize_polygon(polygon_xy: Iterable[Iterable[float]], height: int, width: int) -> List[List[float]]:
    points: List[List[float]] = []
    for pair in polygon_xy:
        try:
            x, y = float(pair[0]), float(pair[1])
        except Exception:
            continue
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        x = min(max(x, 0.0), max(0.0, width - 1.0))
        y = min(max(y, 0.0), max(0.0, height - 1.0))
        points.append([x, y])
    if len(points) < 3:
        return []
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    if (max(xs) - min(xs)) < 1.0 or (max(ys) - min(ys)) < 1.0:
        return []
    return points


def rasterize_polygon(polygon_xy: Iterable[Iterable[float]], height: int, width: int) -> np.ndarray:
    if height <= 0 or width <= 0:
        raise ValueError("Invalid mask size.")
    points = _sanitize_polygon(polygon_xy, height, width)
    if not points:
        return np.zeros((height, width), dtype=np.uint8)
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon([tuple(p) for p in points], outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)


def rasterize_instances(
    polygons: Iterable[Iterable[Iterable[float]]],
    height: int,
    width: int,
    return_list: bool = False,
) -> np.ndarray | List[np.ndarray]:
    masks: List[np.ndarray] = []
    for polygon in polygons:
        mask = rasterize_polygon(polygon, height, width)
        if mask.any():
            masks.append(mask)
    if return_list:
        return masks
    instance = np.zeros((height, width), dtype=np.uint16)
    for idx, mask in enumerate(masks, start=1):
        instance[mask > 0] = idx
    return instance


def split_tubule_polygons(
    tubules: Sequence[TubulePolygon],
    allowed_orientations: Optional[set[Orientation]] = None,
) -> Tuple[List[List[List[float]]], List[List[List[float]]], Dict[str, int]]:
    counts = {
        "pos_full_cross_section": 0,
        "partial_cross_section": 0,
        "neg_longitudinal": 0,
        "neg_oblique": 0,
    }
    pos_polygons: List[List[List[float]]] = []
    neg_polygons: List[List[List[float]]] = []
    for tubule in tubules:
        if allowed_orientations is not None and tubule.orientation not in allowed_orientations:
            continue
        if tubule.orientation == Orientation.CROSS_SECTION:
            if tubule.partial:
                counts["partial_cross_section"] += 1
            else:
                counts["pos_full_cross_section"] += 1
            pos_polygons.append(tubule.polygon_xy)
        elif tubule.orientation == Orientation.LONGITUDINAL:
            counts["neg_longitudinal"] += 1
            neg_polygons.append(tubule.polygon_xy)
        elif tubule.orientation == Orientation.OBLIQUE:
            counts["neg_oblique"] += 1
            neg_polygons.append(tubule.polygon_xy)
    return pos_polygons, neg_polygons, counts


def rasterize_tubule_masks(
    tubules: Sequence[TubulePolygon],
    height: int,
    width: int,
    allowed_orientations: Optional[set[Orientation]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    pos_polygons, neg_polygons, counts = split_tubule_polygons(tubules, allowed_orientations=allowed_orientations)
    pos_instance = rasterize_instances(pos_polygons, height, width, return_list=False)
    neg_instance = rasterize_instances(neg_polygons, height, width, return_list=False)
    pos_mask = (pos_instance > 0).astype(np.uint8)
    neg_mask = (neg_instance > 0).astype(np.uint8)
    return pos_mask, neg_mask, counts


def resolve_pos_neg_conflicts(pos_mask: np.ndarray, neg_mask: np.ndarray) -> Tuple[np.ndarray, int]:
    pos = pos_mask.astype(bool)
    neg = neg_mask.astype(bool)
    overlap = pos & neg
    if not overlap.any():
        return neg_mask, 0
    neg_fixed = neg_mask.copy()
    neg_fixed[overlap] = 0
    return neg_fixed, int(overlap.sum())


def _binary_erosion(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return mask.copy()
    eroded = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(eroded, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[1:-1, 1:-1],
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
            padded[:-2, :-2],
            padded[:-2, 2:],
            padded[2:, :-2],
            padded[2:, 2:],
        ]
        eroded = np.logical_and.reduce(neighbors)
        if not eroded.any():
            break
    return eroded


def _binary_dilation(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    if iterations <= 0:
        return mask.copy()
    dilated = mask.astype(bool)
    for _ in range(iterations):
        padded = np.pad(dilated, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[1:-1, 1:-1],
            padded[:-2, 1:-1],
            padded[2:, 1:-1],
            padded[1:-1, :-2],
            padded[1:-1, 2:],
            padded[:-2, :-2],
            padded[:-2, 2:],
            padded[2:, :-2],
            padded[2:, 2:],
        ]
        dilated = np.logical_or.reduce(neighbors)
    return dilated


def dilate_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    return _binary_dilation(mask.astype(bool), iterations=max(int(radius), 0))


def build_ignore_target_mask(pos_mask: np.ndarray, ignore_radius_px: int) -> np.ndarray:
    pos = pos_mask.astype(bool)
    target = np.zeros_like(pos, dtype=np.int8)
    if not pos.any():
        return target
    if ignore_radius_px <= 0:
        target[pos] = 1
        return target
    ignore = _binary_dilation(pos, iterations=ignore_radius_px)
    target[pos] = 1
    target[ignore & ~pos] = -1
    return target


def _label_connected_components(mask: np.ndarray, min_area: int = 0) -> np.ndarray:
    height, width = mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    label = 0
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or labels[y, x] != 0:
                continue
            coords = []
            queue = deque([(y, x)])
            labels[y, x] = -1
            while queue:
                cy, cx = queue.popleft()
                coords.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = -1
                        queue.append((ny, nx))
            if len(coords) < max(1, min_area):
                for cy, cx in coords:
                    labels[cy, cx] = 0
                continue
            label += 1
            for cy, cx in coords:
                labels[cy, cx] = label
    return labels


def split_mask_instances(
    mask: np.ndarray,
    erosion_iters: int = 2,
    min_area: int = 20,
    seed_min_area: int = 5,
) -> np.ndarray:
    if mask.size == 0:
        return np.zeros_like(mask, dtype=np.int32)
    mask_bool = mask.astype(bool)
    if not mask_bool.any():
        return np.zeros_like(mask_bool, dtype=np.int32)
    seed_mask = _binary_erosion(mask_bool, iterations=erosion_iters)
    seed_labels = _label_connected_components(seed_mask, min_area=seed_min_area)
    if seed_labels.max() == 0:
        return _label_connected_components(mask_bool, min_area=min_area)

    height, width = mask_bool.shape
    instance_map = np.zeros((height, width), dtype=np.int32)
    queue = deque()
    seed_points = np.argwhere(seed_labels > 0)
    for y, x in seed_points:
        label = seed_labels[y, x]
        instance_map[y, x] = label
        queue.append((y, x, label))

    while queue:
        cy, cx, label = queue.popleft()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < height and 0 <= nx < width and mask_bool[ny, nx] and instance_map[ny, nx] == 0:
                instance_map[ny, nx] = label
                queue.append((ny, nx, label))

    unlabeled = mask_bool & (instance_map == 0)
    if unlabeled.any():
        extra = _label_connected_components(unlabeled, min_area=min_area)
        if extra.max() > 0:
            instance_map[extra > 0] = extra[extra > 0] + instance_map.max()

    if min_area > 1:
        areas = np.bincount(instance_map.ravel())
        for label, area in enumerate(areas):
            if label == 0:
                continue
            if area < min_area:
                instance_map[instance_map == label] = 0

    labels = np.unique(instance_map)
    labels = labels[labels > 0]
    relabeled = np.zeros_like(instance_map, dtype=np.int32)
    for new_label, old_label in enumerate(labels, start=1):
        relabeled[instance_map == old_label] = new_label
    return relabeled


def boxes_from_instance_map(
    instance_map: np.ndarray,
    probs: Optional[np.ndarray] = None,
    min_area: int = 20,
) -> List[dict]:
    boxes: List[dict] = []
    labels = np.unique(instance_map)
    labels = labels[labels > 0]
    for label in labels:
        coords = np.argwhere(instance_map == label)
        if coords.size == 0:
            continue
        area = coords.shape[0]
        if area < min_area:
            continue
        ys = coords[:, 0]
        xs = coords[:, 1]
        min_x = int(xs.min())
        max_x = int(xs.max())
        min_y = int(ys.min())
        max_y = int(ys.max())
        score = 1.0
        if probs is not None:
            score = float(np.mean(probs[instance_map == label]))
        boxes.append(
            {
                "bbox_xywh": [float(min_x), float(min_y), float(max_x - min_x + 1), float(max_y - min_y + 1)],
                "score": float(score),
                "label": "tubule",
            }
        )
    return boxes
