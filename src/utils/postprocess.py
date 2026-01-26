from __future__ import annotations

from collections import deque
from typing import List, Optional, Tuple

import numpy as np

try:
    from scipy import ndimage as _ndimage
except Exception:  # pragma: no cover - fallback if scipy isn't available
    _ndimage = None


def threshold_probs(probs: np.ndarray, threshold: float) -> np.ndarray:
    if probs is None:
        raise ValueError("probs array is required.")
    return (probs >= float(threshold))


def _label_components(mask: np.ndarray) -> Tuple[np.ndarray, int]:
    mask_bool = mask.astype(bool)
    if mask_bool.size == 0:
        return np.zeros_like(mask_bool, dtype=np.int32), 0
    if _ndimage is not None:
        structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        labels, num = _ndimage.label(mask_bool, structure=structure)
        return labels.astype(np.int32), int(num)
    height, width = mask_bool.shape
    labels = np.zeros((height, width), dtype=np.int32)
    label = 0
    for y in range(height):
        for x in range(width):
            if not mask_bool[y, x] or labels[y, x] != 0:
                continue
            label += 1
            queue = deque([(y, x)])
            labels[y, x] = label
            while queue:
                cy, cx = queue.popleft()
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and mask_bool[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = label
                        queue.append((ny, nx))
    return labels, label


def _components_from_labels(
    labels: np.ndarray,
    probs: Optional[np.ndarray] = None,
    include_mask: bool = False,
) -> List[dict]:
    components: List[dict] = []
    if labels.size == 0:
        return components
    label_ids = np.unique(labels)
    label_ids = label_ids[label_ids > 0]
    for label_id in label_ids:
        coords = np.argwhere(labels == label_id)
        if coords.size == 0:
            continue
        ys = coords[:, 0]
        xs = coords[:, 1]
        area = int(coords.shape[0])
        x1 = int(xs.min())
        x2 = int(xs.max())
        y1 = int(ys.min())
        y2 = int(ys.max())
        centroid = (float(xs.mean()), float(ys.mean()))
        comp = {
            "label": int(label_id),
            "area": area,
            "bbox": (x1, y1, x2, y2),
            "centroid": centroid,
        }
        if probs is not None:
            comp["mean_prob"] = float(np.mean(probs[labels == label_id]))
        if include_mask:
            comp["mask"] = labels == label_id
        components.append(comp)
    return components


def extract_components(
    mask: np.ndarray,
    probs: Optional[np.ndarray] = None,
    include_mask: bool = False,
) -> List[dict]:
    mask_bool = mask.astype(bool)
    labels, _ = _label_components(mask_bool)
    return _components_from_labels(labels, probs=probs, include_mask=include_mask)


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if min_area_px <= 0 or mask_bool.size == 0 or not mask_bool.any():
        return mask_bool
    labels, num = _label_components(mask_bool)
    if num == 0:
        return np.zeros_like(mask_bool, dtype=bool)
    areas = np.bincount(labels.ravel())
    keep = areas >= int(min_area_px)
    keep[0] = False
    return keep[labels]


def fill_small_holes(mask: np.ndarray, max_hole_area_px: int) -> np.ndarray:
    mask_bool = mask.astype(bool)
    if max_hole_area_px <= 0 or mask_bool.size == 0 or not mask_bool.any():
        return mask_bool
    background = ~mask_bool
    if not background.any():
        return mask_bool
    labels, num = _label_components(background)
    if num == 0:
        return mask_bool
    border_labels = np.unique(
        np.concatenate([labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]])
    )
    border_set = set(border_labels.tolist())
    areas = np.bincount(labels.ravel())
    fill_labels = [
        label_id
        for label_id in range(1, len(areas))
        if label_id not in border_set and areas[label_id] <= int(max_hole_area_px)
    ]
    if not fill_labels:
        return mask_bool
    filled = mask_bool.copy()
    filled[np.isin(labels, fill_labels)] = True
    return filled


def keep_largest_components(
    mask: np.ndarray,
    max_components: int,
    probs: Optional[np.ndarray] = None,
    include_mask: bool = False,
) -> Tuple[np.ndarray, List[dict]]:
    mask_bool = mask.astype(bool)
    labels, num = _label_components(mask_bool)
    if num == 0:
        return np.zeros_like(mask_bool, dtype=bool), []
    components = _components_from_labels(labels, probs=probs, include_mask=include_mask)
    if max_components is None or max_components <= 0 or len(components) <= max_components:
        return labels > 0, components
    components_sorted = sorted(components, key=lambda c: c["area"], reverse=True)
    kept = components_sorted[: int(max_components)]
    keep_labels = {c["label"] for c in kept}
    kept_mask = np.isin(labels, list(keep_labels))
    if include_mask:
        for comp in kept:
            comp["mask"] = labels == comp["label"]
    return kept_mask, kept


def postprocess_mask(
    mask: np.ndarray,
    min_area_px: int,
    max_components: int,
    fill_holes: bool = True,
    max_hole_area_px: int = 0,
    probs: Optional[np.ndarray] = None,
    include_mask: bool = False,
) -> Tuple[np.ndarray, List[dict]]:
    processed = mask.astype(bool)
    if min_area_px > 0:
        processed = remove_small_components(processed, min_area_px=min_area_px)
    if fill_holes and max_hole_area_px > 0:
        processed = fill_small_holes(processed, max_hole_area_px=max_hole_area_px)
    processed, components = keep_largest_components(
        processed,
        max_components=max_components,
        probs=probs,
        include_mask=include_mask,
    )
    return processed, components
