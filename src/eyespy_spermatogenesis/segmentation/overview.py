"""Untuned overview candidate generator, not a validated PAS tubule detector."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import binary_closing, disk
from skimage.segmentation import watershed


@dataclass(frozen=True)
class CandidateConfig:
    """Demonstration settings in overview pixels; no physical-size assumptions."""

    threshold: float | None = None
    sigma: float = 1.0
    closing_radius: int = 1
    seed_distance: int = 12
    min_area: int = 100

    def __post_init__(self):
        if self.threshold is not None and not 0 <= self.threshold <= 1:
            raise ValueError("Darkness threshold must be between 0 and 1")
        if not np.isfinite(self.sigma) or self.sigma < 0:
            raise ValueError("Smoothing sigma must be finite and nonnegative")
        for value, minimum in (
            (self.closing_radius, 0),
            (self.seed_distance, 1),
            (self.min_area, 1),
        ):
            if type(value) is not int or value < minimum:
                raise ValueError("Invalid morphology size; use integer overview pixels")


def find_candidates(image, config: CandidateConfig, *, max_candidates=20_000):
    """Dark foreground → closing/fill → distance seeds → watershed.

    All candidates, including small and border-touching ones, are retained for
    review. White lumens are filled as an experimental heuristic. Interstitial
    tissue can also be detected; these IDs do not establish biological identity.
    """
    if image.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("Candidate generator requires uint8/uint16")
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError("Candidate generator requires gray or RGB")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    gray = image.astype(np.float32) / np.iinfo(image.dtype).max
    if gray.ndim == 3:
        gray = gray.mean(axis=2)
    darkness = gaussian(1 - gray, sigma=config.sigma, preserve_range=True)
    if float(np.ptp(darkness)) < 1e-7:
        return (
            np.zeros(gray.shape, dtype=np.int32),
            [],
            {
                "threshold": config.threshold,
                "warnings": ["uniform_overview_no_candidates"],
            },
        )
    threshold = (
        float(threshold_otsu(darkness))
        if config.threshold is None
        else config.threshold
    )
    foreground = darkness > threshold
    if config.closing_radius:
        foreground = binary_closing(foreground, footprint=disk(config.closing_radius))
    foreground = ndi.binary_fill_holes(foreground)
    # Padding makes the distance to the true image border explicit.
    distance = ndi.distance_transform_edt(np.pad(foreground, 1))[1:-1, 1:-1]
    components, count = ndi.label(foreground)
    if count > max_candidates:
        raise ValueError("Too many foreground components; review settings")
    peaks = peak_local_max(
        distance,
        min_distance=config.seed_distance,
        labels=components,
        exclude_border=False,
        num_peaks=max_candidates + 1,
    )
    if len(peaks) > max_candidates:
        raise ValueError("Too many watershed seeds; review settings")
    markers = np.zeros(gray.shape, dtype=np.int32)
    for index, (y, x) in enumerate(peaks, 1):
        markers[y, x] = index
    # Never drop a foreground component solely because it received no peak.
    seeded = set(components[markers > 0].tolist())
    for component, slices in enumerate(ndi.find_objects(components), 1):
        if component in seeded:
            continue
        local_distance = np.where(components[slices] == component, distance[slices], -1)
        y, x = np.unravel_index(np.argmax(local_distance), local_distance.shape)
        markers[y + slices[0].start, x + slices[1].start] = len(peaks) + 1
        peaks = np.vstack((peaks, [y + slices[0].start, x + slices[1].start]))
    if len(peaks) > max_candidates:
        raise ValueError("Too many candidates; review settings")
    labels = watershed(-distance, markers, mask=foreground, watershed_line=True)
    labels = labels.astype(np.int32)
    # Several markers within one component are a separation hypothesis.
    seed_components = components[markers > 0]
    seed_counts = np.bincount(seed_components, minlength=count + 1)
    label_components = np.zeros(len(peaks) + 1, dtype=np.int32)
    label_components[markers[markers > 0]] = seed_components
    candidates = []
    for region in regionprops(labels):
        y0, x0, y1, x1 = region.bbox
        reasons = ["prototype_unvalidated", "native_boundary_not_refined"]
        if region.area < config.min_area:
            reasons.append("below_review_area")
        if x0 == 0 or y0 == 0 or x1 == image.shape[1] or y1 == image.shape[0]:
            reasons.append("source_border_contact")
        component = label_components[region.label]
        if seed_counts[component] > 1:
            reasons.append("touching_split_hypothesis")
        candidates.append(
            {
                "label": int(region.label),
                "overview_bounds": [int(v) for v in (x0, y0, x1, y1)],
                "overview_area_px2": int(region.area),
                "overview_centroid_xy": [
                    float(region.centroid[1]),
                    float(region.centroid[0]),
                ],
                "review_status": "needs_review",
                "reasons": reasons,
                "isolation_confidence": None,
                "stage": None,
            }
        )
    return labels, candidates, {"threshold": threshold, "warnings": []}
