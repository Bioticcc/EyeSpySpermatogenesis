"""Membrane-network tubule candidates (v0.5), not a validated PAS tubule detector.

Tubules in a section are packed edge to edge, so foreground thresholding selects
the whole section as one object (see docs/history/SEPARATION_FAILURE_ANALYSIS.md). This
method segments the *boundaries* instead: each tubule is wrapped in a
PAS-positive basement membrane, which appears as a thin magenta line. Regions
enclosed by that line network become candidates.

1. Coarse tissue mask (section silhouette only; never used to split tubules).
2. Chromatic contrast ``(R+B)/2 - G`` and a Sato ridge filter -> membrane map.
3. Seeds at points farthest from any membrane; watershed on the membrane map.
4. Merge neighbours whose shared border has weak membrane evidence, re-split
   oversized regions on a local membrane threshold, then merge regions entirely
   enclosed by one other region (lumens ringed off by stained sperm tails).
5. Clean-up (v0.3): absorb splinters and inner pieces (lumen cores, partial
   rings) into the neighbour that surrounds them; split regions holding two
   lumens separated by solid epithelium (adjacent tubules sharing a wall too
   faint to see at overview scale); drop thin strips along the capsule.
6. Trim (v0.4): cut unstained gaps outside each region's stained convex hull,
   then open the outline to remove thin spikes into interstitial space.
7. Faint walls (v0.4, needs native access): re-examine large or irregular
   regions at a finer scale, split along membranes as strong as the region's
   own outer wall, and fold back pieces that are small or barely touch the
   outer edge (lumens lined with PAS-positive material).
8. Re-join (v0.5): merge a piece wrapped by its neighbour, where an
   acrosome ring inside the epithelium was mistaken for a wall.
9. Per-candidate quality flags relative to the section's median candidate area.

``(R+B)/2 - G`` is a display-derived colour feature, not calibrated PAS stain
separation. All settings are in overview pixels and were chosen by inspecting
two sections at an 8x overview. The v0.3 clean-up thresholds were set against
506 reviewed v0.2 crops from three sections (outputs/tubule-review-2026-11-06-02);
the v0.4 trim and wall settings were set against the v0.3 review of sources
07-09 (outputs/tubule-review-2026-11-06-03); the v0.5 re-join step and the
0.9 wall ratio against the v0.4 review of sources 10-11 (-04). All remain
unvalidated on other sections.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import gaussian, sato, threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import (
    binary_opening,
    convex_hull_image,
    disk,
    remove_small_objects,
)
from skimage.segmentation import find_boundaries, relabel_sequential, watershed

METHOD = "membrane_ridge_watershed_v0.5"


@dataclass(frozen=True)
class MembraneConfig:
    """Overview-pixel settings; no physical-size assumptions."""

    seed_distance: int = 22
    seed_min_clearance: float = 6.0
    ridge_sigmas: tuple[float, ...] = (1.0, 1.5, 2.0)
    ridge_threshold: float | None = None
    merge_threshold: float = 0.5
    merge_min_border: int = 8
    tissue_sigma: float = 4.0
    tissue_opening: int = 8
    min_tissue_fraction: float = 0.05
    small_ratio: float = 0.25
    large_ratio: float = 2.5
    # v0.4 flag thresholds, set after outline trimming against the v0.3 review.
    min_solidity: float = 0.87
    min_axis_ratio: float = 0.4
    # v0.3 clean-up; ratios are relative to the section's median region area,
    # spans and gaps to the median region diameter.
    min_keep_ratio: float = 0.15
    absorb_ratio: float = 0.6
    absorb_share: float = 0.5
    max_span: float = 6.0
    lumen_split_ratio: float = 1.3
    lumen_min_ratio: float = 0.02
    lumen_gap: float = 0.35
    lumen_margin: int = 4
    split_min_piece: float = 0.5
    edge_fraction: float = 0.2
    # v0.4: outline trimming and the fine-scale faint-wall check.
    trim_ratio: float = 0.2
    fine_factor: int = 4
    fine_sigmas: tuple[float, ...] = (1.0, 1.5, 2.0, 3.0)
    wall_check_ratio: float = 1.3
    wall_check_solidity: float = 0.85
    wall_threshold: float = 0.4
    wall_ratio: float = 0.9
    min_outer_fraction: float = 0.3
    # v0.5: re-join a piece wrapped by its neighbour (inner zone vs. outer
    # crescent, split along the PAS-positive acrosome ring of some stages).
    rejoin_share: float = 0.4
    rejoin_solidity: float = 0.88
    rejoin_max_ratio: float = 2.2

    def __post_init__(self):
        if type(self.seed_distance) is not int or self.seed_distance < 1:
            raise ValueError("seed_distance must be a positive integer")
        if type(self.tissue_opening) is not int or self.tissue_opening < 0:
            raise ValueError("tissue_opening must be a nonnegative integer")
        if type(self.fine_factor) is not int or self.fine_factor < 1:
            raise ValueError("fine_factor must be a positive integer")
        if not self.fine_sigmas or any(
            not np.isfinite(s) or s <= 0 for s in self.fine_sigmas
        ):
            raise ValueError("fine_sigmas must be finite and positive")
        if type(self.lumen_margin) is not int or self.lumen_margin < 0:
            raise ValueError("lumen_margin must be a nonnegative integer")
        if type(self.merge_min_border) is not int or self.merge_min_border < 1:
            raise ValueError("merge_min_border must be a positive integer")
        if not self.ridge_sigmas or any(
            not np.isfinite(s) or s <= 0 for s in self.ridge_sigmas
        ):
            raise ValueError("ridge_sigmas must be finite and positive")
        for name in (
            "seed_min_clearance",
            "merge_threshold",
            "tissue_sigma",
            "small_ratio",
            "large_ratio",
            "min_keep_ratio",
            "absorb_ratio",
            "max_span",
            "lumen_split_ratio",
            "lumen_min_ratio",
            "lumen_gap",
            "split_min_piece",
            "trim_ratio",
            "wall_check_ratio",
            "wall_threshold",
            "wall_ratio",
            "rejoin_max_ratio",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        for name in (
            "min_tissue_fraction",
            "min_solidity",
            "min_axis_ratio",
            "absorb_share",
            "edge_fraction",
            "wall_check_solidity",
            "min_outer_fraction",
            "rejoin_share",
            "rejoin_solidity",
        ):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.ridge_threshold is not None and not np.isfinite(self.ridge_threshold):
            raise ValueError("ridge_threshold must be finite")


def tissue_mask(rgb: np.ndarray, config: MembraneConfig) -> np.ndarray:
    """Section silhouette. Hole filling is safe here: it is never used to split."""
    darkness = gaussian(1 - rgb.mean(axis=2), sigma=config.tissue_sigma)
    if float(np.ptp(darkness)) < 1e-7:
        return np.zeros(darkness.shape, dtype=bool)
    mask = darkness > threshold_otsu(darkness)
    if config.tissue_opening:
        mask = binary_opening(mask, disk(config.tissue_opening))
    mask = ndi.binary_fill_holes(mask)
    components, count = ndi.label(mask)
    if count == 0:
        return mask
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return sizes[components] >= config.min_tissue_fraction * sizes.max()


def membrane_ridges(rgb: np.ndarray, config: MembraneConfig) -> np.ndarray:
    """Thin bright lines in magenta-vs-green contrast."""
    chroma = (rgb[..., 0] + rgb[..., 2]) / 2 - rgb[..., 1]
    return sato(chroma, sigmas=config.ridge_sigmas, black_ridges=False)


def _border_pairs(labels: np.ndarray, values: np.ndarray | None = None):
    """Unique adjacent label pairs (a < b), with pixel counts and value sums."""
    firsts, seconds, sums = [], [], []
    for a, b, va, vb in (
        (labels[:, :-1], labels[:, 1:], *(_shift(values, 1))),
        (labels[:-1], labels[1:], *(_shift(values, 0))),
    ):
        different = a != b
        firsts.append(np.minimum(a, b)[different])
        seconds.append(np.maximum(a, b)[different])
        if values is not None:
            sums.append(((va + vb) / 2)[different])
    lo, hi = np.concatenate(firsts), np.concatenate(seconds)
    key = lo.astype(np.int64) * (int(labels.max()) + 1) + hi
    unique, inverse, counts = np.unique(key, return_inverse=True, return_counts=True)
    stride = int(labels.max()) + 1
    pairs = np.stack((unique // stride, unique % stride), axis=1)
    totals = (
        np.bincount(inverse, weights=np.concatenate(sums))
        if values is not None
        else None
    )
    return pairs, counts, totals


def _shift(values, axis):
    if values is None:
        return None, None
    if axis == 1:
        return values[:, :-1], values[:, 1:]
    return values[:-1], values[1:]


def _union(parent: np.ndarray, a: int, b: int):
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


def _resolve(parent: np.ndarray) -> np.ndarray:
    for i in range(len(parent)):
        root = i
        while parent[root] != root:
            root = parent[root]
        parent[i] = root
    return parent


def merge_weak_borders(labels, elevation, membrane, config: MembraneConfig):
    """Join neighbours whose shared border lacks membrane evidence."""
    if labels.max() < 2 or not membrane.any():
        return labels, 0
    scale = float(np.percentile(elevation[membrane], 50)) or 1.0
    normalised = np.clip(elevation / scale, 0, 1)
    pairs, counts, totals = _border_pairs(labels, normalised)
    parent = np.arange(int(labels.max()) + 1)
    merged = 0
    for (a, b), count, total in zip(pairs, counts, totals):
        if a == 0 or count < config.merge_min_border:
            continue
        if total / count < config.merge_threshold:
            _union(parent, int(a), int(b))
            merged += 1
    return _resolve(parent)[labels], merged


def merge_enclosed(labels: np.ndarray, max_passes: int = 3):
    """Merge each region whose only neighbour is one other (nonzero) region."""
    total = 0
    for _ in range(max_passes):
        pairs, _, _ = _border_pairs(labels)
        neighbours: dict[int, set[int]] = {}
        for a, b in pairs:
            neighbours.setdefault(int(a), set()).add(int(b))
            neighbours.setdefault(int(b), set()).add(int(a))
        # Regions on the image edge are not enclosed.
        edge = set(
            np.unique(
                np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
            ).tolist()
        )
        parent = np.arange(int(labels.max()) + 1)
        changed = 0
        for label, adjacent in neighbours.items():
            if label == 0 or label in edge or len(adjacent) != 1:
                continue
            (other,) = adjacent
            if other != 0:
                _union(parent, label, other)
                changed += 1
        if not changed:
            break
        labels = _resolve(parent)[labels]
        total += changed
    return labels, total


def _seed_and_flood(ridges, elevation, mask, threshold, config, max_candidates):
    """Seeds far from membrane pixels, flooded over the membrane map in ``mask``."""
    membrane = remove_small_objects((ridges > threshold) & mask, 20)
    clearance = gaussian(ndi.distance_transform_edt(mask & ~membrane), 3)
    peaks = peak_local_max(
        clearance,
        min_distance=config.seed_distance,
        threshold_abs=config.seed_min_clearance,
        labels=mask.astype(np.int32),
        exclude_border=False,
        num_peaks=max_candidates + 1,
    )
    if len(peaks) > max_candidates:
        raise ValueError("Too many seeds; review settings")
    markers = np.zeros(mask.shape, dtype=np.int32)
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    labels = watershed(elevation, markers, mask=mask).astype(np.int32)
    return labels, membrane, int(len(peaks))


def split_large_regions(labels, ridges, elevation, config, max_candidates):
    """Re-segment oversized regions with a membrane threshold local to each.

    Where staining is light, membranes fall below the section-wide threshold
    and one seed floods several tubules. Inside such a region the membranes are
    still its strongest ridges, so a local Otsu threshold can recover them.
    """
    regions = regionprops(labels)
    if len(regions) < 3:
        return labels, 0
    median = float(np.median([r.area for r in regions]))
    next_label = int(labels.max()) + 1
    resplit = 0
    for region in regions:
        if region.area <= config.large_ratio * median:
            continue
        sl = region.slice
        local = labels[sl] == region.label
        values = ridges[sl][local]
        if float(np.ptp(values)) < 1e-9:
            continue
        pieces, _, count = _seed_and_flood(
            ridges[sl],
            elevation[sl],
            local,
            float(threshold_otsu(values)),
            config,
            max_candidates,
        )
        if count < 2:
            continue
        view = labels[sl]
        view[local] = pieces[local] + next_label - 1
        next_label += count
        resplit += 1
    return labels, resplit


def _neighbour_borders(labels):
    """``{label: Counter(neighbour -> shared border pixels)}`` including 0."""
    pairs, counts, _ = _border_pairs(labels)
    borders: dict[int, dict[int, int]] = {}
    for (a, b), n in zip(pairs.tolist(), counts.tolist()):
        borders.setdefault(a, {})[b] = borders.setdefault(a, {}).get(b, 0) + n
        borders.setdefault(b, {})[a] = borders.setdefault(b, {}).get(a, 0) + n
    return borders


def absorb_fragments(labels, config: MembraneConfig, max_passes: int = 3):
    """Fold splinters and inner pieces into the region that surrounds them.

    A region is absorbed into its longest-bordering nonzero neighbour when it is
    tiny (``< min_keep_ratio`` x median), or when it is small (``< absorb_ratio``
    x median), that neighbour is larger, and it holds ``>= absorb_share`` of the
    region's border. Tiny regions with no nonzero neighbour are removed.
    """
    total = 0
    for _ in range(max_passes):
        areas = np.bincount(labels.ravel())
        present = np.flatnonzero(areas[1:]) + 1
        if len(present) < 3:
            break
        median = float(np.median(areas[present]))
        borders = _neighbour_borders(labels)
        parent = np.arange(len(areas))
        changed = 0
        for label in present.tolist():
            ratio = areas[label] / median
            if ratio >= config.absorb_ratio:
                continue
            around = borders.get(label, {})
            nonzero = {k: v for k, v in around.items() if k != 0}
            if not nonzero:
                if ratio < config.min_keep_ratio:
                    parent[label] = 0
                    changed += 1
                continue
            host = max(nonzero, key=nonzero.get)
            share = nonzero[host] / sum(around.values())
            if ratio < config.min_keep_ratio or (
                share >= config.absorb_share and areas[host] > areas[label]
            ):
                if parent[host] == label:  # avoid a two-way swap in one pass
                    continue
                parent[label] = host
                changed += 1
        if not changed:
            break
        for _ in range(len(parent)):  # resolve chains (a -> b -> c)
            nxt = parent[parent]
            if np.array_equal(nxt, parent):
                break
            parent = nxt
        labels = parent[labels]
        total += changed
    return labels, total


def _lumen_groups(lumens, keep, darkline, gap):
    """Group lumen pieces joined by a short dark run (one fragmented lumen)."""
    centres = np.array(ndi.center_of_mass(lumens > 0, lumens, keep))
    group = list(range(len(keep)))

    def find(i):
        while group[i] != i:
            i = group[i]
        return i

    for i in range(len(keep)):
        for j in range(i + 1, len(keep)):
            length = int(np.hypot(*(centres[i] - centres[j]))) + 1
            ys = np.linspace(centres[i][0], centres[j][0], length).round().astype(int)
            xs = np.linspace(centres[i][1], centres[j][1], length).round().astype(int)
            run = longest = 0
            for dark in darkline[ys, xs]:
                run = run + 1 if dark else 0
                longest = max(longest, run)
            if longest < gap:
                group[find(j)] = find(i)
    return [find(i) for i in range(len(keep))]


def split_two_lumen_regions(labels, bright, elevation, config: MembraneConfig):
    """Split large regions holding separate lumens divided by solid epithelium.

    Adjacent tubules often share one thin wall with no interstitial gap; at
    overview scale that wall can be too faint to stop the flood. Each tubule
    still has its own lumen, so two lumens separated by more than ``lumen_gap``
    median diameters of non-lumen tissue mark a merge. The region is re-flooded
    from the lumens; the split is kept only if every piece is tubule-sized.
    """
    regions = regionprops(labels)
    if len(regions) < 3:
        return labels, 0
    median = float(np.median([r.area for r in regions]))
    diameter = 2 * np.sqrt(median / np.pi)
    next_label = int(labels.max()) + 1
    splits = 0
    for region in regions:
        if region.area < config.lumen_split_ratio * median:
            continue
        sl = region.slice
        local = labels[sl] == region.label
        inner = ndi.binary_erosion(local, iterations=config.lumen_margin)
        lumens, count = ndi.label(bright[sl] & inner)
        if count < 2:
            continue
        sizes = np.bincount(lumens.ravel())
        keep = [
            i
            for i in range(1, count + 1)
            if sizes[i] >= config.lumen_min_ratio * median
        ]
        if len(keep) < 2:
            continue
        groups = _lumen_groups(lumens, keep, ~bright[sl], config.lumen_gap * diameter)
        ids = sorted(set(groups))
        if len(ids) < 2:
            continue
        markers = np.zeros(local.shape, dtype=np.int32)
        for lumen, group in zip(keep, groups):
            markers[lumens == lumen] = ids.index(group) + 1
        pieces = watershed(elevation[sl], markers, mask=local, compactness=0.01)
        piece_areas = np.bincount(pieces[local], minlength=len(ids) + 1)[1:]
        if piece_areas.min() < config.split_min_piece * median:
            continue
        view = labels[sl]
        view[local] = pieces[local] + next_label - 1
        next_label += len(ids)
        splits += 1
    return labels, splits


def reject_strips(labels, config: MembraneConfig):
    """Remove regions far longer than any tubule (capsule and section rim)."""
    regions = regionprops(labels)
    if len(regions) < 3:
        return labels, 0
    median = float(np.median([r.area for r in regions]))
    limit = config.max_span * 2 * np.sqrt(median / np.pi)
    rejected = 0
    for region in regions:
        y0, x0, y1, x1 = region.bbox
        if max(y1 - y0, x1 - x0) > limit:
            labels[labels == region.label] = 0
            rejected += 1
    return labels, rejected


def trim_outlines(labels, unstained, config: MembraneConfig):
    """Remove interstitial gaps and thin spikes from each region's outline.

    Unstained pixels connected to a region's edge are dropped when most of that
    patch lies outside the convex hull of the region's stained pixels, so a
    lumen opening onto the outline is kept. A morphological opening with a disk
    of ``trim_ratio`` x the median region radius then removes thin protrusions,
    and only the largest remaining piece is kept. Removed pixels become 0.
    """
    regions = regionprops(labels)
    if len(regions) < 3:
        return labels, 0
    median = float(np.median([r.area for r in regions]))
    radius = max(1, int(round(config.trim_ratio * np.sqrt(median / np.pi))))
    footprint = disk(radius)
    removed = 0
    for region in regions:
        y0, x0, y1, x1 = region.bbox
        pad = radius + 1
        y0, x0 = max(y0 - pad, 0), max(x0 - pad, 0)
        y1, x1 = min(y1 + pad, labels.shape[0]), min(x1 + pad, labels.shape[1])
        view = labels[y0:y1, x0:x1]
        mask = view == region.label
        keep = mask.copy()
        gaps, count = ndi.label(unstained[y0:y1, x0:x1] & mask)
        if count:
            rim = find_boundaries(mask, mode="inner")
            touching = np.unique(gaps[rim & (gaps > 0)])
            stained = mask & (gaps == 0)
            if len(touching) and stained.sum() > 10:
                hull = convex_hull_image(stained)
                outside = [t for t in touching if (~hull[gaps == t]).mean() > 0.5]
                keep &= ~np.isin(gaps, outside)
        keep = binary_opening(keep, footprint)
        pieces, count = ndi.label(keep)
        if count:
            sizes = np.bincount(pieces.ravel())
            sizes[0] = 0
            keep = ndi.binary_fill_holes(pieces == sizes.argmax()) & mask
        drop = mask & ~keep
        view[drop] = 0
        removed += int(drop.sum())
    return labels, removed


def _pair_counts(pieces, values=None):
    """``{(a, b): [count, value_sum]}`` for adjacent nonzero labels a < b."""
    out: dict[tuple[int, int], list[float]] = {}
    for a, b, va, vb in (
        (pieces[:, :-1], pieces[:, 1:], *_shift(values, 1)),
        (pieces[:-1], pieces[1:], *_shift(values, 0)),
    ):
        diff = (a != b) & (a > 0) & (b > 0)
        lo, hi = np.minimum(a, b)[diff], np.maximum(a, b)[diff]
        vals = np.maximum(va, vb)[diff] if values is not None else np.zeros(len(lo))
        for x, y, v in zip(lo.tolist(), hi.tolist(), vals.tolist()):
            entry = out.setdefault((x, y), [0, 0.0])
            entry[0] += 1
            entry[1] += v
    return out


def split_at_fine_scale(rgb, mask, median_area, config: MembraneConfig):
    """Split one region along membranes as strong as its own outer wall.

    ``rgb`` (float, 0-1) and ``mask`` are at ``fine_factor`` pixels per overview
    pixel. Returns a label array of pieces on that grid, or ``None``.
    """
    f = config.fine_factor
    chroma = (rgb[..., 0] + rgb[..., 2]) / 2 - rgb[..., 1]
    ridges = sato(chroma, sigmas=config.fine_sigmas, black_ridges=False)
    strongest = ndi.maximum_filter(ridges, size=2 * f + 1)
    edge = find_boundaries(mask, mode="inner")
    wall = float(np.median(strongest[edge])) if edge.any() else 0.0
    if wall <= 0:
        return None
    membrane = remove_small_objects(
        (ridges > config.wall_threshold * wall) & mask, 20 * f
    )
    clearance = gaussian(ndi.distance_transform_edt(mask & ~membrane), f)
    peaks = peak_local_max(
        clearance,
        min_distance=config.seed_distance * f,
        threshold_abs=config.seed_min_clearance * f,
        labels=mask.astype(np.int32),
        exclude_border=False,
    )
    if len(peaks) < 2:
        return None
    markers = np.zeros(mask.shape, dtype=np.int32)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    pieces = watershed(gaussian(ridges, 1), markers, mask=mask)
    parent = np.arange(len(peaks) + 1)
    for (a, b), (count, total) in _pair_counts(pieces, strongest).items():
        if total / count / wall < config.wall_ratio:
            _union(parent, a, b)
    pieces = _resolve(parent)[pieces]
    # Fold back pieces that are too small or barely touch the region's outer
    # edge (typically a lumen ringed by PAS-positive material).
    while True:
        ids = [int(i) for i in np.unique(pieces) if i]
        if len(ids) < 2:
            return None
        worst, score = None, 0.0
        for i in ids:
            piece = pieces == i
            rim = find_boundaries(piece, mode="inner")
            outer = (rim & edge).sum() / max(rim.sum(), 1)
            area = piece.sum() / f**2
            bad = max(
                config.split_min_piece * median_area / max(area, 1) - 1,
                (config.min_outer_fraction - outer) / config.min_outer_fraction,
            )
            if bad > score:
                worst, score = i, bad
        if worst is None:
            return pieces
        borders = {
            (b if a == worst else a): c
            for (a, b), (c, _) in _pair_counts(pieces).items()
            if worst in (a, b)
        }
        pieces[pieces == worst] = max(borders, key=borders.get) if borders else 0


def split_faint_walls(labels, fetch, config: MembraneConfig):
    """Re-check large or irregular regions at a finer scale (see module doc).

    ``fetch((x0, y0, x1, y1))`` returns RGB pixels for those overview bounds at
    ``fine_factor`` pixels per overview pixel, or ``None`` when unavailable.
    """
    regions = regionprops(labels)
    if len(regions) < 3:
        return labels, 0, 0
    median = float(np.median([r.area for r in regions]))
    f = config.fine_factor
    next_label = int(labels.max()) + 1
    checked = splits = 0
    for region in regions:
        if (
            region.area < config.wall_check_ratio * median
            and region.solidity >= config.wall_check_solidity
        ):
            continue
        y0, x0, y1, x1 = region.bbox
        pixels = fetch((x0, y0, x1, y1))
        if pixels is None:
            continue
        checked += 1
        rgb = pixels.astype(np.float32) / np.iinfo(pixels.dtype).max
        local = labels[y0:y1, x0:x1] == region.label
        fine_mask = np.kron(local, np.ones((f, f), dtype=bool))
        h, w = (
            min(rgb.shape[0], fine_mask.shape[0]),
            min(rgb.shape[1], fine_mask.shape[1]),
        )
        pieces = split_at_fine_scale(rgb[:h, :w], fine_mask[:h, :w], median, config)
        if pieces is None:
            continue
        coarse = np.zeros(local.shape, dtype=np.int32)
        sampled = pieces[f // 2 :: f, f // 2 :: f]
        coarse[: sampled.shape[0], : sampled.shape[1]] = sampled
        coarse[~local] = 0
        unassigned = local & (coarse == 0)
        if unassigned.any():
            nearest = ndi.distance_transform_edt(
                coarse == 0, return_distances=False, return_indices=True
            )
            coarse[unassigned] = coarse[tuple(i[unassigned] for i in nearest)]
        ids = [int(i) for i in np.unique(coarse[local]) if i]
        if len(ids) < 2:
            continue
        view = labels[y0:y1, x0:x1]
        for k, i in enumerate(ids):
            view[local & (coarse == i)] = next_label + k
        next_label += len(ids)
        splits += 1
    return labels, checked, splits


def rejoin_wrapped_pieces(labels, config: MembraneConfig, max_passes: int = 3):
    """Merge neighbour pairs that are one tubule cut into an inner and outer part.

    In some stages the acrosomes of developing spermatids form a PAS-positive
    ring partway through the epithelium, which the membrane map mistakes for a
    wall. The result is a piece mostly wrapped by its neighbour. A pair is
    merged when their shared border is ``>= rejoin_share`` of the smaller
    region's outline, their union is compact (solidity ``>= rejoin_solidity``)
    and no larger than ``rejoin_max_ratio`` x the median area. Two genuinely
    adjacent tubules share a much shorter border.
    """
    total = 0
    for _ in range(max_passes):
        regions = {r.label: r for r in regionprops(labels)}
        if len(regions) < 3:
            break
        median = float(np.median([r.area for r in regions.values()]))
        pairs, counts, _ = _border_pairs(labels)
        outline: dict[int, int] = {}
        for (a, b), n in zip(pairs.tolist(), counts.tolist()):
            outline[a] = outline.get(a, 0) + n
            outline[b] = outline.get(b, 0) + n
        candidates = []
        for (a, b), n in zip(pairs.tolist(), counts.tolist()):
            if a == 0:
                continue
            ra, rb = regions[a], regions[b]
            if ra.area + rb.area > config.rejoin_max_ratio * median:
                continue
            share = n / outline[a if ra.area <= rb.area else b]
            if share < config.rejoin_share:
                continue
            y0, x0 = min(ra.bbox[0], rb.bbox[0]), min(ra.bbox[1], rb.bbox[1])
            y1, x1 = max(ra.bbox[2], rb.bbox[2]), max(ra.bbox[3], rb.bbox[3])
            union = np.isin(labels[y0:y1, x0:x1], (a, b)).astype(np.uint8)
            if regionprops(union)[0].solidity >= config.rejoin_solidity:
                candidates.append((share, a, b))
        used: set[int] = set()
        changed = 0
        for _, a, b in sorted(candidates, reverse=True):
            if a in used or b in used:
                continue
            labels[labels == b] = a
            used |= {a, b}
            changed += 1
        if not changed:
            break
        total += changed
    return labels, total


def find_tubules(image, config: MembraneConfig, *, max_candidates=20_000, fetch=None):
    """Return ``(labels, candidates, diagnostics)`` like ``find_candidates``.

    ``fetch`` gives native-derived pixels for the faint-wall check (see
    ``split_faint_walls``); without it that step is skipped and recorded.
    """
    if image.dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("Membrane method requires uint8/uint16")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Membrane method requires RGB colour (PAS magenta contrast)")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    rgb = image.astype(np.float32) / np.iinfo(image.dtype).max
    tissue = tissue_mask(rgb, config)
    empty = np.zeros(image.shape[:2], dtype=np.int32)
    if not tissue.any():
        return empty, [], {"warnings": ["no_tissue_detected"]}
    ridges = membrane_ridges(rgb, config)
    ridge_threshold = (
        float(threshold_otsu(ridges[tissue]))
        if config.ridge_threshold is None
        else config.ridge_threshold
    )
    elevation = gaussian(ridges, 1)
    labels, membrane, seeds = _seed_and_flood(
        ridges, elevation, tissue, ridge_threshold, config, max_candidates
    )
    if not seeds:
        return empty, [], {"warnings": ["no_seeds"], "ridge_threshold": ridge_threshold}
    labels, weak_merges = merge_weak_borders(labels, elevation, membrane, config)
    labels, local_splits = split_large_regions(
        labels, ridges, elevation, config, max_candidates
    )
    labels, enclosed_merges = merge_enclosed(labels)
    labels, absorbed = absorb_fragments(labels, config)
    brightness = gaussian(rgb.mean(axis=2), 1)
    lumen = brightness > threshold_otsu(brightness[tissue])
    scale = float(np.percentile(elevation[membrane], 50)) if membrane.any() else 0.0
    labels, lumen_splits = split_two_lumen_regions(
        labels, lumen, elevation / (scale or 1.0), config
    )
    labels, strips = reject_strips(labels, config)
    labels, trimmed = trim_outlines(labels, lumen, config)
    wall_checked = wall_splits = 0
    if fetch is not None:
        labels, wall_checked, wall_splits = split_faint_walls(labels, fetch, config)
    labels, rejoined = rejoin_wrapped_pieces(labels, config)
    labels = relabel_sequential(labels)[0].astype(np.int32)
    candidates = describe_candidates(labels, tissue, config)
    diagnostics = {
        "warnings": [],
        "ridge_threshold": ridge_threshold,
        "tissue_fraction": float(tissue.mean()),
        "seeds": seeds,
        "weak_border_merges": weak_merges,
        "large_regions_resplit": local_splits,
        "enclosed_region_merges": enclosed_merges,
        "fragments_absorbed": absorbed,
        "two_lumen_splits": lumen_splits,
        "strips_rejected": strips,
        "outline_pixels_trimmed": trimmed,
        "faint_wall_check": "done" if fetch is not None else "skipped_no_fetch",
        "faint_wall_regions_checked": wall_checked,
        "faint_wall_splits": wall_splits,
        "wrapped_pieces_rejoined": rejoined,
        "median_candidate_area_px2": (
            float(np.median([c["overview_area_px2"] for c in candidates]))
            if candidates
            else None
        ),
    }
    return labels, candidates, diagnostics


def describe_candidates(labels, tissue, config: MembraneConfig):
    regions = regionprops(labels)
    if not regions:
        return []
    median = float(np.median([r.area for r in regions]))
    outside = ndi.binary_dilation(~tissue, iterations=2)
    height, width = labels.shape
    candidates = []
    for region in regions:
        y0, x0, y1, x1 = region.bbox
        flags = []
        ratio = region.area / median
        if ratio < config.small_ratio:
            flags.append("small_vs_section_median")
        if ratio > config.large_ratio:
            flags.append("large_vs_section_median")
        if region.solidity < config.min_solidity:
            flags.append("irregular_shape")
        major = region.axis_major_length
        axis_ratio = region.axis_minor_length / major if major else 1.0
        if axis_ratio < config.min_axis_ratio:
            flags.append("elongated")
        local = labels[y0:y1, x0:x1] == region.label
        rim = find_boundaries(np.pad(local, 1), mode="inner")[1:-1, 1:-1]
        edge_fraction = float((rim & outside[y0:y1, x0:x1]).sum() / max(rim.sum(), 1))
        if edge_fraction >= config.edge_fraction:
            flags.append("section_edge_contact")
        if x0 == 0 or y0 == 0 or x1 == width or y1 == height:
            flags.append("source_border_contact")
        candidates.append(
            {
                "label": int(region.label),
                "overview_bounds": [int(v) for v in (x0, y0, x1, y1)],
                "overview_area_px2": int(region.area),
                "overview_centroid_xy": [
                    float(region.centroid[1]),
                    float(region.centroid[0]),
                ],
                "quality": {
                    "area_vs_section_median": round(float(ratio), 3),
                    "solidity": round(float(region.solidity), 3),
                    "axis_ratio": round(float(axis_ratio), 3),
                    "section_edge_fraction": round(edge_fraction, 3),
                },
                "flags": flags,
                "crop_recommended": not flags,
                "review_status": "needs_review",
                "reasons": ["prototype_unvalidated", "native_boundary_not_refined"]
                + flags,
                "isolation_confidence": None,
                "stage": None,
            }
        )
    return candidates


def centered_square_bounds(candidate, geometry, scale: float):
    """Square source box centred on the candidate, clipped to the source extent.

    The square is centred on the candidate's centroid and just contains its
    whole overview bounding box; ``scale`` adds a margin (1.05 = 5%). Clipping
    at the image edge means edge crops may be non-square; recorded by the
    caller via the returned bounds.
    """
    x0, y0, x1, y1 = candidate["overview_bounds"]
    # Centroids are in pixel-index coordinates; bounds are pixel edges.
    cx, cy = (v + 0.5 for v in candidate["overview_centroid_xy"])
    half_x = max(cx - x0, x1 - cx) * geometry.scale_x
    half_y = max(cy - y0, y1 - cy) * geometry.scale_y
    side = 2 * max(half_x, half_y) * scale
    centre_x, centre_y = cx * geometry.scale_x, cy * geometry.scale_y
    left = int(np.floor(centre_x - side / 2))
    top = int(np.floor(centre_y - side / 2))
    right = int(np.ceil(centre_x + side / 2))
    bottom = int(np.ceil(centre_y + side / 2))
    return (
        max(left, 0),
        max(top, 0),
        min(right, geometry.source_width),
        min(bottom, geometry.source_height),
    )
