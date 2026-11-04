import numpy as np

from src.dataset.annotations import TubulePolygon
from src.dataset.masks import (
    build_ignore_target_mask,
    rasterize_instances,
    rasterize_polygon,
    rasterize_tubule_masks,
    resolve_pos_neg_conflicts,
    split_mask_instances,
)
from src.dataset.schema import Orientation


def test_rasterize_polygon_square() -> None:
    polygon = [[5, 5], [20, 5], [20, 20], [5, 20]]
    mask = rasterize_polygon(polygon, height=32, width=32)
    assert mask.shape == (32, 32)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0
    assert mask.sum() <= (16 * 16 + 16)


def test_rasterize_polygon_pos_frac_reasonable() -> None:
    polygon = [[10, 10], [30, 10], [30, 30], [10, 30]]
    mask = rasterize_polygon(polygon, height=64, width=64)
    pos_frac = float(mask.mean())
    assert mask.sum() < mask.size
    assert 0.01 <= pos_frac <= 0.8


def test_rasterize_instances_ids() -> None:
    poly1 = [[2, 2], [6, 2], [6, 6], [2, 6]]
    poly2 = [[10, 10], [14, 10], [14, 14], [10, 14]]
    inst = rasterize_instances([poly1, poly2], height=20, width=20, return_list=False)
    assert inst.max() >= 2
    assert (inst > 0).any()


def test_split_mask_instances_separates_bridge() -> None:
    mask = np.zeros((7, 9), dtype=bool)
    mask[1:4, 1:4] = True
    mask[1:4, 5:8] = True
    mask[2, 4] = True
    instance_map = split_mask_instances(mask, erosion_iters=1, min_area=1, seed_min_area=1)
    labels = np.unique(instance_map)
    labels = labels[labels > 0]
    assert len(labels) >= 2


def test_build_ignore_target_mask_dilates() -> None:
    pos_mask = np.zeros((7, 7), dtype=np.uint8)
    pos_mask[3, 3] = 1
    target = build_ignore_target_mask(pos_mask, ignore_radius_px=1)
    assert target[3, 3] == 1
    assert (target == -1).sum() > 0
    assert target[0, 0] == 0


def test_rasterize_tubule_masks_positive_only() -> None:
    tubules = [
        TubulePolygon(
            tubule_id="t1",
            polygon_xy=[[2, 2], [6, 2], [6, 6], [2, 6]],
            orientation=Orientation.CROSS_SECTION,
            partial=False,
        )
    ]
    pos_mask, neg_mask, counts = rasterize_tubule_masks(tubules, height=10, width=10)
    assert pos_mask.sum() > 0
    assert neg_mask.sum() == 0
    assert counts["pos_full_cross_section"] == 1


def test_rasterize_tubule_masks_partial_is_positive() -> None:
    tubules = [
        TubulePolygon(
            tubule_id="t1",
            polygon_xy=[[2, 2], [6, 2], [6, 6], [2, 6]],
            orientation=Orientation.CROSS_SECTION,
            partial=True,
        )
    ]
    pos_mask, neg_mask, counts = rasterize_tubule_masks(tubules, height=10, width=10)
    assert pos_mask.sum() > 0
    assert neg_mask.sum() == 0
    assert counts["partial_cross_section"] == 1


def test_rasterize_tubule_masks_oblique_is_negative() -> None:
    tubules = [
        TubulePolygon(
            tubule_id="t1",
            polygon_xy=[[2, 2], [6, 2], [6, 6], [2, 6]],
            orientation=Orientation.OBLIQUE,
        )
    ]
    pos_mask, neg_mask, counts = rasterize_tubule_masks(tubules, height=10, width=10)
    assert pos_mask.sum() == 0
    assert neg_mask.sum() > 0
    assert counts["neg_oblique"] == 1


def test_resolve_pos_neg_conflicts_pos_wins() -> None:
    pos = np.zeros((5, 5), dtype=np.uint8)
    neg = np.zeros((5, 5), dtype=np.uint8)
    pos[2, 2] = 1
    neg[2, 2] = 1
    neg_fixed, overlap = resolve_pos_neg_conflicts(pos, neg)
    assert overlap == 1
    assert neg_fixed[2, 2] == 0
