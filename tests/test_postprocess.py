import numpy as np

from src.utils.postprocess import fill_small_holes, keep_largest_components, remove_small_components


def test_remove_small_components_removes_specks() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[1:5, 1:5] = True
    mask[8, 8] = True
    mask[0, 9] = True
    mask[9, 0] = True

    cleaned = remove_small_components(mask, min_area_px=3)

    assert cleaned.sum() == 16
    assert not cleaned[8, 8]
    assert not cleaned[0, 9]
    assert not cleaned[9, 0]


def test_fill_small_holes_fills_hole() -> None:
    mask = np.ones((8, 8), dtype=bool)
    mask[3:5, 3:5] = False

    filled = fill_small_holes(mask, max_hole_area_px=5)

    assert filled.all()


def test_keep_largest_components_limits_count() -> None:
    mask = np.zeros((20, 20), dtype=bool)
    starts = [(1, 1), (1, 6), (1, 11), (6, 1), (6, 6), (6, 11)]
    for y, x in starts:
        mask[y : y + 2, x : x + 2] = True

    kept_mask, components = keep_largest_components(mask, max_components=4)

    assert len(components) == 4
    assert int(kept_mask.sum()) == 16
