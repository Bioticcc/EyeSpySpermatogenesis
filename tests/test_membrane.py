"""Synthetic mechanics for the v0.5 membrane method.

The synthetic section mimics the property that broke v0.1 (tubules packed edge
to edge, white lumens, a surrounding capsule). Passing these tests is not
evidence of performance on real PAS sections.
"""

import json

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("skimage")

from eyespy_spermatogenesis.io.regions import OverviewGeometry  # noqa: E402
from eyespy_spermatogenesis.overview_pipeline import main  # noqa: E402
from eyespy_spermatogenesis.segmentation import review_sheet  # noqa: E402
from eyespy_spermatogenesis.segmentation.membrane import (  # noqa: E402
    MembraneConfig,
    absorb_fragments,
    centered_square_bounds,
    find_tubules,
    merge_enclosed,
    reject_strips,
    rejoin_wrapped_pieces,
    split_at_fine_scale,
    split_faint_walls,
    split_large_regions,
    split_two_lumen_regions,
    trim_outlines,
)

RADIUS = 40


def packed_section(seed=0):
    """Hexagonally packed tubules: epithelium, magenta membrane, white lumen."""
    rng = np.random.default_rng(seed)
    height, width = 420, 460
    y, x = np.mgrid[:height, :width]
    image = np.empty((height, width, 3), dtype=np.float32)
    image[:] = [208, 208, 216]  # slide background
    centres = []
    for row in range(4):
        for col in range(4):
            cx = 95 + col * 2 * RADIUS + (RADIUS if row % 2 else 0)
            cy = 90 + row * int(RADIUS * 1.75)
            centres.append((cy, cx))
    section = np.zeros((height, width), dtype=bool)
    for cy, cx in centres:
        section |= (y - cy) ** 2 + (x - cx) ** 2 <= (RADIUS + 4) ** 2
    image[section] = [240, 234, 242]  # interstitial gaps
    for cy, cx in centres:
        d = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        image[d <= RADIUS] = [175, 140, 200]  # epithelium
        image[(d > RADIUS - 2.5) & (d <= RADIUS)] = [200, 85, 200]  # membrane
        image[d <= RADIUS * 0.3] = [236, 228, 242]  # lumen
    image += rng.normal(0, 6, image.shape)
    return np.clip(image, 0, 255).astype(np.uint8), centres


def test_packed_tubules_get_one_centred_candidate_each():
    image, centres = packed_section()
    labels, candidates, diagnostics = find_tubules(image, MembraneConfig())
    assert not diagnostics["warnings"]
    hit = [int(labels[cy, cx]) for cy, cx in centres]
    assert 0 not in hit
    assert len(set(hit)) == len(centres), "tubules merged or lumens split off"
    by_label = {c["label"]: c for c in candidates}
    for (cy, cx), label in zip(centres, hit):
        x, y = by_label[label]["overview_centroid_xy"]
        assert abs(x - cx) < 8 and abs(y - cy) < 8
    for c in candidates:
        assert c["stage"] is None and c["isolation_confidence"] is None
        assert c["crop_recommended"] == (not c["flags"])
        assert "prototype_unvalidated" in c["reasons"]


def test_membrane_method_rejects_gray_and_handles_blank():
    with pytest.raises(ValueError, match="RGB"):
        find_tubules(np.zeros((32, 32), dtype=np.uint8), MembraneConfig())
    labels, candidates, diagnostics = find_tubules(
        np.full((64, 64, 3), 200, dtype=np.uint8), MembraneConfig()
    )
    assert not candidates and not labels.any() and diagnostics["warnings"]


def test_enclosed_regions_merge_but_edge_regions_do_not():
    labels = np.zeros((40, 60), dtype=np.int32)
    labels[5:35, 5:35] = 1
    labels[15:25, 15:25] = 2  # enclosed by 1 only -> merged
    labels[0:10, 40:60] = 3  # touches the image edge -> kept
    merged, count = merge_enclosed(labels)
    assert count == 1
    assert set(np.unique(merged)) == {0, 1, 3}


def test_oversized_region_is_resplit_on_local_membrane():
    labels = np.zeros((120, 260), dtype=np.int32)
    for i, x0 in enumerate((0, 30, 60, 90), 1):
        labels[0:25, x0 : x0 + 25] = i  # four typical-size regions
    labels[30:110, 20:240] = 5  # far larger than the median
    ridges = np.zeros(labels.shape, dtype=np.float32)
    ridges[30:110, 128:131] = 0.2  # faint membrane only inside region 5
    resplit, count = split_large_regions(
        labels.copy(), ridges, ridges, MembraneConfig(), 1000
    )
    assert count == 1
    assert resplit[70, 60] != resplit[70, 200]
    assert resplit[70, 60] not in (1, 2, 3, 4) and resplit[70, 200] not in (1, 2, 3, 4)
    np.testing.assert_array_equal(resplit[0:25, 0:115], labels[0:25, 0:115])


def test_inner_pieces_and_splinters_are_absorbed():
    labels = np.zeros((60, 240), dtype=np.int32)
    for i in range(4):
        labels[:, i * 60 : (i + 1) * 60] = i + 1
    labels[10:50, 25:60] = 5  # inner piece: 3 sides on region 1, 1 on region 2
    labels[0:2, 100:102] = 6  # splinter inside region 2
    absorbed, count = absorb_fragments(labels.copy(), MembraneConfig())
    assert count == 2
    assert absorbed[30, 40] == 1 and absorbed[0, 100] == 2
    assert set(np.unique(absorbed)) == {1, 2, 3, 4}


def test_two_lumen_region_splits_but_fragmented_lumen_does_not():
    labels = np.zeros((100, 400), dtype=np.int32)
    for i in range(3):
        labels[:, i * 60 : (i + 1) * 60] = i + 1
    labels[:, 180:] = 4  # several times the median area
    y, x = np.mgrid[:100, :400]
    elevation = np.zeros(labels.shape, dtype=np.float32)

    def lumens(*centres, radius):
        return np.any(
            [(y - cy) ** 2 + (x - cx) ** 2 <= radius**2 for cy, cx in centres], axis=0
        )

    two = lumens((50, 235), (50, 345), radius=12)
    split, count = split_two_lumen_regions(
        labels.copy(), two, elevation, MembraneConfig()
    )
    assert count == 1
    assert split[50, 235] != split[50, 345]
    assert split[50, 235] > 4 and split[50, 345] > 4
    np.testing.assert_array_equal(split[:, :180], labels[:, :180])
    one = lumens((50, 282), (50, 302), radius=8)  # 4 px apart: one lumen
    kept, count = split_two_lumen_regions(
        labels.copy(), one, elevation, MembraneConfig()
    )
    assert count == 0
    np.testing.assert_array_equal(kept, labels)


def test_capsule_strips_are_rejected():
    labels = np.zeros((40, 400), dtype=np.int32)
    for i in range(4):
        labels[0:30, i * 30 : (i + 1) * 30] = i + 1
    labels[35:38, 0:300] = 5
    kept, count = reject_strips(labels.copy(), MembraneConfig())
    assert count == 1 and not (kept == 5).any()
    np.testing.assert_array_equal(kept[:30], labels[:30])


def test_trim_removes_spikes_and_outside_gaps_but_keeps_lumens():
    y, x = np.mgrid[:200, :400]
    labels = np.zeros((200, 400), dtype=np.int32)
    for i, cx in enumerate((50, 150, 250, 350), 1):
        labels[(y - 100) ** 2 + (x - cx) ** 2 <= 40**2] = i
    labels[98:102, 50:100] = 1  # thin spike leaving tubule 1
    gap = (y - 100) ** 2 + (x - 102) ** 2 <= 12**2
    labels[gap & (labels == 0)] = 2  # unstained gap patch stuck to tubule 2
    unstained = gap | ((y - 100) ** 2 + (x - 250) ** 2 <= 10**2)  # + lumen of 3
    trimmed, removed = trim_outlines(labels.copy(), unstained, MembraneConfig())
    assert removed > 0
    assert trimmed[100, 95] == 0, "spike kept"
    assert trimmed[100, 100] == 0, "gap kept"
    assert trimmed[100, 250] == 3, "lumen removed"
    assert trimmed[100, 50] == 1 and trimmed[100, 150] == 2


def two_tubules(wall=True, lumen_ring=False):
    """Fine-scale RGB: two touching discs, optional shared magenta wall."""
    y, x = np.mgrid[:200, :360]
    rgb = np.empty((200, 360, 3), dtype=np.float32)
    rgb[:] = [0.7, 0.58, 0.78]  # epithelium
    left = (y - 100) ** 2 + (x - 95) ** 2 <= 90**2
    right = (y - 100) ** 2 + (x - 265) ** 2 <= 90**2
    mask = left | right
    from scipy import ndimage as ndi

    outer = mask & ~ndi.binary_erosion(mask, iterations=3)
    rgb[outer] = [0.8, 0.3, 0.8]
    if wall:  # two basement membranes back to back stain more intensely
        rgb[np.abs(x - 180) <= 1] = [0.85, 0.2, 0.85]
    if lumen_ring:
        d = np.sqrt((y - 100) ** 2 + (x - 95) ** 2)
        rgb[np.abs(d - 30) <= 1.5] = [0.8, 0.3, 0.8]
    rgb[~mask] = [0.94, 0.92, 0.95]
    return rgb, mask


def test_fine_scale_split_needs_a_real_wall():
    config = MembraneConfig()
    median = 2000  # overview px^2; each disc is ~1590 overview px^2 (0.8x)
    rgb, mask = two_tubules(wall=True)
    pieces = split_at_fine_scale(rgb, mask, median, config)
    assert pieces is not None and pieces[100, 60] != pieces[100, 300]
    rgb, mask = two_tubules(wall=False, lumen_ring=True)
    assert split_at_fine_scale(rgb, mask, median, config) is None


def test_split_faint_walls_uses_fetched_pixels():
    config = MembraneConfig()
    rgb, mask = two_tubules(wall=True)
    f = config.fine_factor
    labels = np.zeros((50, 200), dtype=np.int32)
    labels[:, 90:] = 0
    coarse = mask[f // 2 :: f, f // 2 :: f]
    labels[: coarse.shape[0], : coarse.shape[1]][coarse] = 1  # merged pair
    for i, x0 in enumerate((100, 130, 160), 2):
        labels[5:40, x0 : x0 + 28] = i  # three typical regions
    fine = (rgb * 255).astype(np.uint8)
    calls = []

    def fetch(bounds):
        calls.append(bounds)
        x0, y0, x1, y1 = bounds
        return fine[y0 * f : y1 * f, x0 * f : x1 * f]

    split, checked, splits = split_faint_walls(labels.copy(), fetch, config)
    assert calls and checked >= 1 and splits == 1
    assert split[25, 20] != split[25, 65]
    assert split[25, 20] not in (0, 1, 2, 3, 4)


def test_wrapped_inner_piece_rejoins_but_neighbours_do_not():
    y, x = np.mgrid[:120, :480]
    labels = np.zeros((120, 480), dtype=np.int32)
    for i, cx in enumerate((60, 180, 300, 420), 1):
        labels[(y - 60) ** 2 + (x - cx) ** 2 <= 50**2] = i
    # Tubule 1 cut into an inner zone and an outer crescent (acrosome ring).
    labels[((y - 60) ** 2 + (x - 60) ** 2 <= 30**2)] = 5
    joined, count = rejoin_wrapped_pieces(labels.copy(), MembraneConfig())
    assert count == 1
    assert joined[60, 60] == joined[60, 25] and joined[60, 60] != 0
    for cx in (180, 300, 420):  # touching neighbours stay separate
        assert joined[60, cx] != joined[60, 60]
    assert len({joined[60, cx] for cx in (180, 300, 420)}) == 3


def test_centered_square_bounds_contain_candidate_and_clip():
    geometry = OverviewGeometry(800, 600, 800, 600, 8)
    centre = {
        "overview_bounds": [40, 30, 60, 50],
        "overview_centroid_xy": [49.5, 39.5],
    }
    assert centered_square_bounds(centre, geometry, 1.0) == (320, 240, 480, 400)
    # Off-centre centroid: the square still contains the whole bounding box.
    corner = {"overview_bounds": [0, 0, 20, 20], "overview_centroid_xy": [2, 2]}
    x0, y0, x1, y1 = centered_square_bounds(corner, geometry, 1.5)
    assert (x0, y0) == (0, 0) and x1 == y1 == 230


@pytest.mark.parametrize(
    "kwargs",
    [{"seed_distance": 0}, {"merge_threshold": float("nan")}, {"min_solidity": 2}],
)
def test_invalid_membrane_config(kwargs):
    with pytest.raises(ValueError):
        MembraneConfig(**kwargs)


def test_cli_thumbnails_and_review_sheet(tmp_path):
    image, centres = packed_section()
    source = tmp_path / "section.tif"
    tifffile.imwrite(source, image, photometric="rgb", tile=(64, 64))
    parent = tmp_path / "batch"
    output = parent / "source_01"
    parent.mkdir()
    args = [str(source), "--series", "0", "--downsample", "1", "--detect"]
    assert (
        main([*args, "--skip-crops", "--thumbnail-size", "96", "--output", str(output)])
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["segmentation"]["method"] == "membrane_ridge_watershed_v0.5"
    assert manifest["segmentation"]["diagnostics"]["faint_wall_check"] == "done"
    assert manifest["counts"]["detected_candidates"] >= len(centres)
    assert not list(output.glob("candidate_*.tif"))
    for c in manifest["candidates"]:
        assert c["export_status"] == "thumbnail_only"
        assert (output / c["thumbnail"]).exists()
        assert c["thumbnail"] in manifest["artifacts"]
        assert c["thumbnail_outline"] in manifest["artifacts"]
    from PIL import Image

    first = manifest["candidates"][0]
    with Image.open(output / first["thumbnail"]) as thumb:
        rgb = np.asarray(thumb.convert("RGB")).astype(int)
    with Image.open(output / first["thumbnail_outline"]) as overlay:
        assert overlay.mode == "RGBA" and overlay.size == thumb.size
        alpha = np.asarray(overlay)[..., 3]
    assert (alpha == 255).sum() > 50, "candidate outline not drawn"
    yellow = (rgb[..., 0] > 200) & (rgb[..., 1] > 160) & (rgb[..., 2] < 100)
    assert yellow.sum() < 10, "outline must stay off the photo itself"
    assert review_sheet.main([str(parent)]) == 0
    page = (parent / "review.html").read_text()
    assert "source_01/thumb_" in page and "section.tif" in page
    assert "source_01/outline_" in page
    rows = (parent / "review_template.csv").read_text().strip().splitlines()
    assert len(rows) == 1 + len(manifest["candidates"])
    assert review_sheet.main([str(parent)]) == 1  # never overwrites

    # Full crops are centred squares of exact source pixels.
    full = tmp_path / "full"
    assert main([*args, "--output", str(full)]) == 0
    manifest = json.loads((full / "manifest.json").read_text())
    for c in manifest["candidates"]:
        x0, y0, x1, y1 = c["source_bounds"]
        np.testing.assert_array_equal(
            tifffile.imread(full / c["crop"]), image[y0:y1, x0:x1]
        )
