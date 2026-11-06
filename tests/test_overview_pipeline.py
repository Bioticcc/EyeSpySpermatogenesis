"""Synthetic mechanics, not evidence that PAS tubules are identified correctly."""

import json

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")
pytest.importorskip("skimage")

from eyespy_spermatogenesis.overview_pipeline import main  # noqa: E402
from eyespy_spermatogenesis.segmentation.overview import (  # noqa: E402
    CandidateConfig,
    find_candidates,
)


def synthetic_section():
    y, x = np.mgrid[:257, :385]
    image = np.full((257, 385, 3), 255, dtype=np.uint8)
    # Annuli, one crossing a hypothetical old grid seam, plus a true border cut.
    for cy, cx, radius in [(100, 190, 55), (175, 305, 38), (15, 0, 24)]:
        distance = (y - cy) ** 2 + (x - cx) ** 2
        wall = (distance <= radius**2) & (distance >= (radius - 10) ** 2)
        image[wall] = [110, 25, 100]
    return image


def test_end_to_end_native_crops_masks_and_provenance(tmp_path):
    source = tmp_path / "section.ome.tif"
    data = synthetic_section()
    tifffile.imwrite(
        source,
        data,
        photometric="rgb",
        tile=(32, 32),
        compression="deflate",
        metadata={"axes": "YXS"},
        iccprofile=b"test ICC",
    )
    before = source.read_bytes()
    output = tmp_path / "run"
    assert (
        main(
            [
                str(source),
                "--series",
                "0",
                "--output",
                str(output),
                "--detect",
                "--method",
                "darkness-v0.1",
                "--downsample",
                "4",
                "--threshold",
                "0.15",
                "--sigma",
                "0",
                "--closing-radius",
                "0",
                "--seed-distance",
                "8",
                "--margin",
                "8",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete_needs_review"
    assert manifest["physical_spacing_um"] is None
    assert manifest["native_detail_status"] == "resolution_unverified"
    assert len(manifest["source"]["sha256"]) == 64
    assert len(manifest["candidates"]) == 3
    assert any("source_border_contact" in c["reasons"] for c in manifest["candidates"])
    coverage = np.zeros(data.shape[:2], dtype=np.uint8)
    for candidate in manifest["candidates"]:
        assert candidate["review_status"] == "needs_review"
        assert candidate["stage"] is None and candidate["isolation_confidence"] is None
        assert candidate["export_status"] == "written"
        x0, y0, x1, y1 = candidate["source_bounds"]
        crop = tifffile.imread(output / candidate["crop"])
        mask = tifffile.imread(output / candidate["seed_mask"])
        np.testing.assert_array_equal(crop, data[y0:y1, x0:x1])
        assert mask.shape == crop.shape[:2] and set(np.unique(mask)) <= {0, 1}
        coverage[y0:y1, x0:x1] += mask
        with tifffile.TiffFile(output / candidate["crop"]) as tiff:
            assert tiff.pages[0].tags[34675].value == b"test ICC"
    assert coverage.max() == 1
    assert source.read_bytes() == before
    assert "overview.tif" in manifest["artifacts"]
    assert "review_overlay.png" in manifest["artifacts"]
    assert manifest["counts"]["detected_candidates"] == len(manifest["candidates"])
    # Existing runs and source images are never overwritten.
    saved_manifest = (output / "manifest.json").read_bytes()
    assert main([str(source), "--series", "0", "--output", str(output)]) == 1
    assert (output / "manifest.json").read_bytes() == saved_manifest


def test_overview_only_and_manual_roi_do_not_require_detection(tmp_path, monkeypatch):
    source = tmp_path / "gray.tif"
    data = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64)
    tifffile.imwrite(source, data, rowsperstrip=8)
    output = tmp_path / "io"

    def forbidden(*args, **kwargs):
        pytest.fail("Detection must be explicit")

    monkeypatch.setattr(
        "eyespy_spermatogenesis.segmentation.overview.find_candidates", forbidden
    )
    assert (
        main(
            [
                str(source),
                "--series",
                "0",
                "--output",
                str(output),
                "--roi",
                "5",
                "7",
                "20",
                "30",
            ]
        )
        == 0
    )
    np.testing.assert_array_equal(
        tifffile.imread(output / "manual_0001.tif"), data[7:30, 5:20]
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert not manifest["candidates"] and "segmentation" not in manifest


def test_uniform_and_small_candidates_are_explicit():
    config = CandidateConfig(sigma=0, closing_radius=0, min_area=10_000)
    labels, candidates, diagnostics = find_candidates(
        np.full((32, 32), 255, dtype=np.uint8), config
    )
    assert not candidates and not labels.any() and diagnostics["warnings"]
    image = np.full((64, 64), 255, dtype=np.uint8)
    image[20:30, 20:30] = 0
    _, candidates, _ = find_candidates(image, config)
    assert len(candidates) == 1 and "below_review_area" in candidates[0]["reasons"]


def test_touching_component_split_retains_hypothesis_flags():
    y, x = np.mgrid[:96, :128]
    foreground = ((y - 48) ** 2 + (x - 45) ** 2 < 28**2) | (
        (y - 48) ** 2 + (x - 87) ** 2 < 28**2
    )
    image = np.where(foreground, 0, 255).astype(np.uint8)
    _, candidates, _ = find_candidates(
        image, CandidateConfig(sigma=0, closing_radius=0)
    )
    assert len(candidates) == 2
    assert all("touching_split_hypothesis" in c["reasons"] for c in candidates)
    with pytest.raises(ValueError, match="Too many"):
        find_candidates(image, CandidateConfig(sigma=0), max_candidates=1)


def test_failed_run_keeps_failed_manifest(tmp_path):
    source = tmp_path / "bad.tif"
    source.write_bytes(b"not TIFF")
    output = tmp_path / "failed"
    assert main([str(source), "--series", "0", "--output", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and manifest["error"]


def test_pyramid_requires_alignment_confirmation(tmp_path):
    source = tmp_path / "input.tif"
    tifffile.imwrite(source, np.zeros((16, 16), dtype=np.uint8))
    output = tmp_path / "not_created"
    assert (
        main(
            [
                str(source),
                "--series",
                "0",
                "--output",
                str(output),
                "--overview-level",
                "1",
            ]
        )
        == 1
    )
    assert not output.exists()


def test_source_modification_invalidates_run(tmp_path, monkeypatch):
    source = tmp_path / "input.tif"
    tifffile.imwrite(source, np.zeros((16, 16), dtype=np.uint8))
    from eyespy_spermatogenesis.io.regions import TiffRegionReader

    original = TiffRegionReader.overview

    def modify(reader, *args, **kwargs):
        result = original(reader, *args, **kwargs)
        with source.open("ab") as handle:
            handle.write(b"changed")
        return result

    monkeypatch.setattr(TiffRegionReader, "overview", modify)
    output = tmp_path / "changed"
    assert main([str(source), "--series", "0", "--output", str(output)]) == 1
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "failed" and "Source changed" in manifest["error"]


def test_oversized_candidates_stay_in_manifest_without_partial_crops(tmp_path):
    source = tmp_path / "large.tif"
    image = np.full((512, 512), 255, dtype=np.uint8)
    image[30:480, 30:480] = 0
    tifffile.imwrite(source, image, tile=(32, 32))
    output = tmp_path / "limited"
    assert (
        main(
            [
                str(source),
                "--series",
                "0",
                "--output",
                str(output),
                "--detect",
                "--method",
                "darkness-v0.1",
                "--max-region-mib",
                "1",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert len(manifest["candidates"]) == 1
    candidate = manifest["candidates"][0]
    assert candidate["export_status"] == "deferred_resource_limit"
    assert "crop" not in candidate
    assert not list(output.glob("candidate_*.tif"))


def test_cli_stored_pyramid_with_noninteger_ratios(tmp_path):
    source = tmp_path / "pyramid.ome.tif"
    data = synthetic_section()
    from skimage.transform import resize_local_mean

    # Synthetic resampling covers the same pixel-edge field of view.
    coarse = np.rint(
        resize_local_mean(data, (86, 129, 3), grid_mode=True, preserve_range=True)
    ).astype(np.uint8)
    with tifffile.TiffWriter(source, ome=True) as writer:
        writer.write(
            data, subifds=1, photometric="rgb", tile=(32, 32), metadata={"axes": "YXS"}
        )
        writer.write(coarse, subfiletype=1, photometric="rgb", tile=(16, 16))
    output = tmp_path / "pyramid"
    assert (
        main(
            [
                str(source),
                "--series",
                "0",
                "--output",
                str(output),
                "--overview-level",
                "1",
                "--confirm-level-alignment",
                "--downsample",
                "1",
                "--detect",
                "--method",
                "darkness-v0.1",
                "--seed-distance",
                "8",
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["geometry"]["alignment"] == "operator_asserted_pyramid_alignment"
    assert manifest["geometry"]["overview_to_source"][0] == 385 / 129
    for candidate in manifest["candidates"]:
        if candidate["export_status"] == "written":
            x0, y0, x1, y1 = candidate["source_bounds"]
            np.testing.assert_array_equal(
                tifffile.imread(output / candidate["crop"]), data[y0:y1, x0:x1]
            )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--downsample", "0"],
        ["--threshold", "nan"],
        ["--sigma", "inf"],
        ["--series", "-1"],
        ["--margin", "-1"],
    ],
)
def test_invalid_numeric_options_create_no_results(tmp_path, arguments):
    output = tmp_path / "absent"
    with pytest.raises(SystemExit) as exc:
        main(["unused.tif", "--series", "0", "--output", str(output), *arguments])
    assert exc.value.code == 2
    assert not output.exists()


def test_review_only_records_all_candidates_without_native_crops(tmp_path):
    source = tmp_path / "section.tif"
    tifffile.imwrite(source, synthetic_section(), photometric="rgb", tile=(32, 32))
    output = tmp_path / "review"
    assert (
        main(
            [
                str(source),
                "--series",
                "0",
                "--downsample",
                "4",
                "--detect",
                "--method",
                "darkness-v0.1",
                "--skip-crops",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["counts"]["detected_candidates"] > 0
    assert all(
        c["export_status"] == "not_requested_review_only"
        for c in manifest["candidates"]
    )
    assert not list(output.glob("candidate_*.tif"))
    assert manifest["reader_statistics"]["native_crops"]["decoded_segments"] == 0
    from PIL import Image

    with Image.open(output / "review_overlay.png") as image:
        assert image.height == manifest["overview_shape"][0] + 125
