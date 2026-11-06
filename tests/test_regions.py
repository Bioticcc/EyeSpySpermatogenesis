"""Reader contracts tested against full decodes of SMALL synthetic images only."""

import pytest

np = pytest.importorskip("numpy")
tifffile = pytest.importorskip("tifffile")

from eyespy_spermatogenesis.io.regions import OverviewGeometry, TiffRegionReader  # noqa: E402


@pytest.mark.parametrize("rgb", [False, True])
@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("storage", ["tile", "strip"])
def test_region_and_streamed_overview_match_reference(tmp_path, rgb, dtype, storage):
    shape = (39, 53, 3) if rgb else (39, 53)
    data = np.arange(np.prod(shape)).reshape(shape).astype(dtype)
    path = tmp_path / "input.ome.tif"
    options = {"tile": (16, 16)} if storage == "tile" else {"rowsperstrip": 7}
    tifffile.imwrite(
        path,
        data,
        photometric="rgb" if rgb else "minisblack",
        metadata={"axes": "YXS" if rgb else "YX"},
        compression="deflate",
        **options,
    )
    before = path.read_bytes()
    with TiffRegionReader(path, series=0) as reader:
        # Region crosses storage seams and must retain dtype and channel order.
        region = reader.read_region((13, 5, 49, 38))
        np.testing.assert_array_equal(region, data[5:38, 13:49])
        assert region.dtype == data.dtype
        for factor in (1, 3, 8, 64):
            previous = reader.decoded_segments
            overview = reader.overview(factor)
            expected = np.empty_like(overview)
            for y in range(overview.shape[0]):
                for x in range(overview.shape[1]):
                    patch = data[
                        y * factor : (y + 1) * factor, x * factor : (x + 1) * factor
                    ]
                    expected[y, x] = np.rint(patch.mean(axis=(0, 1))).astype(dtype)
            np.testing.assert_array_equal(overview, expected)
            assert reader.decoded_segments - previous == len(reader.page.dataoffsets)
    assert path.read_bytes() == before


def test_bounded_access_never_calls_full_array_decode(tmp_path, monkeypatch):
    path = tmp_path / "tiles.tif"
    data = np.arange(128 * 160, dtype=np.uint16).reshape(128, 160)
    tifffile.imwrite(path, data, tile=(16, 16))

    def forbid(*args, **kwargs):
        pytest.fail("Full-page/series decoding is forbidden in this adapter")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbid)
    monkeypatch.setattr(tifffile.TiffPageSeries, "asarray", forbid)
    monkeypatch.setattr(tifffile.TiffFile, "asarray", forbid)
    monkeypatch.setattr(tifffile, "imread", forbid)
    with TiffRegionReader(path, series=0) as reader:
        result = reader.read_region((17, 18, 23, 25))
        np.testing.assert_array_equal(result, data[18:25, 17:23])
        assert reader.decoded_segments == 1
        reader.overview(8)


@pytest.mark.parametrize(
    "kind", ["orientation", "stack", "rgba", "float", "planar", "white"]
)
def test_unsupported_layouts_fail_closed(tmp_path, kind):
    path = tmp_path / "unsupported.tif"
    data, options = np.zeros((32, 32), dtype=np.uint8), {}
    if kind == "orientation":
        options["extratags"] = [(274, "H", 1, 6, False)]
    elif kind == "stack":
        data = np.zeros((2, 32, 32), dtype=np.uint8)
        options = {"metadata": {"axes": "ZYX"}, "photometric": "minisblack"}
    elif kind == "rgba":
        data = np.zeros((32, 32, 4), dtype=np.uint8)
        options["photometric"] = "rgb"
    elif kind == "float":
        data = data.astype(np.float32)
    elif kind == "planar":
        data = np.zeros((3, 32, 32), dtype=np.uint8)
        options = {"photometric": "rgb", "planarconfig": "separate"}
    elif kind == "white":
        options["photometric"] = "miniswhite"
    tifffile.imwrite(path, data, **options)
    with pytest.raises(ValueError):
        TiffRegionReader(path, series=0)


def test_limits_and_invalid_bounds_fail_before_decoding(tmp_path):
    path = tmp_path / "input.tif"
    tifffile.imwrite(path, np.zeros((64, 64), dtype=np.uint8), rowsperstrip=64)
    with pytest.raises(ValueError, match="max_segment_bytes"):
        TiffRegionReader(path, series=0, max_segment_bytes=128)
    with TiffRegionReader(path, series=0, max_region_bytes=128) as reader:
        with pytest.raises(ValueError, match="max_region_bytes"):
            reader.read_region((0, 0, 64, 64))
        with pytest.raises(ValueError, match="max_pixels"):
            reader.overview(1, max_pixels=32)
        for bounds in [(-1, 0, 1, 2), (0, 0, 65, 64), (1, 1, 1, 2), (0.0, 0, 2, 2)]:
            with pytest.raises(ValueError):
                reader.read_region(bounds)
        assert reader.decoded_segments == 0


def test_pyramid_selection_and_noninteger_transform(tmp_path):
    path = tmp_path / "pyramid.ome.tif"
    with tifffile.TiffWriter(path, ome=True) as writer:
        writer.write(
            np.zeros((43, 65, 3), dtype=np.uint8),
            subifds=1,
            tile=(16, 16),
            photometric="rgb",
            metadata={"axes": "YXS"},
        )
        writer.write(
            np.full((21, 32, 3), 42, dtype=np.uint8),
            subfiletype=1,
            tile=(16, 16),
            photometric="rgb",
        )
    with TiffRegionReader(path, series=0, level=1) as reader:
        assert reader.width == 32 and reader.height == 21
        assert np.all(reader.overview(2) == 42)
    geometry = OverviewGeometry(65, 43, 32, 21, 2)
    assert geometry.source_bounds((1, 2, 3, 4)) == (4, 8, 13, 17)
    assert geometry.source_bounds((15, 10, 16, 11), 3) == (57, 37, 65, 43)


def test_projection_of_partial_edge_bins_is_clipped_and_aligned():
    geometry = OverviewGeometry(19, 13, 19, 13, 8)
    labels = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    projected = geometry.project_labels(labels, (0, 0, 19, 13))
    assert projected.shape == (13, 19)
    assert np.all(projected[:8, :8] == 1)
    assert np.all(projected[8:, 16:] == 6)
    assert geometry.source_bounds((2, 1, 3, 2), margin=5) == (11, 3, 19, 13)


def test_bigtiff_big_endian_and_explicit_series_selection(tmp_path):
    path = tmp_path / "multiple.tif"
    data = np.arange(35 * 47, dtype=np.uint16).reshape(35, 47)
    with tifffile.TiffWriter(path, bigtiff=True, byteorder=">") as writer:
        writer.write(np.zeros((16, 16), dtype=np.uint8), photometric="minisblack")
        writer.write(data, photometric="minisblack", tile=(16, 16))
    with TiffRegionReader(path, series=1) as reader:
        np.testing.assert_array_equal(
            reader.read_region((7, 3, 43, 32)), data[3:32, 7:43]
        )
        assert reader.overview(3).dtype == np.dtype("uint16")
    with pytest.raises(ValueError, match="existing series"):
        TiffRegionReader(path, series=2)


def test_multifile_ome_is_refused(tmp_path):
    path = tmp_path / "single.ome.tif"
    tifffile.imwrite(path, np.zeros((16, 16), dtype=np.uint8), metadata={"axes": "YX"})
    with tifffile.TiffFile(path) as tiff:
        xml = tiff.ome_metadata
    # Add an extra companion declaration; it must not be silently accepted as
    # the complete source merely because the first plane lives in this file.
    xml = xml.replace(
        "</Pixels>",
        '<TiffData><UUID FileName="elsewhere.ome.tif">'
        "urn:uuid:00000000-0000-0000-0000-000000000001</UUID>"
        "</TiffData></Pixels>",
    )
    tifffile.tiffcomment(path, xml)
    with pytest.raises(ValueError, match="Multifile"):
        TiffRegionReader(path, series=0)


def test_jpeg_ycbcr_regions_are_decoded_rgb(tmp_path):
    pytest.importorskip("imagecodecs")
    path = tmp_path / "jpeg.tif"
    y, x = np.mgrid[:47, :61]
    data = np.stack((x * 4, y * 5, (x + y) * 2), axis=-1).astype(np.uint8)
    tifffile.imwrite(path, data, photometric="rgb", compression="jpeg", tile=(16, 16))
    # JPEG is already lossy: the reference is decoded RGB, not pre-encoding data.
    with tifffile.TiffFile(path) as tiff:
        expected = tiff.pages[0].asarray()
        assert tiff.pages[0].photometric == 6
    with TiffRegionReader(path, series=0) as reader:
        actual = reader.read_region((13, 9, 57, 44))
        np.testing.assert_array_equal(actual, expected[9:44, 13:57])
        assert reader.pixel_history == [
            "JPEG YCbCr decoded to RGB by tifffile/imagecodecs"
        ]


def test_renamed_ome_self_reference_resolves_by_uuid(tmp_path):
    path = tmp_path / "renamed.ome.tif"
    tifffile.imwrite(
        path, np.full((16, 16), 42, dtype=np.uint8), metadata={"axes": "YX"}
    )
    with tifffile.TiffFile(path) as tiff:
        xml = tiff.ome_metadata
    from xml.etree import ElementTree

    root = ElementTree.fromstring(xml)
    source_uuid = root.get("UUID")
    assert source_uuid
    xml = xml.replace(
        "</Pixels>",
        '<TiffData><UUID FileName="old-name.ome.tif">'
        f"{source_uuid}</UUID></TiffData></Pixels>",
    )
    tifffile.tiffcomment(path, xml)
    with TiffRegionReader(path, series=0) as reader:
        assert np.all(reader.read_region((0, 0, 16, 16)) == 42)
        assert reader.ome_file_references[0]["resolved_as"] == "current_file_uuid"
