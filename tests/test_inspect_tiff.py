"""Exercise metadata-only inspection and CLI integrity on synthetic TIFFs."""

import hashlib
import json

import pytest

np = pytest.importorskip("numpy", reason="Install .[io] for TIFF inspection tests")
tifffile = pytest.importorskip(
    "tifffile", reason="Install .[io] for TIFF inspection tests"
)

from eyespy_spermatogenesis.io.inspect_tiff import (  # noqa: E402
    _ome_metadata,
    inspect_tiff,
    main,
)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_plain_tiff_preserves_native_metadata_without_claiming_calibration(
    tmp_path, dtype
):
    source = tmp_path / "mouse_40x.TiF"
    tifffile.imwrite(source, np.zeros((23, 35), dtype=dtype), photometric="minisblack")
    result = inspect_tiff(source)
    level = result["series"][0]["levels"][0]
    assert level["axes"] == "YX"
    assert level["shape"] == [23, 35]
    assert level["dtype"] == dtype
    assert level["estimated_array_bytes"] == 23 * 35 * np.dtype(dtype).itemsize
    assert level["representative_page"]["tiled"] is False
    assert result["source"]["sha256"] is None
    assert result["native_detail_status"] == "resolution_unverified"
    assert result["calibration_status"] == "pending_verification"
    assert result["ome"]["images"] == []
    json.dumps(result, allow_nan=False)


def test_pyramidal_ome_and_thumbnail_are_reported_without_decoding(
    tmp_path, monkeypatch
):
    source = tmp_path / "section.ome.tiff"
    rgb = np.zeros((64, 80, 3), dtype="uint8")
    with tifffile.TiffWriter(source, bigtiff=True, ome=True) as writer:
        writer.write(
            rgb,
            subifds=1,
            tile=(16, 16),
            photometric="rgb",
            metadata={
                "axes": "YXS",
                "PhysicalSizeX": 0.25,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": 0.4,
                "PhysicalSizeYUnit": "µm",
            },
        )
        writer.write(rgb[::2, ::2], subfiletype=1, tile=(16, 16), photometric="rgb")
        writer.write(np.zeros((16, 16), dtype="uint8"), metadata={"Name": "thumbnail"})

    def forbid_pixels(*args, **kwargs):
        raise AssertionError("Inspection attempted image-array decoding")

    for cls in (tifffile.TiffFile, tifffile.TiffPage, tifffile.TiffPageSeries):
        monkeypatch.setattr(cls, "asarray", forbid_pixels)
    result = inspect_tiff(source)
    assert result["is_bigtiff"] is True
    assert result["is_ome"] is True
    assert result["pixel_data_decoded"] is False
    assert len(result["series"]) == 2
    levels = result["series"][0]["levels"]
    assert [level["shape"] for level in levels] == [[64, 80, 3], [32, 40, 3]]
    assert levels[1]["dimension_ratios_to_level0"] == {"X": 2.0, "Y": 2.0}
    page = levels[0]["representative_page"]
    assert page["photometric"] == "RGB"
    assert page["samples_per_pixel"] == 3
    assert page["tile_width"] == page["tile_height"] == 16
    pixels = result["ome"]["images"][0]["pixels"]
    assert pixels["PhysicalSizeX"] == "0.25"
    assert pixels["PhysicalSizeY"] == "0.4"
    assert pixels["PhysicalSizeXUnit"] == "µm"
    assert result["calibration_status"] == "pending_verification"
    json.dumps(result, allow_nan=False)


def test_checksums_and_icc_are_recorded_without_modifying_source(tmp_path):
    source = tmp_path / "profile.tif"
    profile = b"synthetic profile bytes for header inspection only"
    tifffile.imwrite(
        source,
        np.zeros((16, 16, 3), dtype="uint8"),
        photometric="rgb",
        iccprofile=profile,
    )
    original = source.read_bytes()
    result = inspect_tiff(source, checksum=True)
    assert result["source"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert result["series"][0]["levels"][0]["representative_page"]["icc_profile"] == {
        "bytes": len(profile),
        "sha256": hashlib.sha256(profile).hexdigest(),
    }
    assert source.read_bytes() == original


def test_multichannel_axes_and_orientation_are_not_silently_normalized(tmp_path):
    source = tmp_path / "channels.ome.tif"
    tifffile.imwrite(
        source,
        np.zeros((2, 16, 32), dtype="uint16"),
        photometric="minisblack",
        metadata={"axes": "CYX"},
        extratags=[(274, "H", 1, 6, False)],
    )
    result = inspect_tiff(source)
    level = result["series"][0]["levels"][0]
    assert level["axes"] == "CYX"
    assert level["shape"] == [2, 16, 32]
    assert level["representative_page"]["samples_per_pixel"] == 1
    assert level["representative_page"]["orientation"] in (6, "RIGHTTOP")
    assert len(result["ome"]["images"][0]["channels"]) == 2


def test_ome_objective_and_companion_references_remain_reported_not_inferred():
    xml = """<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">
      <Instrument ID="Instrument:0"><Objective ID="Objective:0"
        NominalMagnification="40"/></Instrument>
      <Image ID="Image:0"><ObjectiveSettings ID="Objective:0"/>
        <Pixels PhysicalSizeX="0.3"><Channel ID="Channel:0:0"/>
          <TiffData><UUID FileName="companion.ome.tif"/></TiffData>
        </Pixels></Image></OME>"""
    result = _ome_metadata(xml)
    assert result["objectives"][0]["NominalMagnification"] == "40"
    image = result["images"][0]
    assert image["objective_settings"] == {"ID": "Objective:0"}
    assert image["companion_files"] == ["companion.ome.tif"]
    assert image["pixels"] == {"PhysicalSizeX": "0.3"}
    assert "PhysicalSizeXUnit" not in image["pixels"]


@pytest.mark.parametrize(
    "xml", [None, "<OME>", '<!DOCTYPE OME [<!ENTITY x "y">]><OME/>']
)
def test_missing_or_invalid_ome_is_explicit(xml):
    result = _ome_metadata(xml)
    assert result["warnings"]
    assert not result["images"]


def test_cli_deduplicates_sources_and_lists_non_tiffs(tmp_path):
    source = tmp_path / "sample.TIFF"
    tifffile.imwrite(source, np.zeros((16, 16), dtype="uint8"))
    (tmp_path / "notes.txt").write_text("not an image")
    nested = tmp_path / "nested"
    nested.mkdir()
    tifffile.imwrite(nested / "other.tif", np.zeros((16, 16), dtype="uint8"))
    output = tmp_path / "reports" / "inspection.json"
    assert main([str(tmp_path), str(source), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert len(report["sources"]) == 1
    assert report["skipped_non_tiff_files"] == [str(tmp_path / "notes.txt")]
    assert report["schema_version"] == "0.1"
    recursive_output = tmp_path / "reports" / "recursive.json"
    assert main([str(tmp_path), "--recursive", "--output", str(recursive_output)]) == 0
    assert len(json.loads(recursive_output.read_text())["sources"]) == 2


def test_cli_reports_corrupt_file_and_continues(tmp_path):
    source = tmp_path / "good.tif"
    tifffile.imwrite(source, np.zeros((16, 16), dtype="uint8"))
    (tmp_path / "bad.tif").write_bytes(b"not a TIFF")
    output = tmp_path / "inspection.json"
    assert main([str(tmp_path), "--output", str(output)]) == 1
    report = json.loads(output.read_text())
    assert len(report["sources"]) == 1
    assert report["errors"][0]["path"].endswith("bad.tif")


def test_cli_never_overwrites_existing_output_or_source(tmp_path, capsys):
    source = tmp_path / "source.tif"
    tifffile.imwrite(source, np.zeros((16, 16), dtype="uint8"))
    original = source.read_bytes()
    assert main([str(source), "--output", str(source)]) == 1
    assert source.read_bytes() == original
    assert "Cannot write" in capsys.readouterr().err
    output = tmp_path / "existing.json"
    output.write_text("owner report")
    assert main([str(source), "--output", str(output)]) == 1
    assert output.read_text() == "owner report"


@pytest.mark.parametrize("kind", ["missing", "empty", "unsupported"])
def test_cli_rejects_unusable_input(tmp_path, kind):
    path = tmp_path / kind
    if kind == "empty":
        path.mkdir()
    elif kind == "unsupported":
        path = path.with_suffix(".jpg")
        path.write_bytes(b"not TIFF")
    with pytest.raises(SystemExit) as exc:
        main([str(path)])
    assert exc.value.code == 2


def test_cli_reports_directory_access_failure_without_traceback(tmp_path, monkeypatch):
    def deny_access(path):
        raise PermissionError("Synthetic directory access denial")

    monkeypatch.setattr(type(tmp_path), "iterdir", deny_access)
    with pytest.raises(SystemExit) as exc:
        main([str(tmp_path)])
    assert exc.value.code == 2
