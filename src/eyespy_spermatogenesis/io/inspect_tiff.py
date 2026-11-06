"""Report TIFF headers and OME metadata without decoding image arrays."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

TIFF_SUFFIXES = {".tif", ".tiff", ".btf", ".tf2", ".tf8"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ome_metadata(xml: str | None) -> dict[str, Any]:
    """Keep OME values as reported, without assigning images to TIFF series."""
    result: dict[str, Any] = {"images": [], "objectives": [], "warnings": []}
    if not xml:
        result["warnings"].append(
            "OME-XML is absent; physical calibration is unresolved."
        )
        return result
    if "<!DOCTYPE" in xml.upper() or "<!ENTITY" in xml.upper():
        result["warnings"].append(
            "OME-XML with DTD/entity declarations was not parsed."
        )
        return result
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as exc:
        result["warnings"].append(f"OME-XML could not be parsed: {exc}")
        return result
    for objective in root.findall(".//{*}Objective"):
        result["objectives"].append(dict(objective.attrib))
    for image in root.findall("./{*}Image"):
        pixels = image.find("./{*}Pixels")
        settings = image.find("./{*}ObjectiveSettings")
        result["images"].append(
            {
                "id": image.get("ID"),
                "name": image.get("Name"),
                "objective_settings": dict(settings.attrib)
                if settings is not None
                else None,
                "pixels": dict(pixels.attrib) if pixels is not None else None,
                "channels": [
                    dict(c.attrib) for c in image.findall("./{*}Pixels/{*}Channel")
                ],
                "companion_files": [
                    u.get("FileName")
                    for u in image.findall("./{*}Pixels/{*}TiffData/{*}UUID")
                    if u.get("FileName") is not None
                ],
            }
        )
    return result


def _tag(page: Any, name: str | int) -> Any:
    tag = page.tags.get(name)
    if tag is None:
        return None
    value = tag.value
    if isinstance(value, tuple):
        return list(value)
    return getattr(value, "name", value)


def _page_metadata(page: Any) -> dict[str, Any]:
    page = page.keyframe
    profile = page.tags.get(34675)
    return {
        "width": page.imagewidth,
        "height": page.imagelength,
        "samples_per_pixel": page.samplesperpixel,
        "bits_per_sample": _tag(page, "BitsPerSample"),
        "sample_format": _tag(page, "SampleFormat"),
        "photometric": getattr(page.photometric, "name", page.photometric),
        "planar_configuration": _tag(page, "PlanarConfiguration"),
        "compression": getattr(page.compression, "name", page.compression),
        "orientation": _tag(page, "Orientation"),
        "tiled": page.is_tiled,
        "tile_width": page.tilewidth if page.is_tiled else None,
        "tile_height": page.tilelength if page.is_tiled else None,
        "rows_per_strip": None if page.is_tiled else page.rowsperstrip,
        "segment_count": len(page.dataoffsets),
        "subifd_count": len(page.subifds or ()),
        "tiff_resolution": {
            "x": _tag(page, "XResolution"),
            "y": _tag(page, "YResolution"),
            "unit": _tag(page, "ResolutionUnit"),
        },
        "icc_profile": (
            {
                "bytes": len(profile.value),
                "sha256": hashlib.sha256(profile.value).hexdigest(),
            }
            if profile is not None
            else None
        ),
        "make": _tag(page, "Make"),
        "model": _tag(page, "Model"),
        "software": _tag(page, "Software"),
    }


def inspect_tiff(path: str | Path, *, checksum: bool = False) -> dict[str, Any]:
    """Inspect one source. Reported metadata does not establish native 40× detail."""
    try:
        import tifffile
    except ImportError as exc:
        raise RuntimeError(
            "Install the optional reader: pip install -e '.[io]'"
        ) from exc

    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise ValueError(f"Not a file: {path}")
    try:
        return _inspect_headers(path, tifffile, checksum)
    except tifffile.TiffFileError as exc:
        raise ValueError(str(exc)) from exc


def _inspect_headers(path: Path, reader: Any, checksum: bool) -> dict[str, Any]:
    with reader.TiffFile(path) as tif:
        series_reports = []
        for index, series in enumerate(tif.series):
            levels = []
            base_size = dict(zip(series.axes, series.shape, strict=True))
            for level_index, level in enumerate(series.levels):
                size = dict(zip(level.axes, level.shape, strict=True))
                ratios = {
                    axis: base_size[axis] / size[axis]
                    for axis in ("X", "Y")
                    if axis in size and axis in base_size and size[axis] > 0
                }
                levels.append(
                    {
                        "level_index": level_index,
                        "axes": level.axes,
                        "shape": list(level.shape),
                        "dtype": str(level.dtype),
                        "estimated_array_bytes": math.prod(level.shape)
                        * level.dtype.itemsize,
                        "dimension_ratios_to_level0": ratios,
                        "page_count": len(level.pages),
                        "representative_page": _page_metadata(level.pages[0]),
                    }
                )
            series_reports.append(
                {
                    "series_index": index,
                    "name": series.name,
                    "kind": series.kind,
                    "levels": levels,
                }
            )
        return {
            "source": {
                "path": str(path),
                "file_size_bytes": path.stat().st_size,
                "sha256": _sha256(path) if checksum else None,
            },
            "is_bigtiff": tif.is_bigtiff,
            "is_ome": tif.is_ome,
            "byte_order": tif.byteorder,
            "top_level_ifd_count": len(tif.pages),
            "reader": {"name": "tifffile", "version": reader.__version__},
            "pixel_data_decoded": False,
            "native_detail_status": "resolution_unverified",
            "calibration_status": "pending_verification",
            "ome": _ome_metadata(tif.ome_metadata),
            "series": series_reports,
            "warnings": [
                "No 40× eligibility is inferred from reported metadata.",
                "OME image/series association requires verification.",
                "Dimension ratios are not verified source-coordinate transforms.",
                "Page tags describe one representative page per level.",
            ],
        }


def _discover(paths: list[Path], recursive: bool) -> tuple[list[Path], list[str]]:
    files = set()
    skipped = set()
    for path in paths:
        if not path.exists():
            raise ValueError(f"Input does not exist: {path}")
        if path.is_dir():
            candidates = path.rglob("*") if recursive else path.iterdir()
            for candidate in candidates:
                if candidate.is_file():
                    if candidate.suffix.lower() in TIFF_SUFFIXES:
                        files.add(candidate.resolve())
                    else:
                        skipped.add(str(candidate.resolve()))
        elif path.suffix.lower() in TIFF_SUFFIXES:
            files.add(path.resolve())
        else:
            raise ValueError(f"TIFF inspection does not support this input: {path}")
    if not files:
        raise ValueError("No TIFF files found in the selected inputs.")
    return sorted(files), sorted(skipped)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="TIFF files or folders")
    parser.add_argument("--recursive", action="store_true", help="Include subfolders")
    parser.add_argument(
        "--sha256", action="store_true", help="Stream-hash each complete file"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="New JSON path; existing files are never overwritten",
    )
    args = parser.parse_args(argv)
    try:
        files, skipped = _discover(args.inputs, args.recursive)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    report: dict[str, Any] = {
        "schema_version": "0.1",
        "inspection": "metadata_only",
        "sources": [],
        "errors": [],
        "skipped_non_tiff_files": skipped,
    }
    for path in files:
        try:
            report["sources"].append(inspect_tiff(path, checksum=args.sha256))
        except (OSError, ValueError, RuntimeError, ElementTree.ParseError) as exc:
            report["errors"].append({"path": str(path), "message": str(exc)})
    try:
        serialized = (
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
        else:
            sys.stdout.write(serialized)
    except (OSError, ValueError, TypeError) as exc:
        print(f"Cannot write inspection report: {exc}", file=sys.stderr)
        return 1
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
