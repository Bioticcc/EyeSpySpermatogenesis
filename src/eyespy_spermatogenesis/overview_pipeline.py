"""Overview-first trial CLI; usage and limits in docs/OVERVIEW_PROTOTYPE.md."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import uuid
from collections import Counter
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from eyespy_spermatogenesis.io.inspect_tiff import _sha256, inspect_tiff


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def _write_image(path, data, *, icc_profile=None):
    import tifffile

    tifffile.imwrite(
        path,
        data,
        mode="x",
        photometric="rgb" if data.ndim == 3 else "minisblack",
        metadata={"axes": "YXS" if data.ndim == 3 else "YX"},
        iccprofile=icc_profile,
    )


OUTLINE_RGB = (255, 214, 0)


def _thumbnail_outline(labels, label, bounds, geometry, shape):
    """Candidate boundary at thumbnail size, sampled from the overview labels.

    Overview cells are several thumbnail pixels wide, so the mask is smoothed
    before thresholding to avoid a staircase outline. Review display only.
    """
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.segmentation import find_boundaries

    height, width = shape
    x0, y0, x1, y1 = bounds
    xs = x0 + (np.arange(width) + 0.5) * (x1 - x0) / width
    ys = y0 + (np.arange(height) + 0.5) * (y1 - y0) / height
    cols = np.clip((xs / geometry.scale_x).astype(int), 0, labels.shape[1] - 1)
    rows = np.clip((ys / geometry.scale_y).astype(int), 0, labels.shape[0] - 1)
    mask = (labels[rows[:, None], cols[None, :]] == label).astype(np.float32)
    cell = width / max((x1 - x0) / geometry.scale_x, 1)
    mask = ndi.gaussian_filter(mask, sigma=max(cell / 2, 0.5)) > 0.5
    edge = find_boundaries(np.pad(mask, 1), mode="inner")[1:-1, 1:-1]
    return ndi.binary_dilation(edge, iterations=1)


def _write_thumbnail(path, data, size, outline=None, outline_path=None):
    """Downscaled JPEG for review only; never an analysis input.

    ``outline(shape) -> bool mask`` gives the candidate boundary, written as a
    separate transparent PNG (``outline_path``) so the review page can toggle it.
    """
    import numpy as np
    from PIL import Image

    if data.dtype != np.uint8:
        data = np.rint(data / np.iinfo(data.dtype).max * 255).astype(np.uint8)
    image = Image.fromarray(data)
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    if outline is not None:
        overlay = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        overlay[outline((image.height, image.width))] = (*OUTLINE_RGB, 255)
        with open(outline_path, "xb") as handle:
            Image.fromarray(overlay).save(handle, format="PNG")
    with open(path, "xb") as handle:
        image.convert("RGB").save(handle, format="JPEG", quality=85)


def _fine_fetcher(native, geometry, fine_factor, max_bytes):
    """Overview bounds -> native pixels area-averaged to ``fine_factor`` px/cell.

    Used only by the membrane method's faint-wall check. Returns ``None`` for
    grayscale sources or reads above ``max_bytes``.
    """
    import numpy as np

    def fetch(bounds):
        if native.channels != 3:
            return None
        x0, y0, x1, y1 = bounds
        source = geometry.source_bounds(bounds)
        width, height = source[2] - source[0], source[3] - source[1]
        if width * height * (native.channels * native.dtype.itemsize + 8) > max_bytes:
            return None
        pixels = native.read_region(source)
        shape = ((y1 - y0) * fine_factor, (x1 - x0) * fine_factor)
        step = geometry.scale_x / fine_factor
        if step == geometry.scale_y / fine_factor and step.is_integer() and step >= 1:
            n = int(step)
            h, w = min(height // n, shape[0]), min(width // n, shape[1])
            blocks = pixels[: h * n, : w * n].reshape(h, n, w, n, -1)
            fine = blocks.mean(axis=(1, 3))
        else:
            from skimage.transform import resize_local_mean

            fine = resize_local_mean(pixels, (*shape, 3), preserve_range=True)
        return np.rint(fine).astype(pixels.dtype)

    return fetch


def _reader_stats(reader):
    return {
        "decoded_segments": reader.decoded_segments,
        "decoded_bytes_including_repeated_reads": reader.decoded_bytes,
        "peak_decoded_segment_bytes": reader.peak_segment_bytes,
    }


def run(args) -> dict:
    """Write to a new directory only; failures retain an explicit failed manifest."""
    import numpy as np

    from eyespy_spermatogenesis.io.regions import OverviewGeometry, TiffRegionReader

    source = Path(args.source).resolve(strict=True)
    original_signature = _signature(source)
    if args.overview_level and not args.confirm_level_alignment:
        raise ValueError(
            "Stored pyramid use requires --confirm-level-alignment after "
            "checking same field of view, orientation and level alignment"
        )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {
        "schema_version": "overview-prototype-0.5",
        "status": "running",
        "validation": "algorithm_unvalidated_manual_review_required",
        "config": vars(args),
        "config_sha256": hashlib.sha256(
            json.dumps(vars(args), sort_keys=True).encode()
        ).hexdigest(),
        "run_id": str(uuid.uuid4()),
        "implementation_sha256": {
            name: _sha256(Path(__file__).parent / name)
            for name in (
                "overview_pipeline.py",
                "io/regions.py",
                "io/inspect_tiff.py",
                "segmentation/overview.py",
                "segmentation/membrane.py",
                "segmentation/review.py",
            )
        },
        "source": {"path": str(source)},
        "candidates": [],
        "artifacts": {},
        "calibration_status": "pending_verification",
        "physical_spacing_um": None,
        "native_detail_status": "resolution_unverified",
        "mask_status": "projected_overview_seed_not_native_refinement",
        "versions": {n: version(n) for n in ("numpy", "tifffile")},
    }
    try:
        print("Hashing original source and inspecting metadata...", file=sys.stderr)
        checksum = _sha256(source)
        manifest["source"].update(
            {"sha256": checksum, "source_id": args.source_id or f"sha256:{checksum}"}
        )
        manifest["reported_metadata"] = inspect_tiff(source)
        options = {
            "series": args.series,
            "max_segment_bytes": args.max_segment_mib * 1024**2,
            "max_region_bytes": args.max_region_mib * 1024**2,
        }
        with (
            TiffRegionReader(source, **options) as native,
            TiffRegionReader(source, level=args.overview_level, **options) as coarse,
        ):
            if coarse.channels != native.channels or coarse.dtype != native.dtype:
                raise ValueError("Pyramid dtype/channels differ from native image")
            geometry = OverviewGeometry(
                native.width,
                native.height,
                coarse.width,
                coarse.height,
                args.downsample,
            )
            manifest["reader_semantics"] = {
                "native": {
                    "pixel_history": native.pixel_history,
                    "ome_file_references": native.ome_file_references,
                },
                "overview_level": {
                    "pixel_history": coarse.pixel_history,
                    "ome_file_references": coarse.ome_file_references,
                },
            }
            try:
                manifest["versions"]["imagecodecs"] = version("imagecodecs")
            except PackageNotFoundError:
                pass
            manifest["geometry"] = {
                **asdict(geometry),
                "overview_to_source": [geometry.scale_x, 0, 0, 0, geometry.scale_y, 0],
                "coordinate_convention": "top-left pixel edges; half-open XYXY bounds",
                "alignment": (
                    "identity_source_grid"
                    if args.overview_level == 0
                    else "operator_asserted_pyramid_alignment"
                ),
                "last_bins": "partial bins clipped to actual source extent",
                "plane": "single YX or YXS; no Z/T selection implied",
            }
            print(
                "Building area-averaged overview from TIFF segments...", file=sys.stderr
            )
            overview = coarse.overview(
                args.downsample, max_pixels=args.max_overview_pixels
            )
            _write_image(
                output / "overview.tif", overview, icc_profile=coarse.icc_profile
            )
            manifest["overview_shape"] = list(overview.shape)
            if args.roi:
                for index, bounds in enumerate(args.roi, 1):
                    crop = native.read_region(bounds)
                    name = f"manual_{index:04d}.tif"
                    _write_image(output / name, crop, icc_profile=native.icc_profile)
                    manifest.setdefault("manual_regions", []).append(
                        {
                            "bounds": bounds,
                            "crop": name,
                            "crop_to_source": [1, 0, bounds[0], 0, 1, bounds[1]],
                        }
                    )
            if args.detect:
                from skimage.segmentation import find_boundaries

                from eyespy_spermatogenesis.segmentation import membrane
                from eyespy_spermatogenesis.segmentation.overview import (
                    CandidateConfig,
                    find_candidates,
                )
                from eyespy_spermatogenesis.segmentation.review import review_overlay

                manifest["versions"].update(
                    {n: version(n) for n in ("scipy", "scikit-image", "pillow")}
                )
                print("Generating provisional overview candidates...", file=sys.stderr)
                if args.method == "membrane":
                    config = membrane.MembraneConfig(
                        seed_distance=args.seed_distance,
                        ridge_threshold=args.ridge_threshold,
                        merge_threshold=args.merge_threshold,
                    )
                    fetch = None
                    if not args.skip_wall_check:
                        fetch = _fine_fetcher(
                            native,
                            geometry,
                            config.fine_factor,
                            options["max_region_bytes"],
                        )
                    labels, candidates, diagnostics = membrane.find_tubules(
                        overview,
                        config,
                        max_candidates=args.max_candidates,
                        fetch=fetch,
                    )
                    method_name = membrane.METHOD
                else:
                    config = CandidateConfig(
                        args.threshold,
                        args.sigma,
                        args.closing_radius,
                        args.seed_distance,
                        args.min_area,
                    )
                    labels, candidates, diagnostics = find_candidates(
                        overview, config, max_candidates=args.max_candidates
                    )
                    method_name = "darkness_closing_fill_distance_watershed_v0.1"
                manifest["segmentation"] = {
                    "method": method_name,
                    "config": asdict(config),
                    "diagnostics": diagnostics,
                    "units": "overview pixels and overview pixels squared",
                    "crop_geometry": (
                        f"centroid_square_containing_bbox_x{args.crop_scale}"
                        if args.method == "membrane"
                        else f"bounds_plus_{args.margin}px_margin"
                    ),
                }
                _write_image(output / "overview_labels.tif", labels.astype(np.uint32))
                display = (
                    overview.astype(np.float32) / np.iinfo(overview.dtype).max * 255
                )
                display = display.astype(np.uint8)
                if display.ndim == 2:
                    display = np.repeat(display[..., None], 3, axis=2)
                display[find_boundaries(labels, mode="inner")] = [0, 255, 0]
                _write_image(output / "overview_overlay.tif", display)
                review = review_overlay(
                    overview,
                    labels,
                    candidates,
                    title=source.name,
                    subtitle=(
                        f"Series {args.series}; stored level {args.overview_level}; "
                        f"additional downsample {args.downsample}; {method_name}"
                    ),
                    muted_labels={c["label"] for c in candidates if c.get("flags")},
                )
                review.save(output / "review_overlay.png")
                from PIL import Image

                original_display = np.rint(
                    overview.astype(np.float32) / np.iinfo(overview.dtype).max * 255
                ).astype(np.uint8)
                Image.fromarray(original_display).save(output / "overview.png")
                manifest["counts"] = {
                    "detected_candidates": len(candidates),
                    "crop_recommended": sum(
                        bool(c.get("crop_recommended")) for c in candidates
                    ),
                    "review_required": len(candidates),
                    "reason_counts": dict(
                        Counter(reason for c in candidates for reason in c["reasons"])
                    ),
                }
                for candidate in candidates:
                    label = candidate["label"]
                    candidate["instance_id"] = (
                        f"{manifest['run_id']}/series-{args.series}/candidate-{label}"
                    )
                    if args.method == "membrane":
                        bounds = membrane.centered_square_bounds(
                            candidate, geometry, args.crop_scale
                        )
                    else:
                        bounds = geometry.source_bounds(
                            candidate["overview_bounds"], args.margin
                        )
                    candidate["source_bounds"] = list(bounds)
                    candidate["crop_to_source"] = [1, 0, bounds[0], 0, 1, bounds[1]]
                    candidate["export_status"] = "pending"
                    manifest["candidates"].append(candidate)
                    if args.skip_crops and not args.thumbnail_size:
                        candidate["export_status"] = "not_requested_review_only"
                        continue
                    # Bound combined native pixel/seed/mask arrays as well as the
                    # reader's per-array cap. This is not a process-RSS guarantee.
                    pixels = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
                    estimate = pixels * (native.channels * native.dtype.itemsize + 8)
                    if estimate > options["max_region_bytes"]:
                        candidate["export_status"] = "deferred_resource_limit"
                        candidate["reasons"].append("crop_working_arrays_exceed_limit")
                        continue
                    crop = native.read_region(bounds)
                    if args.thumbnail_size:
                        thumb_name = f"thumb_{label:05d}.jpg"
                        outline_name = f"outline_{label:05d}.png"
                        _write_thumbnail(
                            output / thumb_name,
                            crop,
                            args.thumbnail_size,
                            outline_path=output / outline_name,
                            outline=lambda shape, label=label, bounds=bounds: (
                                _thumbnail_outline(
                                    labels, label, bounds, geometry, shape
                                )
                            ),
                        )
                        candidate["thumbnail"] = thumb_name
                        candidate["thumbnail_outline"] = outline_name
                    if args.skip_crops:
                        candidate["export_status"] = "thumbnail_only"
                        continue
                    mask = (geometry.project_labels(labels, bounds) == label).astype(
                        np.uint8
                    )
                    crop_name, mask_name = (
                        f"candidate_{label:05d}.tif",
                        f"seed_{label:05d}.tif",
                    )
                    _write_image(
                        output / crop_name, crop, icc_profile=native.icc_profile
                    )
                    _write_image(output / mask_name, mask)
                    candidate.update(
                        {
                            "export_status": "written",
                            "crop": crop_name,
                            "seed_mask": mask_name,
                            "mask_status": manifest["mask_status"],
                        }
                    )
                print(
                    f"Retained {len(candidates)} candidates for review.",
                    file=sys.stderr,
                )
            manifest["reader_statistics"] = {
                "overview": _reader_stats(coarse),
                "native_crops": _reader_stats(native),
            }
        if _signature(source) != original_signature:
            raise ValueError("Source changed during processing; discard this trial")
        for path in sorted(output.iterdir()):
            if path.suffix not in (".tif", ".png", ".jpg"):
                continue
            manifest["artifacts"][path.name] = {"sha256": _sha256(path)}
        manifest["status"] = "complete_needs_review"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = str(exc)
        raise
    finally:
        manifest["elapsed_seconds"] = time.monotonic() - started
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--series", type=int, required=True)
    parser.add_argument("--output", required=True, help="New output directory")
    parser.add_argument("--source-id", help="Optional owner-assigned identity")
    parser.add_argument("--overview-level", type=int, default=0)
    parser.add_argument("--confirm-level-alignment", action="store_true")
    parser.add_argument(
        "--downsample", type=int, default=8, help="Integer area-mean factor"
    )
    parser.add_argument("--max-overview-pixels", type=int, default=4_000_000)
    parser.add_argument("--max-segment-mib", type=int, default=64)
    parser.add_argument("--max-region-mib", type=int, default=256)
    parser.add_argument(
        "--roi", type=int, nargs=4, action="append", metavar=("X0", "Y0", "X1", "Y1")
    )
    parser.add_argument(
        "--detect", action="store_true", help="Run unvalidated candidates/crops"
    )
    parser.add_argument(
        "--skip-crops",
        action="store_true",
        help="Review labels/overlays only; omit native candidate crop and seed files",
    )
    parser.add_argument(
        "--method",
        choices=("membrane", "darkness-v0.1"),
        default="membrane",
        help="membrane (v0.5, default) or the failed v0.1 darkness baseline",
    )
    parser.add_argument(
        "--threshold", type=float, help="v0.1 darkness threshold [0,1]; default Otsu"
    )
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--closing-radius", type=int, default=1)
    parser.add_argument(
        "--seed-distance",
        type=int,
        help="Minimum seed spacing, overview px (default: membrane 22, v0.1 12)",
    )
    parser.add_argument(
        "--ridge-threshold", type=float, help="Membrane ridge cutoff; default Otsu"
    )
    parser.add_argument(
        "--merge-threshold",
        type=float,
        default=0.5,
        help="Merge neighbours whose border membrane strength is below this (0-1)",
    )
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.05,
        help="Membrane crops: centred square containing the candidate, x scale",
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=0,
        help="Also write review JPEG thumbnails + outline overlays (0 = none)",
    )
    parser.add_argument(
        "--min-area", type=int, default=100, help="Flag small candidates; retain them"
    )
    parser.add_argument(
        "--margin", type=int, default=32, help="Crop margin in source pixels"
    )
    parser.add_argument(
        "--skip-wall-check",
        action="store_true",
        help="Membrane method: skip the native-pixel faint-wall check (faster)",
    )
    parser.add_argument("--max-candidates", type=int, default=20_000)
    args = parser.parse_args(argv)
    if args.seed_distance is None:
        args.seed_distance = 22 if args.method == "membrane" else 12
    if (
        any(
            v < 1
            for v in (
                args.downsample,
                args.max_overview_pixels,
                args.max_segment_mib,
                args.max_region_mib,
                args.seed_distance,
                args.min_area,
                args.max_candidates,
            )
        )
        or min(
            args.series,
            args.overview_level,
            args.closing_radius,
            args.margin,
            args.thumbnail_size,
        )
        < 0
    ):
        parser.error(
            "Invalid size/index: use positive limits and nonnegative indices/margins"
        )
    if (
        not math.isfinite(args.sigma)
        or args.sigma < 0
        or args.threshold is not None
        and not 0 <= args.threshold <= 1
        or not 0 <= args.merge_threshold <= 1
        or not math.isfinite(args.crop_scale)
        or args.crop_scale <= 0
        or args.ridge_threshold is not None
        and not math.isfinite(args.ridge_threshold)
    ):
        parser.error("Invalid smoothing, threshold, merge or crop-scale option")
    try:
        run(args)
    except ImportError as exc:
        print(
            f"Missing optional dependencies: {exc}. "
            "Install with python -m pip install -e '.[io,prototype]'",
            file=sys.stderr,
        )
        return 1
    except (OSError, ValueError, NotImplementedError) as exc:
        print(f"Overview trial failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
