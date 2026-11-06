"""Experimental, single-plane TIFF region access without full-page decoding.

This adapter intentionally supports only YX gray and YXS contiguous RGB uint8/16
images. It uses the segment decoder in the pinned tifffile trial version.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import tifffile

from eyespy_spermatogenesis.io.inspect_tiff import _ome_metadata


@dataclass(frozen=True)
class OverviewGeometry:
    """Pixel-edge transform; final partial bins are clipped to source extent."""

    source_width: int
    source_height: int
    level_width: int
    level_height: int
    factor: int

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in vars(self).values()):
            raise ValueError("Geometry dimensions and factor must be positive integers")
        if (
            self.level_width > self.source_width
            or self.level_height > self.source_height
        ):
            raise ValueError("Overview level cannot be larger than the native source")

    @property
    def scale_x(self) -> float:
        return self.factor * self.source_width / self.level_width

    @property
    def scale_y(self) -> float:
        return self.factor * self.source_height / self.level_height

    def source_bounds(self, bounds, margin: int = 0) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = bounds
        return (
            max(0, math.floor(x0 * self.scale_x) - margin),
            max(0, math.floor(y0 * self.scale_y) - margin),
            min(self.source_width, math.ceil(x1 * self.scale_x) + margin),
            min(self.source_height, math.ceil(y1 * self.scale_y) + margin),
        )

    def project_labels(self, labels, bounds) -> np.ndarray:
        """Nearest-cell seed mask projection, never image upsampling."""
        x0, y0, x1, y1 = bounds
        xs = np.floor((np.arange(x0, x1) + 0.5) / self.scale_x).astype(int)
        ys = np.floor((np.arange(y0, y1) + 0.5) / self.scale_y).astype(int)
        return labels[ys[:, None], xs[None, :]]


class TiffRegionReader:
    """Read selected strips/tiles only, with explicit decode/allocation caps."""

    def __init__(
        self,
        path: str | Path,
        *,
        series: int,
        level: int = 0,
        max_segment_bytes: int = 64 * 1024**2,
        max_region_bytes: int = 256 * 1024**2,
    ):
        self.path = Path(path).resolve(strict=True)
        if min(max_segment_bytes, max_region_bytes) < 1:
            raise ValueError("Reader byte limits must be positive")
        self.max_segment_bytes = max_segment_bytes
        self.max_region_bytes = max_region_bytes
        self.decoded_segments = 0
        self.decoded_bytes = 0
        self.peak_segment_bytes = 0
        self._tiff = tifffile.TiffFile(self.path, _multifile=False)
        try:
            if series < 0 or series >= len(self._tiff.series):
                raise ValueError("Select an existing series explicitly")
            selected = self._tiff.series[series]
            if level < 0 or level >= len(selected.levels):
                raise ValueError("Requested pyramid level does not exist")
            self.series_index, self.level_index = series, level
            selected = selected.levels[level]
            self.axes = selected.axes
            if self.axes not in ("YX", "YXS") or len(selected.pages) != 1:
                raise ValueError(
                    "Trial reader requires a single YX or YXS plane; "
                    "Z/T/channel stacks and planar RGB need another adapter"
                )
            self.page = selected.pages[0]
            self.dtype = np.dtype(selected.dtype)
            if self.dtype.kind != "u" or self.dtype.itemsize not in (1, 2):
                raise ValueError("Trial reader supports only uint8 and uint16")
            page = self.page
            if page.tags.get("Orientation") and page.tags["Orientation"].value != 1:
                raise ValueError(
                    "Non-top-left TIFF orientation needs a tested transform"
                )
            self.channels = page.samplesperpixel
            if not (
                (self.axes == "YX" and self.channels == 1 and page.photometric == 1)
                or (
                    self.axes == "YXS"
                    and self.channels == 3
                    and (
                        page.photometric == 2
                        or (
                            page.photometric == 6
                            and page.compression == 7
                            and self.dtype.itemsize == 1
                        )
                    )
                    and page.planarconfig == 1
                )
            ):
                raise ValueError(
                    "Trial reader requires MINISBLACK gray, RGB, or uint8 YCbCr JPEG"
                )
            self.pixel_history = (
                ["JPEG YCbCr decoded to RGB by tifffile/imagecodecs"]
                if page.photometric == 6
                else []
            )
            ome = _ome_metadata(self._tiff.ome_metadata)
            if self._tiff.is_ome and ome["warnings"]:
                raise ValueError("OME metadata could not be safely interpreted")
            self.ome_file_references = []
            if self._tiff.ome_metadata and not ome["warnings"]:
                root = ElementTree.fromstring(self._tiff.ome_metadata)
                source_uuid = root.get("UUID")
                for reference in root.findall(".//{*}TiffData/{*}UUID"):
                    name = reference.get("FileName")
                    reference_uuid = (reference.text or "").strip()
                    by_uuid = bool(source_uuid and reference_uuid)
                    is_self = (
                        reference_uuid == source_uuid
                        if by_uuid
                        else name in (None, self.path.name)
                    )
                    if not is_self:
                        raise ValueError(
                            "Multifile OME needs a separately validated adapter"
                        )
                    self.ome_file_references.append(
                        {
                            "reported_filename": name,
                            "uuid": reference_uuid or None,
                            "resolved_as": "current_file_uuid"
                            if by_uuid
                            else "current_filename",
                        }
                    )
            self.height, self.width = page.imagelength, page.imagewidth
            self.block_height = page.tilelength if page.is_tiled else page.rowsperstrip
            self.block_width = page.tilewidth if page.is_tiled else self.width
            self.columns = math.ceil(self.width / self.block_width)
            expected = self.columns * math.ceil(self.height / self.block_height)
            if len(page.dataoffsets) != expected:
                raise ValueError("Unsupported TIFF segment layout")
            segment_bytes = (
                self.block_height
                * self.block_width
                * self.channels
                * self.dtype.itemsize
            )
            if segment_bytes > max_segment_bytes:
                raise ValueError(
                    "A storage strip/tile exceeds max_segment_bytes; "
                    "use a smaller stored level or validated tiled derivative"
                )
            if max(page.databytecounts) > max_segment_bytes:
                raise ValueError("An encoded TIFF segment exceeds max_segment_bytes")
            if any(
                o == 0 or n == 0 for o, n in zip(page.dataoffsets, page.databytecounts)
            ):
                raise ValueError("Sparse/missing TIFF segments are unsupported")
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self._tiff.close()

    @property
    def icc_profile(self):
        tag = self.page.tags.get(34675)
        return tag.value if tag else None

    def _segment(self, index):
        page = self.page
        handle = self._tiff.filehandle
        handle.seek(page.dataoffsets[index])
        size = page.databytecounts[index]
        encoded = handle.read(size)
        if len(encoded) != size:
            raise ValueError("Truncated TIFF segment")
        options = {}
        if page.compression in {6, 7, 34892, 33007}:
            options = {"jpegtables": page.jpegtables, "jpegheader": page.jpegheader}
        decoded, position, _ = page.decode(encoded, index, **options)
        if decoded is None:
            raise ValueError("Missing decoded TIFF segment")
        y = index // self.columns * self.block_height
        x = index % self.columns * self.block_width
        if position != (0, 0, y, x, 0):
            raise ValueError("Unexpected decoder coordinates")
        self.decoded_segments += 1
        self.decoded_bytes += decoded.nbytes
        self.peak_segment_bytes = max(self.peak_segment_bytes, decoded.nbytes)
        return x, y, decoded[0, : self.height - y, : self.width - x, :]

    def read_region(self, bounds) -> np.ndarray:
        """Exact original pixels inside an in-bounds half-open rectangle."""
        if len(bounds) != 4 or any(type(v) is not int for v in bounds):
            raise ValueError("Bounds must be four integer pixel-edge coordinates")
        x0, y0, x1, y1 = bounds
        if not (0 <= x0 < x1 <= self.width and 0 <= y0 < y1 <= self.height):
            raise ValueError("Requested region lies outside the selected image")
        shape = (y1 - y0, x1 - x0, self.channels)
        if math.prod(shape) * self.dtype.itemsize > self.max_region_bytes:
            raise ValueError("Requested crop exceeds max_region_bytes")
        result = np.empty(shape, dtype=self.dtype)
        for row in range(y0 // self.block_height, (y1 - 1) // self.block_height + 1):
            for col in range(x0 // self.block_width, (x1 - 1) // self.block_width + 1):
                x, y, data = self._segment(row * self.columns + col)
                left, top = max(x, x0), max(y, y0)
                right, bottom = min(x + data.shape[1], x1), min(y + data.shape[0], y1)
                result[top - y0 : bottom - y0, left - x0 : right - x0] = data[
                    top - y : bottom - y, left - x : right - x
                ]
        return result[..., 0] if self.channels == 1 else result

    def overview(self, factor: int, *, max_pixels: int = 4_000_000) -> np.ndarray:
        """Area mean in anchored factor×factor bins, one storage-segment pass.

        Edge bins average only actual pixels. Sums are float64; the returned
        derived image is rounded to the source dtype. No full native array.
        """
        if type(factor) is not int or factor < 1 or max_pixels < 1:
            raise ValueError(
                "Overview factor and pixel limit must be positive integers"
            )
        height, width = math.ceil(self.height / factor), math.ceil(self.width / factor)
        if height * width > max_pixels:
            raise ValueError("Overview exceeds max_pixels; increase downsample factor")
        sums = np.zeros((height, width, self.channels), dtype=np.float64)
        for index in range(len(self.page.dataoffsets)):
            x, y, data = self._segment(index)
            # Reduce partial cells inside each segment, then accumulate globally.
            # This works even when TIFF segment edges do not align with bins.
            rows = np.arange(y, y + data.shape[0]) // factor
            cols = np.arange(x, x + data.shape[1]) // factor
            row_starts = np.r_[0, np.flatnonzero(np.diff(rows)) + 1]
            col_starts = np.r_[0, np.flatnonzero(np.diff(cols)) + 1]
            reduced = np.add.reduceat(data, row_starts, axis=0, dtype=np.float64)
            reduced = np.add.reduceat(reduced, col_starts, axis=1)
            sums[rows[row_starts, None], cols[col_starts]] += reduced
        bin_widths = np.minimum(factor, self.width - np.arange(width) * factor)
        bin_heights = np.minimum(factor, self.height - np.arange(height) * factor)
        for row in range(height):
            sums[row] /= (bin_widths * bin_heights[row])[:, None]
        np.rint(sums, out=sums)
        result = sums.astype(self.dtype)
        return result[..., 0] if self.channels == 1 else result
