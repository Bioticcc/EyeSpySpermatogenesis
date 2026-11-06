# Phase 1: image characterization and region-access validation

**Status: first 13-file lab trial completed; validation remains open.** See the
2026-11-06 real-image entry in [VALIDATION.md](VALIDATION.md). JPEG region checks
passed on sampled areas; individual-tubule separation failed with the untuned
baseline. Owner authorization on
2026-11-04 covers characterization and reader/manual-region experiments. It did
not select a segmentation method or authorize starting a stage classifier. On
2026-11-06 the owner separately authorized an overview-first synthetic prototype
before the test slice arrives; see [OVERVIEW_PROTOTYPE.md](OVERVIEW_PROTOTYPE.md).
This changes implementation scope, not the requirement for real-image validation.

## Implemented now

- A TIFF file/folder metadata inspector with JSON output and optional streamed
  SHA-256 checksums. Inputs are opened read-only; image arrays are not decoded.
- An optional `io` dependency extra: tifffile 2025.6.11 and NumPy 2.2.6. These are
  trial versions for the existing Python 3.11–3.13 range, not final deployment
  choices. The [tifffile release documentation](https://pypi.org/project/tifffile/2025.6.11/)
  describes the underlying metadata/series/level APIs.
- Synthetic tests for header reporting, pyramids, source preservation and CLI
  errors; pilot and ROI request templates.
- A restricted single-plane TIFF segment reader, streamed area-mean overview,
  manual native ROIs, optional unvalidated classical candidates and native crop/
  projected-seed export. This prototype does not implement native boundary refinement.

**Not yet validated:** actual image metadata, calibration, optical detail,
companion-file completeness, large-file memory usage, bounded region decoding,
crop exports, reader comparisons or scientific performance. The common
`AnalysisRegion` contract in [ARCHITECTURE.md](ARCHITECTURE.md) remains proposed.

## Preparing the examples

Keep originals in `data/pilot/` or outside the repository. Do not resize, convert,
normalize or remove metadata before inspection. Keep multifile OME companions
together. TIFF inputs are currently supported by the inspector; other folder
formats require separate reader work after their actual formats are known.

Start with whatever is available: a complete-section OME-TIFF and existing 40×
frames are especially useful. Include different storage/acquisition variants and
a frame with several tubules if available. No tubule polygons or staging labels
are required to test I/O. This is a file-access pilot, not the later biological
evaluation dataset.

Copy `examples/pilot_manifest.csv` to a local data/results folder. It is a blank
organizational template, not an existing dataset or a file read by the inspector.
Use a unique owner-assigned `source_id` and actual file path. Add animal, section,
slide, batch and acquisition identifiers only from real records; leave unknowns
blank. Never infer biological metadata from filenames. Confirmed project scope
(mouse/PAS/FFPE/40×) does not verify an individual file's acquisition history.

Prioritize the staining protocol/counterstain, acquisition convention, physical
calibration and equipment. Remaining details can stay PENDING CONFIRMATION while
we inspect headers; see [EXPERIMENTAL_SPECIFICATIONS.md](EXPERIMENTAL_SPECIFICATIONS.md).

## Running metadata inspection

With an activated environment:

```bash
python -m pip install -e '.[dev,io]'
python -m eyespy_spermatogenesis.io.inspect_tiff data/pilot \
  --output outputs/phase1/inspection.json
```

Multiple file/folder arguments are allowed. Only directly contained TIFFs are
scanned by default; `--recursive` adds subfolders. TIFF suffixes are matched
without case sensitivity. Resolved duplicate paths are inspected once; this is
not content-based duplicate detection. Non-TIFF folder entries are listed as
skipped, not silently treated as valid images. With no `--output`, JSON goes to
standard output. Existing output files are never overwritten.

`--sha256` reads each entire file in bounded chunks without image decoding. It can
add substantial I/O on large sections; an omitted checksum is explicitly `null`.
Do not modify acquisition files during inspection. Store reports privately:
metadata may contain paths and acquisition identifiers.

Exit status is 0 when every selected TIFF was inspected, 1 for per-file inspection
or report-write errors, and 2 for invalid arguments/inputs or no TIFFs found.
Batch reports retain successful sources alongside individual errors. A zero exit
status is not a declaration of scientific suitability.

## Report contract: version 0.1

The root contains `schema_version`, `inspection`, `sources`, `errors` and
`skipped_non_tiff_files`. Each source records:

| Field | Meaning / interpretation |
|---|---|
| `source` | Resolved original path, byte size, optional SHA-256; no inferred specimen identity |
| `reader`, `is_bigtiff`, `is_ome`, `byte_order`, `top_level_ifd_count` | Reader provenance and container facts |
| `series` | Every detected series; no automatic main-image selection |
| `series[].levels` | Level index, shape, axes, dtype, estimated full-array byte count, page count |
| `dimension_ratios_to_level0` | Dimension ratios only; not verified alignment, offsets, magnification or calibration |
| `representative_page` | First page/keyframe at each level: samples, bits, photometric/planar configuration, compression, orientation, tiles/strips, segments, raw TIFF resolution, camera tags and ICC profile digest if present |
| `ome.images`, `ome.objectives` | Raw reported OME image/pixel/channel/objective attributes and companion filenames; no automatic association to TIFF series |
| `calibration_status` | Always `pending_verification` in this first tool |
| `native_detail_status` | Always `resolution_unverified`; neither metadata nor filename automatically establishes 40× detail |
| `pixel_data_decoded` | False for this metadata-only path |

OME values and unit attributes are preserved as reported strings. Missing units
remain missing in the extracted attributes; this tool does not apply schema
defaults or convert measurements. TIFF resolution tags are reported separately
and are not silently substituted for OME physical pixel sizes. Resolve units,
conflicts and X/Y spacing explicitly before physical measurements. Objective
magnification is reported metadata, not a pixel-size calculation.

Page tags are representative, not an exhaustive audit of every plane. The report's
OME attribute extractor records missing/unparsable XML and skips DTD/entity-bearing
XML. Companion references do not prove all companion files are present or correctly
mapped.
Large numbers of series/IFDs can still cost metadata parsing time and memory;
no large-image resource guarantee follows from avoiding decoded arrays.

## Next: real-image I/O before judging identification

The 2026-11-06 order is **whole-section overview → localization → native crop
retrieval → later native boundary refinement and analysis**. Manual rectangles
remain the first actual-file fidelity tests. The prototype can now run synthetic
candidate extraction, but no biological result is accepted before these checks.
The overview's factor and actual boundaries must be reviewed on the test slice.

After actual headers are reviewed, follow this sequence within Phase 1:

1. **Select source/series/plane explicitly.** Identify the actual tissue image,
   channels, orientation and native-detail level. Distinguish thumbnails and
   associated images. Resolve multidimensional and multifile layouts before
   treating anything as an ordinary RGB frame.
2. **Define manual rectangles.** Copy `examples/roi_requests.csv` into a local
   results folder and record requested series, level, Z/T and channels explicitly.
   Bounds `[x0, y0, x1, y1)` are integer pixel-edge coordinates in that requested
   level; they are not yet verified full-resolution source coordinates. Include
   a central region, image edge and storage-tile/strip boundary. Add a region
   crossing an intended analysis-tile seam when that strategy is evaluated.
3. **Trial bounded readers on the same rectangles.** Start with the inspected
   tifffile layout and compare Bio-Formats/QuPath where practical; add pyvips only
   if useful. Install only necessary trial dependencies/codecs. Do not decode a
   full highest-resolution section and slice it afterward as a claimed bounded
   reader. Record requested versus actually decoded work.
4. **Verify crop fidelity.** Keep native dtype/channel order and original colors;
   separate display-rendered previews. Compare dimensions, pixels, orientation
   and ICC handling with an independent reader/viewer. On small synthetic images,
   compare against a full decode used only as a test reference. Record any
   conversion, resampling, clipping, padding or loss of metadata.
5. **Verify provenance and resources.** Save source checksum, series/plane/level,
   bounds, crop-to-source transform, reader/version and output checksums. Check
   landmarks and round-trip coordinates, including anisotropic/noninteger pyramid
   ratios. Measure process-tree peak RAM, elapsed time and temporary disk on the
   intended hardware. Agree a resource budget before claiming acceptable behavior.
6. **Validate overview mapping, then review.** Compare whole-section overview
   landmarks, partial edge bins and several factors against native ROIs. Measure
   localization misses, touching merges/splits and true-border cuts before
   claiming useful tubule extraction. Manual crops prove image access, not
   completeness or segmentation accuracy. Once fidelity/resource checks pass,
   agree the shared production contract and scientific comparison criteria.

A crop/overview trial command now exists in
[OVERVIEW_PROTOTYPE.md](OVERVIEW_PROTOTYPE.md). It deliberately supports a restricted
set of single-plane layouts and fails on unsupported inputs; the actual test file
may require another adapter. If a useful pyramid is absent, derive an overview
from native storage segments. Never upscale coarse pixels as native crop detail.
Insufficient detail or unresolved calibration remains explicitly flagged.

## Evidence and handoff

Run `python -m pytest` with `.[dev,io]` installed. Current tests cover synthetic
TIFFs only. The no-array test blocks tifffile array-decode entry points while
inspecting a pyramidal OME-TIFF; it does not benchmark an unknown scanner format.
Local checks do not establish Windows/macOS support or hosted CI results for
unpublished changes.

When examples arrive, record actual paths/metadata and a short evidence table
here before broadening the trial reader or claiming real-image support. Do not mark Phase 1
complete until actual metadata, crop fidelity, transforms and resource behavior
have been checked. Limited Phase 2 synthetic implementation is authorized;
scientific comparison targets and Phase 4 staging remain separate decisions.
