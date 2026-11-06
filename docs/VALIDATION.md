# Phase 0 verification

Validation date: **2026-11-04**. This record covers repository/scaffold checks,
not image compatibility or biological performance.

## Environment and completed checks

- Linux, Python 3.11.2; local `.venv`.
- Editable installation: `python -m pip install -e '.[dev]'` succeeded.
- Resolved development tools: pytest 8.4.2, Ruff 0.17.3. These are the observed
  local versions, not a hash-locked deployment environment.
- Ruff lint passed; formatting check passed after formatting the import test.
- Package and all six planned submodules imported: **1 smoke test passed**.
- `pip check` found no broken requirements.
- Runtime dependency list is empty. No image/ML/GUI dependencies were installed.

The GitHub Actions workflow covers Python 3.11–3.13 on Linux. All three hosted
jobs passed for the published scaffold; local checks cover only Python 3.11.

## Archive and publication

Archival commit/tree, original-file counts and local-only inventory are in
[GIT_MIGRATION.md](GIT_MIGRATION.md). The remote archive was verified at the exact
original SHA before changing main. Full fetch of original historical objects
completed; `git fsck --full` reported no integrity errors (its unborn-main notice
was expected before the scaffold commit). All **743 reachable legacy objects**
are present, with **zero missing objects**. This is a complete local archive,
not a promise to fetch missing blobs later. The historical object transfer was
substantially larger than GitHub's reported repository size; Git reported about
963 MiB of packs plus temporary/duplicate loose objects before automatic packing.
These objects remain inside `.git`, outside the active project file tree.

The scaffold commit is `38984abe46adfa41dd70f5efc0e552f82bcad189`; its parent is
the exact legacy head. The active tree contains **28 reviewed files**, including
all requested module boundaries and 11 documents (nine requested plus migration
and validation records). No legacy implementation/data/model/output paths remain
in the active tree. Local documentation links and the owner's original note
checksum passed checks; representative mixed-case image/model paths are ignored,
while source data-model modules remain tracked.

**Remote publication is complete.** After owner-approved GitHub authorization,
a normal fast-forward push published `45ddcfa1a06166e4f476e728525e27c0ac5de7e9`
to main. Remote legacy remains at the exact original SHA. No history was rewritten
and no default-branch change was needed. See [GIT_MIGRATION.md](GIT_MIGRATION.md).

[Scaffold checks run 41918532551](https://github.com/Bioticcc/EyeSpySpermatogenesis/actions/runs/41918532551)
completed successfully for that published commit. The Python 3.11, 3.12 and 3.13
jobs each passed installation, Ruff lint/formatting and the package import test.
The publication follow-up changes only documentation; its CI runs separately.

## Scientific and platform limits

No representative image headers were inspected. No reader was validated on actual
OME-TIFF variants, no crops/masks were generated, no classifiers were run, and no
scientific accuracy or memory-budget result exists. Third-party capabilities in
the software evaluation are source/documentation evidence. No native application,
Windows/macOS/WSL deployment or disconnected installation was tested.

## Phase 1 preparation — 2026-11-04

The owner authorized Phase 1. The optional `io` extra installed successfully with
tifffile 2025.6.11 and NumPy 2.2.6 on local Python 3.11.2. The metadata inspector
and CLI are implemented; reader-specific region extraction remains pending actual
files. See [PHASE1_IO_VALIDATION.md](PHASE1_IO_VALIDATION.md).

- Full local suite: **17 tests passed**, with no skips. Cases include plain
  uint8/uint16 TIFF, tiled pyramidal OME/BigTIFF with an associated thumbnail,
  multichannel axes/orientation, ICC/file checksums, corrupt inputs, discovery,
  directory-access errors and output/source preservation.
- Ruff lint and formatting checks passed; `pip check` found no broken requirements.
- The installed module CLI also passed a synthetic smoke run from outside the
  checkout; its report checksum matched the unchanged source. A base-only Python
  invocation reported the missing optional reader clearly.
- The no-array test blocks tifffile array-decode entry points during synthetic
  pyramid inspection. These checks do not prove resource use or decoder behavior
  on an unknown scanner's images.
- CI configuration now includes the I/O extra. Hosted results above are historical
  Phase 0 evidence; no new hosted CI result is claimed for these local changes.

Representative lab files, calibration/counterstain confirmation, matched-region
crop fidelity, transforms and large-file RAM/time measurements are still pending.
Phase 1 is started, not complete. Major segmentation/staging algorithms remain
gated on an approved prototype plan.

## Overview-first prototype — 2026-11-06

The owner approved D2 overview localization and native crop retrieval before the
test slice arrives, superseding grid-first D1. This section supersedes the earlier
implementation-status descriptions above; it does not close Phase 1.
The concrete implementation/real-image plan is in
[OVERVIEW_PROTOTYPE.md](OVERVIEW_PROTOTYPE.md).

- Local Linux/Python 3.11.2 editable installation of `.[dev,io,prototype]` succeeded;
  `pip check` found no broken requirements. Trial pins: NumPy 2.2.6,
  tifffile 2025.6.11, SciPy 1.15.3 and scikit-image 0.25.2. No model/weight download.
- Full local suite: **51 tests passed**, no skips. Ruff lint and formatting passed.
  New tests verify RGB/gray uint8/uint16, compressed tiles/strips, source integrity,
  explicit series, BigTIFF/big-endian, stored pyramid selection, noninteger X/Y
  transforms, partial overview bins, native crop pixel equality, ICC retention,
  projected-seed alignment, border/small candidates, touching-split hypotheses,
  oversized-candidate retention, changed-source invalidation and output protection.
- Bounded-access test forbids full-page/series/file array-decode entry points and
  confirms a small ROI reads exactly one intersecting storage tile. Streamed
  overview tests verify one decode per stored segment and reference area means
  even when bin boundaries cross storage seams. This is bounded adapter evidence
  on synthetic layouts, not a measured scanner-file resource result.
- Installed CLI was run from `/tmp` against a separate synthetic 301×503 RGB
  tiled OME-TIFF. It produced a 76×126 overview and two provisional candidates;
  every exported crop matched source pixels exactly. All generated smoke artifacts
  remained outside tracked source. Base-only CLI help and a clear missing-extra
  error were checked separately.
- CI configuration now installs the prototype extra on Python 3.11–3.13. No hosted
  CI or Windows/macOS acceptance is claimed for these unpublished local changes.

**Still pending:** representative lab TIFF metadata, actual compression/layout
support, independent-reader crop checks, verified physical scale/native optical
detail, pyramid registration, process-tree peak RSS/time/disk, PAS candidate
precision/recall and complete usable crops. All candidates need review. Exported
seed masks are projected overview labels, not refined native boundaries; native
refinement, production batch extraction and staging are unimplemented.

## First supplied-image trial — 2026-11-06

Owner requested the initial separation algorithm on the supplied TIFF folder,
with counted individual-region overlays for manual review. All **13** supplied
files were processed without changing segmentation settings. Private paths,
per-file counts, commands, metadata and hashes are stored only in the ignored
local `outputs/tubule-review-2026-11-06-01/` report. Open `index.html` there for the
gallery, or `contact_sheet.png` for all results. No images or specimen identifiers
are added to tracked documentation.

- Actual layout: one uint8 YXS series per file, tiled 512×512, JPEG/YCbCr,
  with two stored reduced levels. Stored native arrays range from approximately
  144 to 161 million pixels. Overview generation used **native level 0, factor 8**,
  avoiding any unverified stored-pyramid alignment assumption. OME reports physical
  spacing; it remains unverified, and no physical-size/magnification cutoff ran.
- Reader adapter now supports this JPEG layout with optional imagecodecs 2025.3.30.
  JPEG decoding to RGB is explicit in provenance. Old OME filename references in
  these renamed files matched their own file UUIDs; the adapter resolves those
  self-references by UUID and still rejects external/unresolved references.
- All 13 original SHA-256 checksums were unchanged after processing.
- **39 native ROI comparisons** (three per file, including image corners and
  TIFF storage seams) matched exactly against separately assembled Pillow raw-JPEG
  tile decodes. This cross-checks decoder interface, RGB order and regional
  assembly. It is not Bio-Formats/QuPath agreement, optical calibration or a
  validation of all pixels; the decoders can share underlying JPEG libraries.
- Initial algorithm: normalized mean-RGB darkness; Gaussian sigma 1; Otsu threshold;
  disk closing radius 1; hole filling; distance transform; local maxima separated
  by 12 overview pixels; marker watershed. Area below 100 overview pixels² is
  flagged but retained. No training, new classifier, exclusion tuning or stage model.
- **Result: 564 candidate regions across 13 files, of which 471 were flagged
  small. These are not valid tubule counts.** Visual review of the overlays/contact
  sheet shows severe merging of many tubules into section-sized tissue regions
  and background/fragment candidates. The baseline does not achieve individual
  tubule isolation. The main failure is the mask merging tissue before watershed,
  consistent with threshold/closing/hole filling losing boundary information.
- Each file has a numbered color PNG overlay with candidate count, original
  overview PNG, uint32 label TIFF and full manifest. All labels, including small
  ones, appear in the count and overlay. Review-only mode omitted native crop/seed
  exports; no meaningless region crops were presented as accepted tubules.
- Local regression suite: **54 passed**, no skips; Ruff lint/format and `pip check`
  passed. JPEG/YCbCr, UUID-resolved renamed files and counted review-only PNG
  outputs now have synthetic coverage. Hosted CI is not claimed for local changes.

Next: review these failures with the owner, then compare boundary-preserving
classical masks and/or a small QuPath pixel classifier on fixed representative
regions. Do not use this trial's region totals for biology. Expert reference
instances, native masks, calibration, broader file variants, target-platform
acceptance and measured whole-process resource limits remain open.

## Failure mechanism investigation — 2026-11-06

[SEPARATION_FAILURE_ANALYSIS.md](SEPARATION_FAILURE_ANALYSIS.md) records the
subsequent diagnostic audit. All 13 saved label arrays were reproduced exactly;
source and production code hashes matched the trial. Intermediate masks on all
13 images, controlled cleanup/threshold/seed-spacing ablations on three examples,
and matched native-resolution ROI checks isolate the failed foreground/topography
assumptions. The production algorithm was not changed. Diagnostic scripts,
measurements and figures are in the ignored local
`outputs/tubule-investigation-2026-11-06-01/` folder. These tests explain failure;
they do not establish a replacement detector or scientific accuracy.

## Membrane-network method: owner click reviews — 2026-11-06

Each review gave one verdict per crop: does the candidate outline trace one
tubule? The page and verdict CSVs are local and Git-ignored, in
`outputs/tubule-review-2026-11-06-0N/`. Method details are in
[OVERVIEW_PROTOTYPE.md](OVERVIEW_PROTOTYPE.md); the current summary is in
[EXTRACTION_STATUS.md](EXTRACTION_STATUS.md).

| Run | Method | Sections reviewed | Crops | Recommended judged good | Used to tune the next version? |
|---|---|---|---|---|---|
| -02 | v0.2 | 01 (all), 02, 03 | 506 | 272/311 (87%) | yes (v0.3) |
| -03 | v0.3 | 07 (all), 08, 09 | 493 | 336/360 (93%) | yes (v0.4) |
| -04 | v0.4 | 10 (all), 11 | 389 | 258/290 (89%) | yes (v0.5) |

- Each percentage comes from sections that were not used to tune the version
  being reviewed.
- v0.5 has not had its own review. Its changes were checked only by replaying
  the -03 and -04 verdicts (see OVERVIEW_PROTOTYPE.md).
- Sections 04, 05, 06, 12 and 13 have never been reviewed.
- The per-section estimates of tubules with no crop were left blank, so recall
  (missed tubules) is unmeasured.
- These are crop-level judgements by one reviewer. They are not outline
  accuracy, and not biological validation.
