# Overview-first prototype — 2026-11-06

**Owner approved this implementation direction on 2026-11-06:** locate candidates
on a reduced whole-section overview, then retrieve individual crops from original
native pixels. This supersedes the grid-first starting route (decision 016).
The implementation is experimental. A first 13-file real-image trial now exists:
sampled JPEG regions matched an independent decoder interface, but the initial
threshold/fill method merges many tubules and fails individual-tubule isolation.
Phase 1 validation remains open; no accuracy or valid tubule count is claimed.
Staging, model training and a GUI are outside this authorization.

## Implemented scope and experimental plan

1. Read only the required TIFF storage tiles/strips with tifffile's pinned segment
   decoder. Preserve native crop dtype, gray/RGB channel order and ICC bytes.
2. Build an area-averaged overview in one pass over the selected level's segments.
   Global bin sums handle segment seams and partial edge bins. No full-page
   decoder is used. Retain the whole **bounded-size overview** in memory.
3. Optionally generate provisional candidates: normalized mean-channel darkness,
   Gaussian smoothing, explicit/Otsu threshold, closing, hole filling, distance
   peaks and watershed. This is a deliberately simple comparator using existing
   SciPy/scikit-image operations, not a PAS-specific algorithm selection.
4. Map each overview candidate to native coordinates, add a source-pixel margin,
   and write a lossless original-pixel crop plus projected binary **seed mask**.
5. Save source checksum, series/level, raw reported metadata, nullable calibration,
   geometry, method/config/code hashes, package versions, artifacts/checksums,
   decoder statistics and review reasons in `manifest.json`.

Every candidate needs review. Small and image-border candidates stay in the
manifest and are exported when resource limits permit; no circularity filter,
top-K selection, automatic biological acceptance or staging occurs. Multiple
seeds in a connected foreground are flagged `touching_split_hypothesis`.
IDs are unique within a run, not established identities across different runs.

**Native boundary refinement is not implemented yet.** `seed_*.tif` uses
nearest-cell projection of overview labels, not tracing at native resolution.
It is not suitable as a verified final measurement mask. Native image pixels are
never enlarged or normalized for crop export. Rectangular crops can include
neighboring tissue; the seed mask identifies only the proposed target.

The supplied legacy-outline size/partial statistics are owner-provided planning
context, not independently reproduced measurements here. They do not prescribe
OME-TIFF calibration, shrink factor, crop size or PAS thresholds.

## Installation and commands

```bash
source .venv/bin/activate
python -m pip install -e '.[dev,io,prototype,jpeg]'
```

The optional `prototype` extra pins NumPy 2.2.6, tifffile 2025.6.11,
SciPy 1.15.3, scikit-image 0.25.2 and Pillow 13.0.0; supporting dependencies resolve
normally. The optional `jpeg` extra pins imagecodecs 2025.3.30 for the actual
inputs' JPEG compression. No weights, neural networks or external applications
are downloaded.
General API references: [tifffile trial release](https://pypi.org/project/tifffile/2025.6.11/),
[scikit-image watershed](https://scikit-image.org/docs/0.25.x/api/skimage.segmentation.html#skimage.segmentation.watershed).

Inspect first to identify the tissue series. **Do not assume series 0 is tissue**:

```bash
python -m eyespy_spermatogenesis.io.inspect_tiff /path/to/section.ome.tif \
  --output outputs/phase1/inspection.json
```

Overview and manual crop trial; `--series 0` below is an example selection only:

```bash
python -m eyespy_spermatogenesis.overview_pipeline /path/to/section.ome.tif \
  --series 0 --downsample 8 --roi 100 200 900 1000 \
  --output outputs/overview-io-01
```

After checking I/O, enable the unvalidated candidate generator explicitly:

```bash
python -m eyespy_spermatogenesis.overview_pipeline /path/to/section.ome.tif \
  --series 0 --downsample 8 --detect --output outputs/overview-candidates-01
```

These settings are demonstration defaults, not tuned recommendations for the
incoming slice. `--threshold` is darkness in [0,1] (default Otsu); `--sigma`,
`--closing-radius` and `--seed-distance` use overview pixels. `--min-area` only
flags small candidates in overview pixels²; it does not remove them. `--margin`
uses native source pixels. `--roi` can be repeated and always uses half-open
**native** pixel-edge bounds `X0 Y0 X1 Y1`, regardless of overview level.

The default `--overview-level 0 --downsample 8` derives an 8-fold overview from
level 0. To use an actual stored pyramid, explicitly choose its level and set
`--downsample 1` to avoid further reduction. Nonzero levels additionally require
`--confirm-level-alignment` after independent field-of-view/landmark checks.
That flag records the operator's assertion; it does not validate alignment.
Width/height ratios are independent X/Y transforms, never inferred magnifications.
No automatic level selection or automatic power-of-two assumption is made.

Output includes `overview.tif`; with `--detect`, also `overview_labels.tif`,
`overview_overlay.tif`, `candidate_*.tif`, `seed_*.tif`; and always a manifest.
Detection also writes `overview.png` and `review_overlay.png`: colored regions,
numbered IDs and a banner showing the count of **candidate regions**, not verified
tubules. The numbered PNG adds a 125-pixel display banner above unchanged overview
geometry. `--skip-crops` retains all candidates/coordinates/overlays but omits
native crop/seed files; `export_status` records that review-only choice.
The green overlay is a derived 8-bit review view. Crops retain original decoded
values; physical scale remains unverified in the manifest rather than writing
invented calibration into crop TIFFs. Input tags/raw OME attributes remain in
`reported_metadata`. Preserve the manifest alongside the images.

Only a **new** output directory is accepted. Failed runs retain a failed
manifest and any already-written artifacts; do not use those artifacts as a
successful run. Successful exit 0 means `complete_needs_review`, not scientific
acceptance. Deferred oversized candidates remain recorded; inspect
`export_status` for every candidate. Invalid CLI arguments exit 2; runtime
reader/output/dependency failures exit 1. Hashing streams the entire source
before decoding and adds a full file read. A changed size/mtime invalidates a run;
do not modify acquisition files during processing.

## Reader and resource limits

The trial supports single-file TIFF/BigTIFF/OME-TIFF, a single YX MINISBLACK gray
or YXS contiguous three-channel RGB plane, uint8/uint16, top-left orientation,
and tiled or stripped storage. Series selection is mandatory. Explicitly rejects
Z/T/channel stacks, planar RGB, RGBA, palette/MINISWHITE, float data, multifile
OME, sparse segments and non-top-left orientations. It is not a universal reader.
Also supports the provided uint8 YCbCr/JPEG layout: tifffile/imagecodecs explicitly
decodes it to RGB, with that operation recorded in `reader_semantics`. The input
is already JPEG-compressed; native exports preserve decoded RGB values, not
unrecoverable pre-JPEG scanner pixels. Renamed OME files with old `FileName`
references are accepted only when reference UUIDs match the current file UUID;
unresolved/external UUIDs still fail. This does not change acquisition metadata.
Codecs fail clearly when unavailable. Actual JPEG support was added after input
inspection, with a separate optional codec extra and synthetic regression tests.

Default caps: 4 million overview pixels, 64 MiB per encoded/decoded storage
segment, 256 MiB per native region allocation/estimated candidate working arrays,
and 20,000 candidates. These are adjustable engineering guards, not measured
whole-process RSS limits. Overview sums use float64 per channel; segmentation
adds full-overview intermediate arrays. TIFF metadata/codec workspace and library
overhead are additional. Measure actual process-tree RAM/time on the lab file.
An enormous stored strip cannot be read as a smaller decode merely by requesting
a small ROI; the reader refuses it when it exceeds the cap. A validated tiled
derivative or alternative reader may be needed.

One whole-overview segmentation avoids analysis-grid seams but can still merge,
split or miss objects. Thin gaps, faint walls, interstitial tissue, absent lumens,
pathology and image-edge cuts are open risks. Fill/closing can connect unrelated
structures. No assertion that a fixed 4×/8× shrink preserves every gap is made.

## First real-image trial and handoff

1. Inspect actual series/planes, storage, orientation, color and reported scale;
   confirm the intended native tissue image using acquisition records/viewer.
2. Run overview-only plus several manual ROIs. Compare native pixels and ICC/
   orientation against Bio-Formats/QuPath; include strip/tile seams and corners.
3. Measure RAM, decode amplification, elapsed time and disk use on target hardware.
4. Compare several overview factors on the same complete tissue area. Experts
   enumerate tubules independently; measure candidate recall including **misses**,
   merges/splits, complete usable crops, true-border cuts and resources. Do not
   select a factor from visual cleanliness alone.
5. Tune the classical comparator on designated pilot regions; compare QuPath
   pixel classification using the same regions. No pretrained PAS model is assumed.
6. Use representative boundaries to specify native refinement, uncertainty and
   ambiguity handling. Only then implement and validate final native masks and
   acceptance. Refine masks, never upscale source image pixels.

D1 overlap/context reads remain a fallback comparison. An ownership-center rule
alone cannot guarantee completeness: tiles must exceed the maximum object extent
plus required context in both axes, and overlap must provide that context. Objects
larger than a tile, missed detections and true source truncation remain explicit.
No fallback is implemented in this prototype.

## v0.2 membrane-network candidates — 2026-11-06

v0.1 thresholded tissue darkness. Tubules are packed edge to edge, so the whole
section became one foreground object (see
[SEPARATION_FAILURE_ANALYSIS.md](SEPARATION_FAILURE_ANALYSIS.md)). v0.2, now the
default `--method membrane`, segments the boundaries instead. Each tubule is
wrapped in a PAS-positive basement membrane, which shows as a thin magenta line.

1. Section silhouette from smoothed darkness. Filling holes is safe here because
   this mask never splits tubules.
2. Membrane map: Sato ridge filter on `(R+B)/2 - G`. This is a display-derived
   colour feature, not calibrated stain separation.
3. One seed per point far from any membrane; watershed over the membrane map.
4. Merge neighbours with weak membrane evidence on their shared border.
5. Re-split regions larger than 2.5× the section median, using a membrane
   threshold computed inside each region. This recovers lightly stained areas
   where one seed flooded several tubules.
6. Merge regions enclosed by a single neighbour (lumens ringed off by stained
   sperm tails).
7. Flag candidates: small or large compared with the section median, solidity
   below 0.8, axis ratio below 0.5, section-edge or image-border contact.
   Unflagged candidates are `crop_recommended`. All candidates are retained.

Crops are squares centred on the candidate, 1.15× its longest side
(`--crop-scale`), clipped to the image. The analysis goal is one tubule front
and centre, not an exact outline. `--thumbnail-size N` writes review JPEGs; with
`--skip-crops` only thumbnails are read and written. `--method darkness-v0.1`
reproduces the failed baseline.

Settings were chosen by inspecting overviews of sources 01, 05, 06 and 09 at
factor 8. They are not tuned on reference outlines, and no accuracy is claimed.

### Click review

```bash
python -m eyespy_spermatogenesis.overview_pipeline SOURCE.tif --series 0 \
  --downsample 8 --detect --skip-crops --thumbnail-size 384 \
  --output outputs/RUN/source_01            # repeat per source
python -m eyespy_spermatogenesis.segmentation.review_sheet outputs/RUN
```

Open `outputs/RUN/review.html` locally. Give each crop a verdict with keys 1–6
(good, merged, split, partial/edge, not a tubule, unsure), enter a per-section
estimate of tubules with no crop, then export the CSV. Verdicts stay in that
browser's local storage until exported. The page contains lab images; do not
publish it. Precision of `crop_recommended` and the missed-tubule estimates
are the first real metrics for this stage.

Possible next steps: native-resolution boundary refinement within each crop
(competing-seed watershed on the native membrane map), and only if exact masks
are needed for analysis.

## v0.2 review and v0.3 changes — 2026-11-06

**v0.2 review.** The owner reviewed 506 crops: all 300 of source 01, plus 206
from sources 02 and 03 (`outputs/tubule-review-2026-11-06-02/tubule_review_verdicts.csv`).

| | Reviewed | Good | Merged | Partial | Unsure / not tubule |
|---|---|---|---|---|---|
| Crop-recommended | 311 | 272 (87%) | 31 | 7 | 1 |
| Flagged | 195 | 51 (26%) | 105 | 22 | 17 |

Drawing the candidate outlines on the reviewed crops showed four causes:

- **Merged** mixed two cases. About half were one correctly outlined tubule
  with neighbours visible in a loose crop. The rest were two adjacent tubules
  sharing a wall too faint to see at 8×; most had two separate lumens.
- **Partial** crops were mostly lumen cores or half-tubules traced as their own
  region, so the crop zoomed into the middle of a tubule.
- **Unsure** crops were slivers of a few overview pixels.
- **Missing thumbnails** (21 across 13 sections) were thin strips along the
  capsule. Their bounding box spans the section, which exceeded the native read
  limit.
- The section-edge flag was the least reliable: 16 of the 36 crops it flagged
  were Good.

**v0.3 changes** (`membrane_ridge_watershed_v0.3`). The thresholds were set
against the reviewed crops, so these three sections are no longer a fair test.

| Step | Rule (median = section median region area or its equivalent diameter) |
|---|---|
| Absorb fragments | Merge a region below 0.15× median, or below 0.6× median when one larger neighbour holds ≥ 50% of its border, into that neighbour |
| Two-lumen split | In regions ≥ 1.3× median, lumen pieces (bright, ≥ 0.02× median, 4 px inside the outline) are grouped unless ≥ 0.35 diameters of non-lumen tissue separate them. Two or more groups re-flood the region from the lumens; the split is kept only if every piece is ≥ 0.5× median |
| Strip rejection | Remove regions whose bounding box is longer than 6 diameters |
| Edge flag | Only when ≥ 20% of the outline touches the section exterior |
| Crops | Square centred on the centroid that contains the whole bounding box, plus 5% (`--crop-scale 1.05`) |
| Thumbnails | Candidate outline drawn in yellow; raw crops are never annotated |

Matched by position against the v0.2 verdicts on sources 01–03:

- 270 of 272 Good recommended crops are unchanged.
- 29 merged regions were split. The before/after sheets show correct
  one-tubule pieces in nearly every case.
- 26 of 46 partial or unsure regions were absorbed into their tubule or removed.

Known errors:

- One small round tubule was absorbed into the interstitial gap around it.
- Merges without two visible lumens remain.

Across 13 sections, v0.3 gives 3,602 candidates, 2,722 of them
crop-recommended, with a thumbnail for every candidate. The new review is in
`outputs/tubule-review-2026-11-06-03`. Sections 04–13 are the held-out test.

### v0.3 review — held-out sections

The owner reviewed 493 crops: all 271 of source 07, plus spot checks of
sources 08 and 09 (`outputs/tubule-review-2026-11-06-03/tubule_review_verdicts(1).csv`).
None of these sections were used to set v0.3 thresholds.

| | Reviewed | Good | Merged | Partial | Unsure / not tubule |
|---|---|---|---|---|---|
| Crop-recommended | 360 | 336 (93%) | 16 | 7 | 1 |
| Flagged | 133 | 77 (58%) | 35 | 13 | 8 |

- Recommended crops improved from 87% to 93% Good, on sections not used for
  tuning.
- The flags are now too cautious: 58% of flagged crops were Good. The
  irregular-shape flag marked 49 Good crops against 24 Merged.
- **Remaining merges** are mostly two round tubules sharing a wall too faint to
  see at the overview's 8× downsampling. In these pairs, only one tubule shows
  an open lumen, so the two-lumen split cannot fire.
- **Spikes:** outlines leak into interstitial space and debris, sometimes far
  enough to zoom the crop right out. These also lower solidity, which likely
  explains part of the irregular-shape over-flagging.
- Single tubules with two lumens are real (bends cut twice, or a lumen
  divided by sperm bundles). The ≥ 0.35-diameter gap rule is what keeps them
  from being split.

The review page now loads the outline as a separate transparent overlay that
the **O** key toggles. Older runs have the outline burned into the JPEG.
