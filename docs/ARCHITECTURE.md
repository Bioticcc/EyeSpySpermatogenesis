# Proposed architecture

**Design only.** Package modules are empty boundaries. Field names below are
proposals to validate in Phase 1, not an implemented schema or reader API.

```mermaid
flowchart TD
    A[Whole-section OME-TIFF] --> M[Inspect series, planes, scale and native detail]
    B[Folder of existing 40× images] --> F[Discover, validate and preserve originals]
    M --> D1[Option D1: overlapping native-detail regions]
    M --> D2[Option D2: coarse localization then native-detail retrieval]
    M --> D3[Option D3: whole-slide tool regions]
    D1 --> C[Shared AnalysisRegion contract]
    D2 --> C
    D3 --> C
    F --> C
    C --> I[Tubule isolation adapter: A, B or benchmark C]
    I --> R[Reconcile instances in source coordinates]
    R --> Q[Crops, masks, provenance and quality review]
    Q --> U[Unsuitable or ambiguous: retain with reasons]
    Q --> S[Suitable crops: independent staging adapter]
    S --> O[Exact or grouped stage, uncertainty or abstention]
```

## Module boundaries

| Module | Future responsibility |
|---|---|
| `io` | File discovery, metadata reports, reader selection, bounded region retrieval |
| `models` | Versioned data contracts for sources, regions, instances and results |
| `segmentation` | Candidate generation/separation, reconciliation and crop export |
| `staging` | Eligibility, classifier/features, taxonomy, uncertainty and abstention |
| `integrations` | Versioned adapters for external applications/models |
| `evaluation` | Common inputs, grouped comparisons, quality metrics and provenance checks |

Downstream modules use the shared contract. They must not branch on whether the
source was an OME-TIFF or folder. Reader and application-specific conversions
belong in adapters, with explicit records of any irreversible changes.

## Shared analysis input: proposed `AnalysisRegion`

| Field group | Required meaning |
|---|---|
| Identity | `schema_version`, `region_id`, immutable `source_id`, checksum/algorithm, original locator and acquisition ID if known |
| Pixels | Lazy reader handle/reference, selected series and Z/C/T planes, pyramid level, requested bounds; bounded arrays allowed for folder images |
| Axes | Dimensions, dtype/bit depth, channel count/order, color space, photometric interpretation and ICC identity; no silent channel dropping |
| Source geometry | Full-resolution width/height; source bounding box; `region_to_source` transform; coordinate convention and rounding policy |
| Scale | Independent X/Y µm/pixel values or `null`; units; source metadata field or calibration record; verified/conflicting/missing status |
| Acquisition | Reported magnification, its convention and provenance; native-detail eligibility and reason; never infer µm/pixel from 40× |
| Biology | Mouse, PAS, FFPE; counterstain and all other pending biological fields nullable with evidence source |
| History | Ordered preprocessing/resampling/color operations, software/config hashes, interpolation, input/output geometry and derived-image identity |
| Context | Animal/section/batch IDs if supplied; tissue/coverage references and border flags; unknown IDs remain unknown |

Metadata normalization changes representation, not image meaning. Preserve native
unannotated pixels, original bit depth and channels. Record separate display or
algorithm arrays for color conversion, rescaling, stain separation or filtering.
Do not apply H&E stain vectors to PAS without validation. Conflicting physical
spacing remains unresolved until checked; physical measurements stay null.

## Coordinate contract

- Use decoded full-resolution **source pixel-edge coordinates**, origin at the
  upper-left, X right and Y down. Boxes are half-open `[x0, y0, x1, y1)`.
  Pixel `(i,j)` has center `(i+0.5,j+0.5)`. Persist the raw TIFF orientation
  separately; any decoder rotation/flip must have an explicit transform.
- Identify series and plane in every source reference. A folder image is its own
  source with origin `(0,0)`; never invent its location on a whole slide.
- For aligned pyramids, map a level ROI to source by independent verified X/Y
  scale and offsets. `x_source = offset_x + scale_x * x_region`, likewise Y;
  derive effective ratios from actual geometry, not an assumed 2× pyramid.
  Store an affine transform where orientation/cropping requires it.
- Round retrieval bounds outward (floor minima, ceil maxima), clip to source
  extent and record padding. Keep floating geometry distinct from integer crop
  bounds. Verify mappings on landmarks and level-edge cases before use.
- Physical coordinates use verified spacing and a declared local origin. They
  are not a scanner stage-coordinate claim. Anisotropic pixels require X/Y-aware
  area and shape measurements; do not silently apply isotropic formulas.

## Input A: large OME-TIFF

Inspect headers and OME-XML before allocating pixels. Resolve multifile companions,
series, associated images and planes explicitly. Reader trials must establish
bounded decoding; a lazy-looking API is not proof of low memory use. Bound tile
caches, worker count, queue size and output buffers as well as tile dimensions.

D1 evaluates tissue/background exclusion and configurable overlapping tiles with
a context halo. Use ownership cores to nominate candidates, but do not accept an
incomplete halo-edge mask as a complete instance. D2 evaluates lower-resolution
localization separately, retaining localization misses as failures; final crops
come from verified native-detail pixels. D3 exports source-space ROIs through a
whole-slide application. Compare all three; no strategy is selected yet.

An OME-TIFF need not have a 20× level. If no useful coarse level exists, compare
bounded on-demand downsampling, a derived pyramid, or D1. A derived pyramid is
for localization only. If native detail is insufficient, stop staging eligibility
and request a suitable original; unverified detail permits characterization only.

## Boundaries, duplicates and reconciliation

Transform candidates to source coordinates before deduplication. Use overlapping
masks/contours, containment, centroid proximity and quality evidence; box overlap
alone may merge adjacent real tubules. Keep candidate-to-final-instance lineage.
If a candidate touches an internal tile edge, retrieve expanded native context
and reconcile/resegment with the same method. If still unresolved, retain the
ambiguous cluster; do not stitch incompatible partial masks into a claimed tubule.
Distinguish internal tile borders, true image/tissue borders and processing gaps.
Preserve seam diagnostics and coverage maps. No overlap size can be chosen until
observed tubule sizes and target resources are known.

## Shared outputs: proposed `TubuleInstance` and `StageResult`

Each candidate has a stable instance ID, source/region lineage, source-space mask
or contour, native bounding box and crop-to-source transform. Export an
unannotated original crop plus aligned binary mask; a masked crop is optional
and explicitly derived. Lossless TIFF is a candidate format, not a settled reader
dependency. Include measured scale, units, geometry, quality flags, method/version,
config digest, run ID and preprocessing history in versioned JSON/CSV manifests.
Use pixels/pixels² when uncalibrated; never label them µm/µm².

Proposed primary review categories: accepted transverse; oblique/longitudinal;
partial/border-truncated; touching/ambiguous; artifact/rejected. Retain multiple
reason flags because conditions overlap. Biological review must refine these;
irregular shape or absent lumen alone is not proof of artifact or pathology.

Keep isolation measurements/confidence separate from staging eligibility,
probabilities, calibrated confidence, model identity, taxonomy version and
abstention reason. Raw scores need not be probabilities. Stage fields are null
for ineligible/unrun cases; they are never encoded as stage zero.

## Reproducibility and integration

Future runs store checksummed source manifests, explicit file ordering, config,
tool versions, models/checksums, random seeds where applicable, logs, cache/resource
limits and output checksums. External jobs use argument arrays, explicit working
directories, exit-code/log capture and cancellation. A successful exit is followed
by output shape/coordinate checks. Training annotations and tool projects are
versioned artifacts outside Git until approved; none are bundled automatically.
