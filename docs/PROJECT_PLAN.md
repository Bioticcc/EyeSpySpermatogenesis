# Project plan

## Mission and scope

Develop reproducible, preferably offline analysis of mouse PAS/FFPE histology at
native 40×-equivalent detail. Separate **A: tubule isolation** from **B: epithelial
staging**. Reuse validated software and mature algorithms before building new
models. Confirmed requirements are in
[EXPERIMENTAL_SPECIFICATIONS.md](EXPERIMENTAL_SPECIFICATIONS.md).

Phase 0 provides plans and a minimal Python scaffold. There is no image-analysis
implementation, GUI, model installation, or demonstrated scientific performance.
The clean main is published on GitHub, with the original project preserved on
legacy; see [GIT_MIGRATION.md](GIT_MIGRATION.md).

## Phases and exit evidence

| Phase | Work | Exit evidence / approval gate |
|---|---|---|
| 0 — current | Archive legacy; clean active tree; review software; specify shared contracts and comparisons | Verified archive ref, reviewed docs, package import/lint checks; no substantive algorithms |
| 1 | Characterize OME-TIFFs and folder images; verify scale/counterstain; test bounded readers; agree pilot and contracts | Metadata reports, matched-region comparisons, peak memory/time, source transforms; owner approves prototype plan |
| 2 | Compare A/B segmentation, legacy C if fair, and whole-section D1/D2/D3 | Same grouped pilot, complete-crop and instance metrics, boundary stress tests; documented selection |
| 3 | Implement chosen batch extraction and exports; explicit ambiguity and quality handling | Independently reviewed crop outputs, repeatable runs, coordinate/provenance checks |
| 4 | Evaluate S1 pretrained staging, S2 morphology, S3 hybrid; exact vs grouped taxonomy | Held-out animal-level evaluation, coverage/uncertainty and failure analysis; no unsupported staging claim |
| 5 | Integrate both inputs, batch jobs, progress, exports, offline distribution; assess GUI only after core validation | Target-platform and operator acceptance, installation/offline tests, end-to-end scientific validation |

## Exact entry steps for Phase 1

1. Owner authorizes image characterization and bounded-reader experiments; no
   segmentation implementation is implied by that authorization.
2. Supply local paths to representative complete-section OME-TIFFs and existing
   40× folder images. Keep them outside tracked files. Multiple animals/acquisition
   batches are desirable; availability is unknown. No dataset is assumed to exist.
3. Supply the staining protocol, acquisition records, and any existing calibration,
   animal/section identifiers, annotations and expert labels. Prioritize PAS
   counterstain and the meaning of 40×. Record remaining unknowns explicitly.
4. Agree target OS/RAM/hardware and whether initial staging must be exact, grouped,
   or both. These choices need not block metadata inspection.
5. Inventory checksums, axes, dimensions, series/planes, bit depth, channel/color
   semantics, compression, tiles/strips, pyramid levels, and physical spacing.
   Handle associated thumbnails separately; never select a series just by index.
6. Test tifffile metadata plus bounded TIFF access against Bio-Formats/QuPath on
   small, identical ROIs; optionally compare pyvips. Measure RAM, decoding work,
   speed, and coordinate/color agreement. Install only the selected trial tools
   in isolated environments after version review.
7. Draft the pilot manifest, grouped splits and expert annotation rubric. Validate
   the proposed contracts in [ARCHITECTURE.md](ARCHITECTURE.md) against actual data.
8. Review evidence with owner/expert and approve the specific Phase 2 prototype
   configuration, resource budget, dependencies, and success thresholds.

## Handoff

Future agents should read `AGENTS.md`, the experimental specifications, migration
record, software evaluation and decision log. Cite the legacy commit when using
historical evidence. Preserve local owner notes and all private files. Never
interpret a Phase 0 smoke test as validation of I/O, algorithms, or biology.
