# Project plan

## Mission and scope

Develop reproducible, preferably offline analysis of mouse PAS/FFPE histology at
native 40×-equivalent detail. Separate **A: tubule isolation** from **B: epithelial
staging**. Reuse validated software and mature algorithms before building new
models. Confirmed requirements are in
[EXPERIMENTAL_SPECIFICATIONS.md](EXPERIMENTAL_SPECIFICATIONS.md).

Owner-confirmed order, updated 2026-11-06: **whole-section overview → provisional
tubule localization → native original-pixel crop retrieval → later boundary
refinement and analysis**. D2 supersedes the original grid-first D1 route.

Phase 0 is complete. Phase 1 real-image characterization and I/O validation remain
open. The owner authorized an overview-first prototype before the test TIFF
arrives: a restricted segment reader, streamed overview, classical provisional
candidates, native crops, projected seed masks and provenance now exist with
synthetic tests. On real sections, the membrane-network method (v0.5) now
gives one-tubule crops that the owner judged good in about 9 of 10 recommended
cases on held-out sections. See [extraction status](EXTRACTION_STATUS.md).
Native boundary refinement, batch adapters and a GUI remain future work.
Staging now follows a five-round plan (owner, 2026-11-09; decision 023); round 1
trains on Yeongseok's paper annotations only. See
[the staging strategy](STAGING_STRATEGY.md#plan-five-rounds-owner-2026-11-09).
See [the prototype plan](OVERVIEW_PROTOTYPE.md) and
[the Phase 1 guide](PHASE1_IO_VALIDATION.md).
The clean main is published on GitHub, with the original project preserved on
legacy; see [GIT_MIGRATION.md](GIT_MIGRATION.md).

## Phases and exit evidence

| Phase | Work | Exit evidence / approval gate |
|---|---|---|
| 0 — complete | Archive legacy; clean active tree; review software; specify shared contracts and comparisons | Verified archive ref, reviewed docs and passing scaffold CI |
| 1 — current | Characterize OME-TIFFs and folder images; verify scale/counterstain; test bounded readers and manual crops; agree pilot and contracts | Metadata reports, matched-region comparisons, peak memory/time, source transforms; owner approves Phase 2 prototype plan |
| 2 — prototype started | Synthetic A-D2 implementation authorized; later compare A/B, legacy C if fair, and D1/D3 | Same grouped pilot, complete-crop and instance metrics, boundary stress tests; documented selection |
| 3 | Implement chosen batch extraction and exports; explicit ambiguity and quality handling | Independently reviewed crop outputs, repeatable runs, coordinate/provenance checks |
| 4 | Evaluate S1 pretrained staging, S2 morphology, S3 hybrid; exact vs grouped taxonomy | Held-out animal-level evaluation, coverage/uncertainty and failure analysis; no unsupported staging claim |
| 5 | Integrate both inputs, batch jobs, progress, exports, offline distribution; assess GUI only after core validation | Target-platform and operator acceptance, installation/offline tests, end-to-end scientific validation |

## Exact entry steps for Phase 1

1. Owner authorized image characterization and bounded-reader experiments on
   2026-11-04. The 2026-11-06 update separately authorized the limited overview-first
   prototype; it does not establish real-image validation or authorize staging.
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
   comparison configuration, resource budget and scientific success thresholds.
   The initial synthetic implementation plan is in `OVERVIEW_PROTOTYPE.md`.

## Handoff

Future agents should read `AGENTS.md`, the experimental specifications, migration
record, software evaluation and decision log. Cite the legacy commit when using
historical evidence. Preserve local owner notes and all private files. Never
interpret a Phase 0 smoke test as validation of I/O, algorithms, or biology.
