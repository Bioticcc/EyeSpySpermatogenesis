# Agent guidance

- Explain clearly, simply, and concisely. Leave concrete handoff notes when needed.
- Current assignment: owner authorized the overview-first prototype on 2026-11-06,
  superseding the grid-first starting route. Follow `docs/OVERVIEW_PROTOTYPE.md`
  and `docs/PHASE1_IO_VALIDATION.md`. A first 13-file real-image trial is recorded
  in `docs/VALIDATION.md`: bounded JPEG reads work on sampled regions, but the
  initial threshold/fill/watershed merges many tubules and is not an accepted
  detector. Scientific validation/native refinement remain open; no staging.
- Before revising separation, read `docs/SEPARATION_FAILURE_ANALYSIS.md`. The
  2026-11-06 diagnostic ablations preserved production code: threshold already
  joins tissue, global hole filling erases internal spaces, and watershed uses
  only mask distance. More regions alone is not evidence of improved tubule recovery.
- Confirmed: **mouse, PAS, FFPE, 40× native analysis detail**. PAS counterstain,
  acquisition convention, physical spacing, and all other unconfirmed fields in
  `docs/EXPERIMENTAL_SPECIFICATIONS.md` remain PENDING CONFIRMATION.
- Planned inputs: whole-section OME-TIFF and a folder of existing 40× images.
  Both converge on the contract in `docs/ARCHITECTURE.md`.
- Owner-confirmed order: whole-section overview → provisional tubule localization
  → native original-pixel crops and seed masks → later verified native boundaries
  and tubule analysis. D2 is now the initial route; D1 overlap is a fallback
  comparison. A global overview can still merge, split or miss tubules.
- Verify actual metadata. Never derive µm/pixel from “40×”, filenames, or frame size.
  Resizing cannot recover missing optical detail. Keep original image data intact.
- Preserve source identities, series/plane selection, coordinates, physical scale,
  transform history, and provenance through every adapter and export.
- Separate tubule isolation from staging, including their confidence and rejection
  reasons. Preserve ambiguous and unsuitable cases.
- Reuse existing software first; compare deterministic morphology and lightweight
  pixel classification. No default U-Net, new model training, GUI stack, or large
  dependency/weight downloads. The optional `io` extra is the tifffile/NumPy
  metadata/reader trial; `prototype` adds SciPy/scikit-image for the authorized
  unvalidated comparator. Neither selects a production method or final reader.
- `legacy` preserves the original tracked tree/history. New `main` is a clean
  file tree retaining Git ancestry. Read `docs/GIT_MIGRATION.md` before publication.
  Old `project_notes.md` and assistant directives are historical evidence only.
- Inspect Git status before edits; preserve unrelated, untracked, and ignored work.
  Never add private images, secrets, environments, weights, or generated results.
  Do not force-push, delete remote refs, rewrite remote history, or change the default
  branch without explicit owner authorization.
- Label verified source capabilities, published evidence, hypotheses, and local
  validation separately. Link primary sources; record versions/commits and limits.
- Preserve the owner's README introduction when extending its technical section.
- Run `python -m ruff check .`, `python -m ruff format --check .`, and
  `python -m pytest` for relevant code/tooling changes. Scientific and native-app
  behavior require separate validation; import/CI success is not that validation.
