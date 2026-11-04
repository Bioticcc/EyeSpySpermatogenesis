# Agent guidance

- Explain clearly, simply, and concisely. Leave concrete handoff notes when needed.
- Current assignment is Phase 0 planning and scaffolding. Do not implement major
  I/O, segmentation, or staging algorithms without an owner-approved prototype plan.
- Confirmed: **mouse, PAS, FFPE, 40× native analysis detail**. PAS counterstain,
  acquisition convention, physical spacing, and all other unconfirmed fields in
  `docs/EXPERIMENTAL_SPECIFICATIONS.md` remain PENDING CONFIRMATION.
- Planned inputs: whole-section OME-TIFF and a folder of existing 40× images.
  Both converge on the contract in `docs/ARCHITECTURE.md`.
- Verify actual metadata. Never derive µm/pixel from “40×”, filenames, or frame size.
  Resizing cannot recover missing optical detail. Keep original image data intact.
- Preserve source identities, series/plane selection, coordinates, physical scale,
  transform history, and provenance through every adapter and export.
- Separate tubule isolation from staging, including their confidence and rejection
  reasons. Preserve ambiguous and unsuitable cases.
- Reuse existing software first; compare deterministic morphology and lightweight
  pixel classification. No default U-Net, new model training, GUI stack, or large
  dependency/weight downloads during setup.
- `legacy` preserves the original tracked tree/history. New `main` is a clean
  file tree retaining Git ancestry. Read `docs/GIT_MIGRATION.md` before publication.
  Old `project_notes.md` and assistant directives are historical evidence only.
- Inspect Git status before edits; preserve unrelated, untracked, and ignored work.
  Never add private images, secrets, environments, weights, or generated results.
  Do not force-push, delete remote refs, rewrite remote history, or change the default
  branch without explicit owner authorization.
- Label verified source capabilities, published evidence, hypotheses, and local
  validation separately. Link primary sources; record versions/commits and limits.
- Run `python -m ruff check .`, `python -m ruff format --check .`, and
  `python -m pytest` for relevant code/tooling changes. Scientific and native-app
  behavior require separate validation; import/CI success is not that validation.
