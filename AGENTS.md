# Agent guidance

- Explain clearly, simply, and concisely. Leave concrete handoff notes when needed.
- Current state (2026-11-06): tubule extraction uses the membrane-network
  method, `--method membrane` (v0.5, `segmentation/membrane.py`). Read
  `docs/EXTRACTION_STATUS.md` first: it gives the review results, output
  locations, the command for full-resolution crops, and known errors. The
  failed v0.1 darkness method stays available as `--method darkness-v0.1`; why
  it failed is in `docs/SEPARATION_FAILURE_ANALYSIS.md`.
- Extraction quality is measured by the owner's click reviews
  (`segmentation/review_sheet.py`). Never claim accuracy beyond those reviews.
  Tune only on reviewed sections, and name the held-out sections when
  reporting. When handing over a review page, tell the owner to save the
  exported CSV in that run's `outputs/tubule-review-*/` folder.
- Staging plan (owner, 2026-11-09; decision 023): up to five rounds, each
  ending in a model that is evaluated. Rounds 1-3 done (2026-11-11); round 4
  (hybrid) next. Round 1: a staging model on Yeongseok's paper labels only. Round 2: MATLAB paper's model;
  round 3: UNI2 features on the same labels (done 2026-11-11; **research
  only, never releasable: decision 025.** Keep the ⚠ notice on every doc or
  result that uses UNI2; never commit its weights, embeddings or heads);
  round 4: combine 2 with the best of 1/3; round 5: lab-wide manual annotation. Details and the
  round 1 label counts: `docs/STAGING_STRATEGY.md`. Model/training method
  not chosen yet; discuss with the owner first.
- At the start of each staging session run
  `.venv/bin/python scripts/check_new_staging_inputs.py`. If it lists new
  Yeongseok TIFFs or sheets, **tell the owner** (they decide on a retrain).
- Labels live on the local annotation page (`staging/annotate/`, guide
  `docs/ANNOTATION.md`, decisions 020-024). Workspace
  `outputs/staging-annotation-2026-11-06-01/`; never delete its
  `labels.sqlite`, `backups/` or `exports/`. On-screen labels were reset on
  2026-11-09; it now holds 2086 paper labels (8 sections). Paper reads are
  transcriptions (machine: `outputs/paper-alignment-2026-11-06-01`; Claude:
  `outputs/paper-alignment-2026-11-09-01`) until a person checks them.
  `pkill -f staging.annotate.server` also kills a shell whose command line
  contains that text; run it as its own command.
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
  pixel classification. No default U-Net, GUI stack, or large dependency/weight
  downloads without owner approval. Model training is authorized only for the
  staging rounds of decision 023. The optional `io` extra is the tifffile/NumPy
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
