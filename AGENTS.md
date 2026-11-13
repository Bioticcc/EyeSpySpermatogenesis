# Agent guidance

- Explain clearly, simply, and concisely. Leave concrete handoff notes when needed.
- Tubule extraction (current since 2026-11-06) uses the membrane-network
  method, `--method membrane` (v0.5, `segmentation/membrane.py`). Read
  `docs/EXTRACTION_STATUS.md` first: it gives the review results, output
  locations, the command for full-resolution crops, and known errors. The
  failed v0.1 darkness method stays available as `--method darkness-v0.1`; why
  it failed is in `docs/history/SEPARATION_FAILURE_ANALYSIS.md`.
- Extraction quality is measured by the owner's click reviews
  (`segmentation/review_sheet.py`). Never claim accuracy beyond those reviews.
  Tune only on reviewed sections, and name the held-out sections when
  reporting. When handing over a review page, tell the owner to save the
  exported CSV in that run's `outputs/tubule-review-*/` folder.
- Staging (owner, 2026-11-13; decision 027): the primary strategy is
  **EyeSpy** (`staging/eyespy.py`): frozen H-optimus-0 + DINOv2-L features
  (releasable) with our own cyclic ridge classifier, trained on the annotation
  page's labels. All new tests use EyeSpy. Every saved model goes in
  `Model_Checkpoints/registry.json` (Git-ignored, never overwritten) as
  `EyeSpy_vN` plus `- Smoothed (k, λ)` and `- Class weighted` (comparison)
  variants; experiments are tagged. **EyeSpy_v1 (paper labels only) is the
  model in use**; v2 and the label experiments with Yeongseok's on-screen
  labels were worse (the on-screen labels run about half a stage later), so
  check those labels before adding more. SATINN (round 2) and UNI2 (round 3;
  **research only, never releasable: decision 025.** Keep the ⚠ notice on
  every doc or result that uses UNI2; never commit its weights, embeddings or
  heads) stay as reference models; round 4 is paused. Details:
  `docs/STAGING_STRATEGY.md`; commands: `docs/ANNOTATION.md`.
- At the start of each staging session run
  `.venv/bin/python scripts/check_new_staging_inputs.py`. If it lists new
  Yeongseok TIFFs or sheets, **tell the owner** (they decide on a retrain).
- Labels live on the local annotation page (`staging/annotate/`, guide
  `docs/ANNOTATION.md`, decisions 020-024). Workspace
  `outputs/staging-annotation-2026-11-06-01/`; never delete its
  `labels.sqlite`, `backups/` or `exports/`. On-screen labels were reset on
  2026-11-09; it holds 2086 paper labels (8 sections) and, from 2026-11-12,
  Yeongseok's on-screen labels (#105 complete, #106 started). Back up
  `labels.sqlite` (sqlite backup API, into `backups/`) before writing to it
  from a script. Paper reads are
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
- `legacy` preserves the original code and history. New `main` is a clean
  file tree retaining Git ancestry. On 2026-11-13 the owner had lab data
  (raw TIFFs, annotations, `.pt` weights) removed from all history, keeping two
  example sections and the overlay PNGs; see `docs/history/GIT_MIGRATION.md`. Read `docs/history/GIT_MIGRATION.md` before publication.
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
