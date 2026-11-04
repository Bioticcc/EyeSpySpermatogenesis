# Staging strategy

Staging is a separate consumer of isolated tubules. No classifier or feature
extractor exists in this scaffold, and no pretrained model has been downloaded.
The initial deliverable's taxonomy is **PENDING CONFIRMATION**.

## Exact and grouped outcomes

- **Exact:** stages I–XII, only if expert labels and measured resolution support
  them. Report twelve-class confusion and per-stage performance.
- **Grouped:** an intermediate outcome if owner/expert approves explicit groups.
  A model's published groups do not automatically become this project's taxonomy.
  Version the mapping and aggregate probabilities only when scientifically valid.
- **Both:** potentially retain exact probabilities with an approved grouped report.
  Never convert grouped predictions into invented exact stages.

## S1: pretrained feasibility

The evidence table and exact software details are maintained in
[SOFTWARE_EVALUATION.md](SOFTWARE_EVALUATION.md). STSP is a first Python-oriented
candidate; SATINN v2 is an important independent bright-field comparator.

STSP's study used selected, readily staged tubules rather than an unrestricted
pathology cohort; its published random image split does not establish animal-held-out
generalization. Its performance is not a target-project estimate.
See the [primary methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC12215975/).

Before inference, verify model access and terms, checksum, class order, required
crop context, dtype/color handling, normalization and scale. Compare preserved
rectangular crops and any explicitly masked representation separately; background
removal changes the input distribution. Do not assume that tighter crops improve
an existing classifier. Record any resampling as a derived model input while
retaining the original native crop.

Counterstain remains unknown. Assess fixation, staining batch, scanner and biological
differences with expert-reviewed examples. Resolve usability before proposing
training; any retraining/fine-tuning needs its own labels, grouped validation,
resources and approval. Avoid model-loader network downloads during offline runs.

## S2: interpretable morphology

Borrow feature concepts, not a claim of a wholly deterministic CSS pipeline.
The [CSS paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC8046964/) motivates
concentric epithelial regions and orientation measurements. A possible experiment
estimates boundary/lumen, radial cell organization, nuclear shape/texture,
PAS-associated structures, germ-cell distribution, and spermatid morphology and
orientation, then trains a compact classifier on measured features.

Radial/polar representations are hypotheses. Test their stability on oblique,
irregular, collapsed or lumen-poor sections. A circular center-based unwrap may
distort tissue; mark failed geometry rather than fabricating radial measurements.
Some nuclear or PAS features may be inaccessible without a confirmed counterstain
or sufficient resolution. Keep feature missingness explicit and evaluate coarse
groups and exact stages separately.

## S3: hybrid

Only compare a hybrid after separate S1/S2 baselines. Train/calibrate combination
rules on development groups, never on the test animals. Measure whether the error
patterns are complementary and whether added complexity improves held-out utility.

## Suitability and uncertainty

Incomplete, damaged, oblique, poorly stained, out-of-focus and ambiguous sections
can be retained for review without stage assignment. Isolation confidence,
quality-category confidence, stage probability and calibrated stage confidence
are different quantities. Keep each separate and preserve exclusion reasons.

Report performance among staged cases together with coverage, rejected cases,
their reference categories and the error-versus-coverage curve. A high accuracy
achieved by rejecting most tubules is not sufficient. Abnormal tubules require
expert interpretation; do not force normal-stage labels on out-of-domain tissue.

Adjacent-stage scoring is secondary and requires an agreed biological convention,
including whether XII↔I wraps cyclically and how mixed transition tubules are
handled. It must not hide poor exact-stage performance or be applied mechanically
to grouped classes. Full evaluation design is in
[EVALUATION_PLAN.md](EVALUATION_PLAN.md).
