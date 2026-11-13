# Proposed evaluation plan

No curated pilot, expert reference set or new-project performance measurements
exist yet. This plan is to be approved before algorithm comparisons.

## Pilot collection and reference standard

An initial **planning target**, contingent on availability: 3–5 complete-section
OME-TIFFs from distinct animals plus 15–30 existing 40× frames across available
animals/batches. This is a feasibility pilot, not a statistically powered accuracy
claim. Increase independent animals for defensible generalization; many tubules
from one animal do not replace independent subjects.

Curate clean transverse, touching, partial, oblique/longitudinal, weak/strong PAS,
common artifacts and expected biological variation. Include tissue edges and
predefined internal tile seams. Record source availability and missing categories;
do not invent groups or treat a filename as biological metadata.

Store a private versioned manifest with source checksums, animal/section/slide,
lab/scanner, staining batch, calibration provenance, ROI coordinates and inclusion
rules. Keep duplicate/overlapping frames linked to the same source. Include a
random or systematic tissue sample as well as a challenge subset; challenge-only
sampling cannot estimate routine performance. Define tissue coverage before seeing
algorithm outputs so easy-region selection cannot inflate recall.

Experts independently enumerate reference tubule instances, mark completeness,
orientation, artifact/ambiguity and staging suitability. Agree a boundary definition
(including lumen and basement-membrane treatment). Annotate masks on a feasible
representative subset and record uncertainty; instance points/boxes can support
counting where mask references are unavailable. Stage labels need expert review,
taxonomy and adjudication of disagreements. Retain uncertain/mixed labels and
inter-rater agreement; never force them into a convenient gold-standard class.

## Leakage safeguards

Split at the **animal** level before selecting training tiles/crops or augmentations.
All sections, overlapping frames, serial sections, pyramid levels and derived
images from an animal stay together. Keep an external lab/scanner/staining-batch
holdout where sufficient independent data exist. Do not infer IDs from filenames;
if group identities cannot be recovered, report the leakage risk and avoid
generalization claims.

With very few animals use explicitly exploratory leave-one-animal-out evaluation;
do not claim a robust independent final test. With sufficient animals reserve
training/tuning, validation/calibration and untouched test groups. Stage balancing,
feature selection, stain normalization, thresholds, pixel classifiers, calibration,
and hyperparameter choice use training/validation groups only. Freeze manifests,
configs and matching rules before final test evaluation. Keep final test labels
hidden during tuning. Check legacy-model training exposure before including C.

## Matching and segmentation measures

Use one-to-one matching in original source coordinates. Pre-register a mask-IoU
threshold (proposed primary 0.5, sensitivity at 0.75) where full masks exist.
Use a separately agreed centroid/box rubric where they do not; label those
results distinctly rather than comparing unlike overlap scores. Never match one
prediction to several references or silently discard unmatched duplicates.

| Measure | Definition / reporting |
|---|---|
| Instance precision/recall | TP/(TP+FP), TP/(TP+FN); explicit TP, FP, FN and missed tubules per source |
| Duplicates | Extra predictions assigned to an already recovered reference / all predictions; also extras per reference; merges and splits separately |
| Mask quality | Matched-instance Dice/IoU distributions; show unmatched failures separately, not just good matched masks |
| Complete usable-crop recovery | Reference eligible tubules recovered as complete expert-approved crops / all reference eligible tubules |
| Crop acceptance precision | Usable single-target accepted crops / all accepted crops; include neighboring tissue contamination and alignment errors |
| Artifact false acceptance | Accepted artifact candidates / all accepted candidates; also fraction of enumerated artifacts accepted where reference inventory is exhaustive |
| Quality categories | Confusion matrix and per-class precision/recall against expert rubric, plus multi-reason agreement |
| Stress strata | Touching, true image/tissue edges, internal tile seams, staining, pathology, source mode and acquisition group |
| Resources and effort | Wall time, peak process-tree RSS, workers/cache, disk/temp bytes, decoded work, expert annotation/tuning time |

Mask overlap and useful-crop recovery answer different questions. Also report
rejected/ambiguous candidates and coverage; a filter must not hide missed valid
tubules. Report localization recall before segmentation for D2, and end-to-end
metrics including its missed proposals. Compare D1/D2/D3 over identical tissue
coverage with matched available native detail. A conversion's time/disk/RAM belong
in total costs, not outside the benchmark.

Use a known ROI straddling tile corners and edges to test duplicates, expanded
retrieval, clipping and source-transform round trips. Verify exported crop/mask
alignment and a small sample of original pixel values against the reader;
overlay images alone cannot prove pixel fidelity or calibration.

## Staging measures, only if reference labels exist

Report exact-stage accuracy, balanced accuracy, confusion matrix, and per-stage
precision/recall with sample counts. Score owner-approved groups separately.
Report expert-agreed neighboring-stage accuracy as secondary, with its cyclic
convention recorded. Do not compare published accuracies from different datasets
as a ranking of candidate tools.

Stratify by crop quality, animal, slide, staining batch and source where possible.
Measure uncertainty with reliability plots and Brier score (plus calibration
error with documented binning), and accuracy/coverage as abstention changes.
Evaluate isolation eligibility and staging eligibility errors independently.

Use two assessments: classifier on fixed expert-approved crops to isolate staging
behavior, and end-to-end pipeline on all eligible reference tubules to include
missed, rejected or contaminated crops. Report the denominator and rejected count
for each result. Avoid claiming classifier accuracy proves whole-section utility.

## Reproducibility and selection

Record method/config/code/project/weight hashes, software and decoder versions,
hardware/OS, random seeds, source manifest and split IDs. Use the same comparison
inputs and frozen evaluation rules. Repeat controlled timing runs with cache state
and summary statistic recorded; do not present a warmed reader against a cold one.
Repeat deterministic runs to check identical geometry and accepted IDs.

Aggregate per-animal as well as pooled results; use animal-clustered bootstrap
intervals when independent group counts permit them. Mark small-n estimates
exploratory. Require owner/expert agreement on minimum usable-crop recall,
acceptance precision, coverage, stage performance, annotation effort and resource
limits **before** selecting a winner. All numerical acceptance targets remain
PENDING CONFIRMATION.
