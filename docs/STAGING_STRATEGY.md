# Staging strategy

Staging is a separate consumer of isolated tubules. No classifier or feature
extractor exists yet, and no pretrained model has been downloaded. Label
taxonomy: 11 classes (I, II–III, IV–XII) plus "can't stage" (decision 020).

## Plan: five rounds (owner, 2026-11-09)

Staging proceeds in up to five rounds (decision 023; the UNI2 round was added
the same day). Each round uses its own source of labels or model and ends by
training (or adopting) a model and evaluating it. If the result is unsatisfactory, the next round starts.

| Round | Labels / model | Status |
|---|---|---|
| 1 · **Baseline** | **Yeongseok's hand-staged paper sheets only** (`Inputs/Staging annotations/`, sections in `Inputs/YeongSeok_Staging/`), transferred onto v0.5 crops. Train a staging model on these labels alone. | Done 2026-11-09: 80% exact, 70% balanced, 99.7% within one stage (leave-one-section-out; below). Expert review pending. |
| 2 · **SATINN** | SATINN v2's pretrained bright-field stager (MATLAB paper), used alone, no training on our labels. | First run 2026-11-09 (label-free preprocessing choice): 31% exact, 52% within one stage (below). Windows-app reference check still open (deferred by the owner 2026-11-11). Fine-tuned on our labels 2026-11-11: 75% exact, 49% balanced ([opt-in experiments](#opt-in-experiments-2026-11-11)). |
| 3 · **UNI2** ⚠ | UNI2 (`MahmoodLab/UNI2-h`) as a frozen feature extractor, trained on the same Yeongseok paper labels as round 1. | Done 2026-11-11: UNI2 alone 81% exact / 72% balanced (no significant difference from round 1); UNI2 + H-optimus-0 ensemble 84% / 75% (below). Expert review pending. **⚠ Research only, never releasable (decision 025).** |
| 4 · **Hybrid** | Combine the round 2 model with our own model from the Yeongseok labels (best of rounds 1 and 3); tune weights/settings to beat the best of rounds 1–3. | After rounds 1–3. |
| 5 · **Manual** | All lab members annotate as many tubules as possible on the annotation page. | Only if rounds 1–4 fail. |

Feature extractor licences (checked on the Hugging Face model pages,
2026-11-09; recheck before release):

| Model | Licence | Free to use and publish? |
|---|---|---|
| H-optimus-0 (Bioptimus, 1.1B, 224 px at 0.5 µm/px) | Apache-2.0, gated (share HF info) | Yes |
| DINOv2 (Meta; not histology-specific) | Apache-2.0 | Yes |
| Prov-GigaPath (Microsoft) | Apache-2.0, gated; card says research use only, deployment out of scope | Licence yes; card discourages deployment |
| UNI2-h (Mahmood lab) | CC-BY-NC-ND-4.0, gated | Research only; no redistribution |
| Virchow2 (Paige) | CC-BY-NC-ND-4.0, gated | Research only; no redistribution |
| Phikon-v2 (Owkin) | Owkin non-commercial | Research only |

Round 1 rules:

- Train only on `origin = paper` labels (annotator Yeongseok). On-screen labels
  were reset on 2026-11-09 (decision 024); round 1 does not use them.
- Paper reads are transcriptions (machine or Claude), not yet checked by a
  person. Readings flagged "to check" and unreadable marks stay out of the
  first training set unless someone checks them.
- Yeongseok may add sheets and TIFFs. Run
  `.venv/bin/python scripts/check_new_staging_inputs.py` at the start of a
  session; if it lists anything, tell the owner, who decides whether to
  retrain round 1 with the new labels.
- Hold out whole sections (animals) for evaluation. Sections show only a few
  stages each (synchronised zones), so a split by section changes which stages
  the test set contains; check per-stage counts in both splits.

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

## Paper annotation transfer (trial, 2026-11-06)

Existing hand-staged printouts can be transferred to v0.5 candidates (decision
022). The trial on `#77_WR2_50d_7-1` registered a phone photo of the printout to
the overview with SIFT + RANSAC (perspective; mean residual ~1.5 overview px)
and assigned 285 circled numerals to candidates. Numerals are read by a
sheet-specific rule (ink height for X / bar count) and are **suggestions to
confirm**, not verified labels: the X-plus-bars cases (XI vs XII) are the main
misreads. Marks shared by one candidate expose v0.5 merges. The trial script
and notes live in `outputs/paper-alignment-2026-11-06-01/` (Git-ignored);
promote it into the package once more sheets confirm the approach.

This WR2 section shows only about four stages (I, II–III, XI, XII), grouped in
zones, which looks like synchronised spermatogenesis (owner to confirm). Such
sections give many examples of few stages. Track label counts per stage across
sections before training.

### Sheets 2–8 (2026-11-09): round 1 label set

Eight new photos arrived; one (`20261109_112115.jpg`, #77_1) is the same
printout as `IMG_6890.tiff` and was not imported again. The other seven were
registered with the trial method (mean residual 1.3–1.5 overview px each) and
their marks assigned to candidates. The rule-based numeral reader could not
read these photos (small, touching strokes; IV–IX appear), so Claude read
every mark by eye from upright contact sheets; per-stage totals agree with
each sheet's handwritten legend within a few marks. Details, legend check and
scripts: `outputs/paper-alignment-2026-11-09-01/RUN_NOTES.md` (Git-ignored).

Paper labels on the page (one per candidate; "accepted" = unflagged crops,
which the page stages):

| Stage | I | II–III | IV | V | VI | VII | VIII | IX | X | XI | XII | no stage¹ | Total |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Accepted | 623 | 452 | 23 | 49 | 69 | 191 | 76 | 26 | 15 | 173 | 133 | 8 | 1838 |
| All candidates | 698 | 516 | 26 | 51 | 81 | 219 | 92 | 29 | 19 | 182 | 142 | 31 | 2086 |

¹ Marks in one candidate that disagree (usually a v0.5 merge), unreadable
marks, or symbols.

Accepted, by section:

| Section | Stages (count) |
|---|---|
| #76_WR2_50d_6-1 | I 74, II–III 119, IV 13, V 4, VI 1, X 1, XI 1, XII 7 |
| #77_WR2_50d_7-1 (sheet 1, older machine reads) | I 82, II–III 43, XI 68, XII 37 |
| #91_WR2_47d_6-1 | V 2, VI 12, VII 90, VIII 55, IX 25, X 14, XI 8 |
| #92_WR2_47d_7-1 | IV 10, V 43, VI 56, VII 101, VIII 21, IX 1 |
| #108_WR2_43d_4-1 | I 126, II–III 115, XII 1 |
| #109_WR2_43d_5-1 | I 129, II–III 103, XII 1 |
| #112_WR2_42d_2-1 | I 132, II–III 72, XI 1, XII 33 |
| #113_WR2_42d_3-1 | I 80, XI 95, XII 54 |

All 11 classes have examples, but very unevenly: I and II–III have over 450
each, while IV, IX and X have 15–26, and most of those come from one or two
sections. IV–IX come almost entirely from #91 and #92 (one animal age, 47 d),
so a section-held-out split cannot test those stages on an unseen animal yet.
#113's XI/XII split is uncertain: Yeongseok crossed out the legend counts
("82?", "83?") pending a microscope recount.

## Round 1 first result (2026-11-09)

Code: `src/eyespy_spermatogenesis/staging/round1/` (`data`, `embed`,
`evaluate`, `review`). Large files on the 20 TB drive
(`/mnt/20TB_A/Adam/EyeSpySpermatogenesisRebooted/`: `models/hf-cache`,
`staging-round1/{data,embeddings}`); results in
`outputs/staging-round1-2026-11-09-01/` (`summary.md`, `confusion_loso.png`).
Expert review and whole-section overlays live on the annotation page (mode
"Model review", tabs Annotate and Overview; see [ANNOTATION.md](ANNOTATION.md#model-review)).

Frozen H-optimus-0 features (2x downsampled 224 px patches, mean pooled) and a
cyclic ridge head, regularization chosen by nested leave-one-section-out, on
1723 accepted tubules with unflagged paper stages:

| | Leave-one-section-out | Random 5-fold (optimistic) |
|---|---|---|
| Accuracy | 0.80 | 0.85 |
| Balanced accuracy (mean per-stage recall) | 0.70 | 0.78 |
| Within one stage | 0.997 | 0.998 |

Per-stage recall: I and II–III 0.86, VII 0.80, XI 0.85, XII 0.77, IX 0.83,
VIII 0.59, IV/V/VI 0.55–0.57, X 0.50. Errors are almost all to an adjacent
stage. DINOv2-L, the general-purpose baseline, was lower (accuracy 0.78,
balanced 0.67). The cyclic head beat logistic regression in every
configuration and can predict a stage it barely saw in training (IX when #91
is held out) by placing it between its neighbours.

Not yet established: agreement with an expert's eye (review page pending),
label accuracy (transcriptions unchecked), performance on sections without
paper labels. fp16 inference was checked against fp32 on 48 patches (cosine
1.0000, mean relative difference 0.14%).

## Round 2 (SATINN) investigation (2026-11-09)

Source: [SATINN repository](https://github.com/conradlab/SATINN) (Windows app,
MATLAB helpers; no licence file) and the pretrained networks on
[Figshare 22853645](https://figshare.com/articles/dataset/SATINN_-_demo_files_and_neural_networks_r_230516_/22853645)
(**CC BY 4.0**), archive `SATINN_demo_files_260531.rar` (md5
`74be537bde02e2ed3b63caf35987eb48`), stored with the extracted files under
`/mnt/20TB_A/Adam/EyeSpySpermatogenesisRebooted/models/satinn/`. RAR5: extract
with `bsdtar`; this system's 7-Zip and `unar` fail on the compressed entries.

Networks: `bf_tnetr-*` bright-field tubule stager (newest `240110`),
`bf_tss-*` tubule segmentation, `bf_cnetr-*` cell classifier, `if_*`
immunofluorescence. `bf_tnetr-240110` is a MATLAB `DAGNetwork`: **ResNet-50**
(23.6M parameters), input 500x500 RGB 0-255 minus mean colour (194.1, 163.7,
204.2), **12 outputs I, II, III, ..., XII** (II and III separate; summed for
our II-III class). Paper: 8,285 tubules from 42 images at 0.137-0.17 µm/px,
63% exact / 96% within one stage on external validation, predictions below
0.80 confidence discarded.

Converted without MATLAB (`staging/satinn.py`, `mat-io` decodes the MATLAB
objects): `convert` writes plain arrays (`bf_tnetr-240110.npz`), `load`
rebuilds it in PyTorch. The rebuild is unit-tested against a NumPy reference.

**Open: the input preprocessing.** The paper and code do not state it for
bright-field. Checking each batch-norm layer's stored training statistics on
SATINN's own demo image (`MS124RS101_med`, about 0.15-0.18 µm/px) shows the
training images had about **3x more contrast** than the demo file (raw input:
first-layer variance 10% of training; with a 3x contrast gain about the mean,
0.8-1.3x at most layers). Scale is less clear (500 px native or 2000 px
reduced 4x both partly match). Predictions change strongly with these
choices, so SATINN must not be scored on our labels until the preprocessing
is pinned down (ask the authors, or run their Windows app as a reference, or
choose it without our labels by matching batch-norm statistics on our crops).

### Round 2 first result (2026-11-09)

`staging/round2.py`; results in `outputs/staging-round2-2026-11-09-01/`
(`summary.md` with the full sensitivity table, `confusion_loso.png`,
`probs.npz` for round 4); Overview panels on the annotation page.
Preprocessing chosen **without labels** (lowest batch-norm mismatch on 300
tubules over 7 box sizes x 6 contrast settings): 500 px native box around the
tubule centre, mean colour moved to SATINN's, no contrast gain.

On the same 1723 tubules as round 1: **30.6% exact, 30.8% balanced, 51.5%
within one stage** (confidence >= 0.80: 72% coverage, 35.6% exact). No setting
in the grid exceeded 35% exact. Errors are systematic rather than random:
most stage I (374 of 589) and XII (90 of 128) tubules are called VI;
II-III splits between II-III, V and VI; VI-XI are mostly right (VII 98%,
IX 62%, XI 60%). Median confidence 0.96, so its 0.80 filter barely helps:
a domain shift (staining, scanner, fixation or undocumented preprocessing),
not random noise. The planned Windows-app run on the demo image and a few of
our crops will show whether our conversion/preprocessing or the domain is to
blame.

### Round 2 preprocessing checks (2026-11-09)

`outputs/staging-round2-2026-11-09-01/checks.{py,md,json}` and
`checks_small.{py,json}`. Both SATINN stagers x box size (250-1600 px) x
outside-tubule fill (none / white / black, using our v0.5 outlines) x contrast.

- SATINN's segmentation net (`bf_tss`, DeepLab-style, 720x960 input) labels
  tubule / interstitium / background / boundary, so their pipeline has exact
  tubule outlines; masking is plausible.
- **Black masking helps** `bf_tnetr-240110`, and the label-free criterion
  agrees: lowest batch-norm mismatch of all settings (0.48) is box 500 px,
  black fill, raw contrast: **41% exact, 72% within one** (vs 31% / 52%
  unmasked). Best by labels (not used to choose): box 400 px, black, 43% /
  78%.
- Accuracy is very sensitive to scale: 400-500 px boxes 41-43%, 650 px
  11-16%, 250 px 29%. The best scale is ~2.5x more zoomed-in than SATINN's
  own helper implies (2000 px at ~0.15 um/px reduced 4x), if our pixels are
  ~0.25 um.
- The older `bf_tnetr-230314` (10 classes, X-XI merged, darker training
  mean) collapses on our data: about 34% exact with 10% balanced accuracy,
  i.e. nearly everything called one stage.

Still far below the published 63% / 96%. Whether the gap is our
reconstruction of their pipeline or a real domain shift (staining, scanner,
fixation, staging convention) is for the Windows-app reference run.

## Round 3 (UNI2-h) result (2026-11-11)

> ## ⚠ RESEARCH ONLY: NOT FOR RELEASE
>
> UNI2-h is licensed CC-BY-NC-ND-4.0 under the Mahmood lab's gated terms
> (model card, read 2026-11-11): non-commercial academic research only; no
> copy of the model may be distributed, published or reproduced; its
> "derivatives", **including models trained on its outputs and datasets
> created from it**, may not be used commercially; every user must register
> on Hugging Face individually. **No model in this section can ship in a public
> or packaged build of this software** (decision 025). If it proves useful, it
> is a proof of concept showing the lab what more annotations could achieve.
> The repo holds only the loader settings (`round1/embed.py`, `RESEARCH_ONLY`);
> weights (`models/hf-cache`), embeddings (`staging-round1/embeddings/uni2-h`)
> and trained heads stay on the 20 TB drive, outside Git.

Same data, split and classifiers as round 1 (1723 tubules, leave-one-section-
out, nested regularization), with UNI2-h (ViT-H/14, 681M parameters, 1536-dim,
ImageNet normalisation, settings from the model card) as the frozen extractor.
fp16 matched fp32 on 48 patches (cosine 1.0000, mean relative difference
0.10%). Embedding took 30 min on the RTX 4060. Results in
`outputs/staging-round3-2026-11-11-01/` (`summary.md`, `results.json`,
`confusion_loso.png`, `compare.py/json`, `ensemble.py/json/log`,
`ensemble_predictions.npz`).

Ensembles average the two heads' cyclic (cos, sin) outputs, each with its own
nested-chosen regularization. 95% intervals are a section bootstrap (2000
resamples) of the difference from H-optimus-0 alone:

| Model | Exact | Balanced | Within one | Δ exact vs H-optimus-0 (95% CI) | Releasable |
|---|---|---|---|---|---|
| H-optimus-0 (round 1) | 0.804 | 0.704 | 0.997 | — | yes |
| DINOv2-L | 0.781 | 0.663 | 0.993 | −0.047 to +0.002 | yes |
| **UNI2-h** ⚠ | 0.811 | 0.721 | 0.995 | −0.036 to +0.041 | **no** |
| UNI2-h + H-optimus-0 ⚠ | **0.838** | **0.746** | 0.996 | **+0.013 to +0.054** | **no** |
| UNI2-h + DINOv2-L ⚠ | 0.817 | 0.691 | 0.997 | −0.009 to +0.034 | no |
| H-optimus-0 + DINOv2-L | 0.823 | 0.727 | 0.996 | +0.004 to +0.034 | yes |
| all three ⚠ | 0.821 | 0.713 | 0.997 | +0.001 to +0.034 | no |

Per-stage recall:

| Stage | n | H-optimus-0 | UNI2-h ⚠ | UNI2 + H-opt ⚠ | H-opt + DINOv2 |
|---|---|---|---|---|---|
| I | 589 | 0.86 | 0.83 | 0.88 | 0.86 |
| II-III | 421 | 0.86 | 0.88 | 0.90 | 0.88 |
| IV | 23 | 0.57 | 0.39 | 0.57 | 0.52 |
| V | 47 | 0.57 | 0.68 | 0.64 | 0.60 |
| VI | 67 | 0.55 | 0.73 | 0.64 | 0.61 |
| VII | 188 | 0.80 | 0.75 | 0.81 | 0.81 |
| VIII | 75 | 0.59 | 0.71 | 0.69 | 0.73 |
| IX | 24 | 0.83 | 0.75 | 0.83 | 0.79 |
| X | 14 | 0.50 | 0.57 | 0.57 | 0.57 |
| XI | 147 | 0.85 | 0.87 | 0.89 | 0.84 |
| XII | 128 | 0.77 | 0.76 | 0.79 | 0.78 |

Findings:

- **UNI2 alone is not measurably better than H-optimus-0** (+0.7 points exact;
  interval spans zero). It is better on V, VI and VIII and worse on IV; the
  two get different tubules right (144 only UNI2, 133 only H-optimus-0; the
  models agree on 83%). Its best view is native-scale patches (`s1`), H-optimus-0's
  2x downsampled (`s05`).
- **The best model is the UNI2-h + H-optimus-0 ensemble**: 83.8% exact, the
  only gain over round 1 whose interval clears zero by more than a point. It
  beats the best releasable ensemble (H-optimus-0 + DINOv2-L, 82.3%) by
  1.5 points (95% CI +0.5 to +2.3). Balanced-accuracy differences are all
  within noise (IV, IX, X have 14-24 tubules, mostly from one section).
- Concatenating UNI2 and H-optimus features into one classifier did not help
  (best 0.800 exact); averaging two separately trained heads did.
- UNI2 did not overcome the label shortage: IV, V, VI and X stay at
  0.39-0.73 recall in every model. More labels for those stages remain the
  main lever.
- As in round 1, the configuration (view, pooling) was picked on the same
  leave-one-section-out screen, so all figures are slightly optimistic, equally
  for every model.

Annotation page: review run `staging-round3-2026-11-11-01` (blind A/B:
UNI2-h vs H-optimus-0) holds **the same 164 tubules as the round 1 run**, so
one pass of answers scores both runs (answers are stored per tubule). The
Overview shows UNI2-h, H-optimus-0 and both ensembles. Ensembles are not in
the A/B review; score them offline against the reviewer's answers from
`ensemble_predictions.npz`.

## Opt-in experiments (2026-11-11)

Owner request: try settings that might raise accuracy, every one reversible
(decision 026). Each is a switch or a separate module; with every switch off,
`staging/experiments.py` reproduces the round 1 and round 3 numbers exactly.
Same 1723 tubules, leave-one-section-out; every choice (regularization,
ensemble weights, smoothing k and strength) is made per held-out section from
its training sections only. Intervals are a section bootstrap (2000
resamples) of the difference from plain H-optimus-0 (0.804 exact).
Results in `outputs/staging-experiments-2026-11-11-01/` (one folder per
comparison, each with `summary.md` incl. per-stage recall).

| Switch | Where | Default |
|---|---|---|
| Rotation/flip averaging (8 copies per patch, mean embedding) | `round1/embed.py --tta d4` → `embeddings/<model>_d4/` | off |
| Class weighting (1/stage count) | `experiments.py --balanced on` | off |
| Tuned ensemble weights | `experiments.py --weights nested` | equal |
| Neighbour smoothing (blend with nearest tubules' predictions, same section; never their labels) | `experiments.py --smooth nested` | **off; never the default** (decision 026) |
| SATINN fine-tuning / AdaBN | `staging/satinn_finetune.py` | separate module |

### Results to show

Headline (unsmoothed) rows first; smoothed rows are opt-in and labelled.
⚠ = uses UNI2-h, research only, never releasable (decision 025).

| Model | Smoothed | Exact | Balanced | Within one | Δ exact (95% CI) | Releasable |
|---|---|---|---|---|---|---|
| H-optimus-0 (round 1) | no | 0.804 | 0.704 | 0.997 | — | yes |
| H-optimus-0, rotation/flip averaged | no | 0.818 | 0.691 | 0.996 | +0.001 to +0.028 | yes |
| H-optimus-0 + DINOv2-L (round 3) | no | 0.823 | 0.727 | 0.996 | +0.004 to +0.034 | yes |
| **H-optimus-0 + H-optimus-0 rot. + DINOv2-L rot.** | no | **0.831** | **0.732** | 0.997 | **+0.008 to +0.044** | **yes** |
| UNI2-h + H-optimus-0 (round 3) ⚠ | no | 0.838 | 0.746 | 0.996 | +0.013 to +0.054 | no |
| SATINN, pretrained (round 2 setting) | no | 0.411 | 0.387 | 0.717 | — | yes (CC BY 4.0) |
| SATINN, fine-tuned on our labels | no | 0.750 | 0.489 | 0.973 | — | yes (CC BY 4.0) |
| H-optimus-0 | **yes** | 0.822 | 0.706 | 0.995 | +0.011 to +0.025 | yes |
| H-optimus-0 + H-opt. rot. + DINOv2-L rot. | **yes** | 0.837 | 0.735 | 0.997 | +0.020 to +0.045 | yes |
| UNI2-h + H-optimus-0 ⚠ | **yes** | 0.838 | 0.737 | 0.998 | +0.018 to +0.050 | no |

Findings:

- **Best releasable model: the three-part ensemble** (plain H-optimus-0,
  rotation-averaged H-optimus-0, rotation-averaged DINOv2-L; equal weights):
  83.1% exact, 73.2% balanced, within 0.7 points of the research-only UNI2
  ensemble. Its gain over the round 3 releasable pair (+0.8 points) was not
  tested directly and is probably within noise.
- **Rotation/flip averaging** helps H-optimus-0 (+1.4 points) and DINOv2-L
  (+0.4) a little and does **not** help UNI2-h (0.798 vs 0.811). Averaging
  both members of the releasable pair alone gives 0.825, no better than the
  plain pair; the gain comes from adding the rotated copies as extra members.
- **Smoothing** gives +1.8 points on a single model (interval clears zero)
  but little or nothing on the ensembles (UNI2 ensemble unchanged at 0.838).
  It costs isolated stages: IV recall falls on most variants, and balanced
  accuracy does not improve. Keep it opt-in (decision 026).
- **Class weighting** trades exact for balanced accuracy (e.g. UNI2 ensemble
  0.838 → 0.799 exact, 0.746 → 0.749 balanced); no net gain. Not
  recommended.
- **Tuned ensemble weights** do worse than 50/50 (0.832 vs 0.838): with eight
  sections the nested choice is noisy. Keep equal weights.
- **SATINN fine-tuning** (Linux, PyTorch conversion; ResNet-50, 8 epochs,
  AdamW lr 1e-4, rotation/flip augmentation, batch-norm statistics frozen
  after AdaBN, 500 px black-masked boxes) lifts SATINN from 41% to 75% exact
  but stays below every frozen-feature model and loses the rare stages
  (recall IV 0.04, IX 0.12, X 0.00). AdaBN alone (statistics re-estimated
  without labels): 57% exact. Per section 0.50-0.83; #92 is weakest (0.50).
  No Windows PC was needed. Its probabilities (`satinn-finetune/
  predictions.npz`) are a candidate input for round 4.
- Implementation note: AdaBN must estimate batch-norm layers one at a time in
  network order, in fp32; estimating all layers in one pass, or in fp16,
  diverges on this network (found and fixed 2026-11-11; the guard now stops
  any run with non-finite outputs). Round 2's published numbers do not use
  AdaBN and are unaffected.
- The stages short of labels (IV, V, VI, IX, X) remain the limit for every
  method; more labels for them are still the main lever.
