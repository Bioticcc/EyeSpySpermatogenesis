# Prototype implementation and planned comparisons

Owner update (2026-11-06): D2 overview localization then native crop retrieval
is the initial whole-section route, superseding the 2026-11-04 D1 choice. The
owner authorized implementation before the test slice arrives. A restricted I/O
and classical-candidate prototype now exists; see
[OVERVIEW_PROTOTYPE.md](OVERVIEW_PROTOTYPE.md) for its concrete plan, dependencies,
commands, masks' provisional status and remaining validation. This is not a
scientifically selected segmentation method or authorization for staging.

The comparisons below still require real pilot data and agreed evaluation
criteria. No overview factor, PAS threshold, physical cutoff or winner is validated.

## Tubule-isolation candidates

| ID | Hypothesis, required inputs and proposed dependencies | Outputs and evaluation | Advantages, limitations and decision rule |
|---|---|---|---|
| A — deterministic | PAS boundary/interior evidence plus morphology can recover useful instances without training. Needs verified native-detail regions, reviewed color/scale, and tuning-only expert masks. Candidate dependencies: NumPy, scikit-image, OpenCV; add only needed components. | Common crops/masks/quality records; instance precision/recall, complete usable crops, touching/border errors, artifacts, RAM/time and annotation/tuning effort. | Small offline stack and inspectable rules; stain/pathology variation may break thresholds or seeds. Select if held-out complete-crop utility and failure handling meet agreed targets. |
| B — mature software/pixel classifier | Small representative pixel annotations plus established programs can improve robustness. Compare QuPath classification → instances; ilastik probabilities → Fiji/Python morphology; and Fiji macro-only segmentation. Needs native regions, training annotations, saved project/class order and external installations. | Same output schema; include probability maps and exact application/project versions. Same A metrics plus expert time and headless/batch reliability. | Interactive refinement and existing tools; bridges, classifier domain shift, runtime size and external-app integration remain concerns. Prefer over A only for measured benefit worth the operational cost. |
| C — legacy benchmark | Frozen legacy predictions offer a useful comparator if reconstruction is fair and inexpensive. Needs original checkpoint/config, compatible isolated environment, verified preprocessing and training-image identities. PyTorch stack stays outside core. | Identical output adapter and held-out instances; report semantic masks and crop utility separately. | Historical reference; prior training exposure, missing grouped split provenance or reconstruction cost may prevent a fair benchmark. Do not retrain/improve it during setup; omit with a documented reason if unsafe/unfair. |

A explores RGB/HSV/Lab/intensity and boundary gradients without assuming that PAS
isolates one universal color. Compare tissue/background exclusion, lumen/boundary
cues, fixed/adaptive thresholds, opening/closing, hole filling, distance transforms,
seeded watershed, connected components and contours. These are proposed experiments,
not a mandated sequence. Missing lumens and noncircular sections must remain visible.

Measure calibrated area when available, circularity, eccentricity, solidity,
boundary completeness, lumen geometry, border contact, neighboring-tissue overlap
and section orientation. Keep raw measurements and review reasons. A physical
filter cannot run on invented calibration; pixel-only experiments must be labeled
as such. No circularity threshold or maximum instance count transfers from legacy.

B trains pixel labels on training groups only, with held-out stain variation.
Compare annotation burden explicitly. GUI training is compatible with an eventual
headless application adapter; a macro-recordable plugin is not automatically safe
in a fully headless JVM. Test that behavior before promising unattended operation.

## Whole-section strategies, orthogonal to A/B/C

Run a practical cross-comparison, keeping reader/localization strategy separate
from segmentation method. Start with provisional A-D2, compare B-D2/D3, and
retain D1 as a fallback comparison over the same tissue coverage.

| ID | Hypothesis / inputs / dependencies | Expected outputs and comparison | Advantages, risks and decision rule |
|---|---|---|---|
| D1 — direct native-detail tiling | Overlapping 40×-equivalent regions provide enough context. Requires verified native-detail source, reader, tissue coverage, configurable tile/overlap/halo and bounded resources. | Reconciled whole-section instances and coverage map; complete recovery, misses, duplicates, seams, time and peak RAM. | Direct detail and simpler level mapping; repeated decoding, edge truncation and reconciliation cost. Fallback comparison; tiles must accommodate the full object plus context. |
| D2 — coarse localization then retrieval | Lower-resolution candidates reduce expensive high-resolution work without materially reducing usable-crop recall. Requires a suitable pyramid level or documented derived overview, calibrated transforms and genuine native-detail retrieval. | Coarse proposals plus corresponding native crops/masks and lineage; end-to-end recall including localization misses, complete crops, duplicates, transform error, time/RAM and preprocessing cost. | Potential efficiency; weak boundaries or small/abnormal tubules may vanish. Owner-selected initial prototype; retain only if measured recall/resources meet agreed limits. Never upscale coarse pixels as final detail. |
| D3 — whole-slide application | QuPath's reader/classifier/ROI tools reduce custom integration work. Needs an opening-tested source, frozen QuPath project/classifier and export script. | Source-space ROIs, native crops, masks, quality and coverage; same D1 metrics plus export fidelity and unattended reliability. | Existing slide context; external runtime, project conventions, coordinate/export checks and licensing work. Select if practical integration is more reliable than library orchestration. |

Folder images bypass whole-slide localization but use the same isolation/output
contract. Existing image borders remain genuine truncation boundaries. Test the
same folder images standalone and as an equivalent reader ROI where possible;
do not claim whole-slide coordinates for standalone frames.

## Independent staging candidates

| ID | Hypothesis / inputs / dependencies | Outputs / evaluation | Advantages, limitations and selection |
|---|---|---|---|
| S1 — existing pretrained | An existing mouse bright-field model can transfer. Needs suitable isolated crops, compatible documented preprocessing and authorized/checksummed weights. Investigate STSP and SATINN v2 in separate environments. | Exact/grouped stage probabilities where supported, eligibility and abstention; grouped holdout accuracy, calibration, coverage, failure cases and resource cost. | Reuse existing expertise; stain, fixation, scale, crop context and platform mismatch can invalidate transfer. Select only after local validation; fine-tuning is a separate approval. |
| S2 — interpretable morphology | Spatial features plus a compact statistical classifier provide useful staging or grouping. Needs expert labels, boundaries and reliable cell/lumen cues; candidate scikit-image/OpenCV plus a selected statistical package. | Feature table with units/missingness, classifier/taxonomy IDs and uncertainty. Same holdout; compare feature ablations and expert interpretability. | Inspectable radial/texture/orientation evidence; feature extraction may be difficult and exact stages inseparable. Select on measured performance/effort; never promise deterministic twelve-stage rules. |
| S3 — hybrid | S1 and explicit features have complementary errors. Needs paired S1/S2 outputs and sufficient training/validation groups. | Combined probabilities, component provenance and abstention; held-out improvement at matched coverage, calibration and added cost. | Potential complementary evidence; complexity and overfitting. Attempt only after component baselines exist; discard if benefit is unconvincing. |

See [STAGING_STRATEGY.md](STAGING_STRATEGY.md) for taxonomy and suitability gates.

## Selection procedure

Before experiments, agree primary metrics, minimum useful coverage, maximum false
acceptance/duplicate rates, expert effort, runtime/RAM budget and acceptable
confidence intervals. Values are PENDING CONFIRMATION. Freeze methods on training
and validation groups, score the untouched common test set once, then review
Pareto tradeoffs with the owner. Favor the simplest maintainable integration that
meets the scientific requirements. A negative result is a valid prototype outcome.
