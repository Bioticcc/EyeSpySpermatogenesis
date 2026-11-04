# Legacy findings

Review date: 2026-11-04. Evidence is source inspection, not a rerun or independent
scientific validation. The owner's account of unsatisfactory segmentation and
training-data burden motivates the reset; logs alone do not prove its causes.

## Reviewed revisions

| Repository | Reviewed head |
|---|---|
| [EyeSpySpermatogenesis](https://github.com/Bioticcc/EyeSpySpermatogenesis/tree/4af472c53beb3810828923ddd252f451374d2a01) | `4af472c53beb3810828923ddd252f451374d2a01` |
| [SCP-SpermColorPixels](https://github.com/Bioticcc/SCP-SpermColorPixels/tree/dbaba00451c9a0a74114a02ca599c182f9c71e50) | `dbaba00451c9a0a74114a02ca599c182f9c71e50` |
| [EyeSpySC](https://github.com/Bioticcc/EyeSpySC/tree/46c836f71acf2bf09b57404d61b4a124b17bd5a9) | `46c836f71acf2bf09b57404d61b4a124b17bd5a9` |

The neighboring EyeSpySC checkout has uncommitted work and was not used as a
frozen baseline or modified. This review used the published revision above.

## Verified observations: original EyeSpySpermatogenesis

- The [README](https://github.com/Bioticcc/EyeSpySpermatogenesis/blob/4af472c53beb3810828923ddd252f451374d2a01/README.md)
  centers polygon-supervised binary segmentation, with older box detection and
  classification retained. Its latest banner says development has resumed. The
  supplied research note's claim that the current README says cancelled is stale.
- [The model](https://github.com/Bioticcc/EyeSpySpermatogenesis/blob/4af472c53beb3810828923ddd252f451374d2a01/src/training/seg_model.py)
  implements `UNetSmall`, with convolution/BatchNorm/ReLU blocks, pooling,
  transposed-convolution upsampling and one-channel logits. This is verified
  implementation, not just a README architectural label.
- [Historical instructions](https://github.com/Bioticcc/EyeSpySpermatogenesis/blob/4af472c53beb3810828923ddd252f451374d2a01/project_notes.md)
  explicitly designate the CNN segmentation approach authoritative and postpone
  staging/instance separation. They also contain inconsistent summaries of
  negative/ignored orientations. They are historical evidence, superseded by the
  reset request and new `AGENTS.md`.
- [The training log](https://github.com/Bioticcc/EyeSpySpermatogenesis/blob/4af472c53beb3810828923ddd252f451374d2a01/TRAINING_EVAL_LOG.md)
  records training recipes, hard negatives, ignore bands and visually selected
  thresholds (including a 0.4–0.9 sweep). Several metric/result fields are empty.
  It contains no completed quantitative crop-recovery benchmark.
- Annotations preserve original pixel polygons, orientation, partial status and
  image-level inclusion/quality. An inspected
  [empty annotation](https://github.com/Bioticcc/EyeSpySpermatogenesis/blob/4af472c53beb3810828923ddd252f451374d2a01/data/labels/annotations/20_66.json)
  is excluded. A separately inspected
  [annotated example](https://github.com/Bioticcc/EyeSpySpermatogenesis/blob/4af472c53beb3810828923ddd252f451374d2a01/data/labels/annotations/Stage%20IX%20Oct%2014%20Kyle%20approved.json)
  has one 28-vertex longitudinal polygon, `partial=false`, and unknown quality.
  Its filename is not a verified stage label or biological record.
- [Segmentation training](https://github.com/Bioticcc/EyeSpySpermatogenesis/blob/4af472c53beb3810828923ddd252f451374d2a01/src/training/seg_train.py)
  archives model exports and uses record-level shuffled `train_test_split`.
  Animal/section grouping is not enforced there; actual leakage is unverified
  without source identities and saved training manifests.
- [TIFF I/O](https://github.com/Bioticcc/EyeSpySpermatogenesis/blob/4af472c53beb3810828923ddd252f451374d2a01/src/utils/tiff_io.py)
  reads a selected page eagerly, converts to float and can repeat/drop channels.
  It is not a verified large-OME region reader or physical-scale contract.
- The tracked tree contains annotations, raw images, checkpoint/export weights,
  overlays, helper scripts, tests and a PyInstaller spec. Presence of artifacts
  establishes preservation, not their correctness or reproducibility. No weights
  were loaded, training rerun, or biological overlays rescored during this review.

## Interpretation and unresolved performance

The project built useful annotation, training, export and visual-review machinery.
Those are engineering assets, but source inspection does not establish accurate
segmentation, usable individual crops, or staging. Training-data burden and
unsatisfactory results are owner-reported; the available log does not isolate
whether annotation semantics, domain variation, model choice or other causes
dominated. Do not generalize this into evidence that all deep learning fails.

Potential future reuse: annotation readers, explicit include/quality/orientation
flags, synthetic test ideas, command/config logs, immutable model archives and
visual review patterns. Reuse requires auditing semantics and licensing, then
adapting to the new contracts. Do not import old modules, scale assumptions,
top-K instance limits, preprocessing or assistant instructions wholesale.

Benchmark C is optional: freeze checkpoint/config/environment and reconstruct
preprocessing without training. Recover training exposure and animal IDs; otherwise
report a descriptive comparison only or omit it as unfair. Budget no substantial
model revival during Phase 0.

## Verified observations: SCP and EyeSpySC

SCP's
[pipeline](https://github.com/Bioticcc/SCP-SpermColorPixels/blob/dbaba00451c9a0a74114a02ca599c182f9c71e50/Algorithmic_Pixel_Mask_For_Tails/algorithmic_tail_mask.py)
provides color/local-darkness candidate masks, connected components, head/tail
association, geometric/path scores, reuse arbitration and separate accepted,
rejected and ambiguous artifacts. Functions inspected include
`build_candidate_masks`, `score_crop_candidate` and `arbitrate_crop_candidates`.
Its [README](https://github.com/Bioticcc/SCP-SpermColorPixels/blob/dbaba00451c9a0a74114a02ca599c182f9c71e50/README.md)
explicitly describes unresolved overlaps and experimental regressions; it is not
evidence that deterministic instance extraction is already solved for histology.

EyeSpySC's
[pipeline](https://github.com/Bioticcc/EyeSpySC/blob/46c836f71acf2bf09b57404d61b4a124b17bd5a9/backend/src/eyespysc/sperm_count.py)
has explicit settings and stages: background flattening, ridge/local-darkness
masks, connected structures, shape/structural checks and reproducible output
planning. `plan_outputs` checks output-path collisions before processing.
Its [README](https://github.com/Bioticcc/EyeSpySC/blob/46c836f71acf2bf09b57404d61b4a124b17bd5a9/README.md)
limits its stated acquisition scope to 20× grayscale phase contrast, calls counts
provisional and describes remaining validation errors. Its pixel settings and
frame-size scaling do not provide physical calibration for this project.

## Lessons retained as new design proposals

Retain explicit stages, inspectable candidate measurements, source-space identity,
configuration provenance, ambiguity categories, collision-safe exports, repeatable
runs and honest validation limits. Reuse mature operations before writing bespoke
algorithms. These are engineering principles, not transferable sperm-color gates,
tail morphology or 20× phase-contrast thresholds.

The new project begins with independently selectable isolation and staging
adapters, two input pathways and common instance/crop evaluation. A frozen U-Net
may compete, but its presence in history gives it no preferred status.
