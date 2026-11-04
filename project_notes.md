# 📄 project_notes.md — EyeSpy Spermatogenesis (Authoritative Codex Context)

IMPORTANT FOR CODEX / AI ASSISTANTS  
This document defines the current ground truth, scope, constraints, and phased objectives of the EyeSpy Spermatogenesis project.  
Any code changes, refactors, or new features MUST respect this document unless explicitly instructed otherwise by the user.

---

## 1. Project Overview (High-Level Goal)

EyeSpy Spermatogenesis is an offline, Python-based AI system for analyzing histological testis images.

The ultimate end goal (Phase 3+) is:
1. Detect individual seminiferous tubules in whole-slide or field images  
2. Separate each tubule  
3. Subdivide tubules into epithelial “pizza slices”  
4. Assign spermatogenic stage classifications (mouse → later human)

However, we are NOT doing all of this yet.

---

## 2. Current Phase (Phase 1 — Tubule Detection Only)

### Current Objective
Train a model that can identify seminiferous tubules, specifically clean circular cross-section tubules, in testis images.

This is a binary segmentation task, not classification.

- Positive class (1): cross-section seminiferous tubules  
- Negative class (0): everything else  

The model should learn:  
“Where are the circular cross-section tubules in this image?”

---

## 3. Explicit Non-Goals (DO NOT IMPLEMENT YET)

The following are explicitly out of scope for now:
- Stage classification (I–XII)  
- Spermatogonia / spermatocyte / spermatid labeling  
- Pizza-slice segmentation  
- Instance-level tubule separation  
- Human tissue generalization  
- Multi-class segmentation  

Any attempt to reintroduce these must be rejected unless the user explicitly asks.

---

## 4. Data & Annotation Rules (Critical)

### 4.1 Annotation Format
- Tubules are annotated using polygon segmentation  
- Each polygon represents one tubule  
- Polygons are rasterized into binary masks during training  

### 4.2 Tubule Orientation Tags
Each annotated tubule has an orientation field stored in JSON metadata.  
Allowed values are:
- cross_section  
- longitudinal  
- oblique  

Each tubule polygon also has a `partial` flag:
- `partial=false` (default): full cross-section tubule
- `partial=true`: truncated/cropped cross-section (any visible cut-off; no % threshold)

### 4.3 Training Rules (Very Important)

The model is trained using **explicit positives and explicit negatives**.

#### Positive Tubules (label = 1)
A tubule is considered a POSITIVE example if and only if:
- orientation == "cross_section"
- partial == false OR partial == true (all cross-sections)

These represent all cross-section tubules (including partial/cropped).

#### Negative Tubules (label = 0)
A tubule is considered a NEGATIVE example if:
- orientation == "longitudinal", OR
- orientation == "oblique"

Negative tubules are **explicitly supervised as incorrect tubules**.  
They are NOT treated as background and MUST contribute to training loss.

This is critical to prevent the model from:
- confusing tubule lumens with background
- overgeneralizing tubule-like textures
- learning that “any tubule structure is acceptable”

#### Background
- Pixels not covered by any tubule polygon are treated as true background (label = 0).
- Background-only images are allowed in principle, but the current training run policy **excludes empty annotations**.

#### Ignore Regions
- An optional ignore-band may be applied around positive tubules to reduce boundary ambiguity.
- Ignore regions are excluded from loss computation.
- Negative tubules must NEVER be ignored.

#### Annotation Philosophy
Annotators should:
- Always annotate clean circular cross-section tubules (positives).
- Optionally annotate obvious incorrect tubules (negatives) when helpful.
- Use the `partial` flag to mark cropped or incomplete cross-sections.
- Not attempt exhaustive annotation of all incorrect tubules.

This keeps annotation scalable while still providing strong negative supervision.

Current experiment note (2026-02-02):
- Treat all cross-section tubules as positives, including `partial=true` (partial no longer negative).

## 5. Model Architecture (Current)

- Task: binary semantic segmentation  
- Output: probability map of shape [H, W] indicating tubule likelihood  
- Backbone: CNN / U-Net–style architecture (exact choice flexible)  
- Loss:
  - Dice + BCE (or equivalent)  
  - No explicit negative-image loss weighting  

Training configuration assumptions:
- --include-yes-only  
- --no-require-positive-annotations  
- --require-any-annotations  
- --neg-tubule-weight 2.0  
- --bg-weight 1.0  
- --balanced-loss  
- --ignore-band-px 32 (ignore ambiguous image borders)  
- --area-penalty 0  

Current dataset inclusion policy:
- Include images with `include=yes` that have **at least one tubule polygon** (positive or negative).
- Do **not** require a positive tubule in each image.
- The image-level `negative_training` flag is only relevant if we ever want to include **empty annotations**.

Model export / archives:
- Best model is exported to `models/exported/seg_model.pt`.
- Each completed training run is archived to `models/exported/past_models/` with UTC timestamp (no auto-deletion).

---

## 6. Evaluation Philosophy

### 6.1 Metrics (Secondary)
Dice, IoU, precision, and recall are useful diagnostics but are not decisive at this stage.

Additional diagnostic:
- Negative-tubule false-positive rate (fraction of neg tubule pixels predicted positive).

### 6.2 Visual Evaluation (Primary)
Visual inspection of segmentation overlays is the primary success criterion.

Overlay colors (current eval):
- GT positives: green (filled mask)
- GT negative tubules: blue (filled mask)
- Predictions: red (filled mask)

Desired qualitative behavior:
- Localization around circular tubules  
- Avoidance of flooding the entire image  
- Partial detection is acceptable  
- Overprediction is acceptable early  
- Shape bias toward circles is a good sign  

---

## 7. Output Behavior Expectations

At this phase, it is expected that:
- The model may still output large blobs  
- The model may miss some tubules  
- The model may detect tubule borders before interiors  

These behaviors are normal early-stage behaviors, not failures.

---

## 8. Future Phases (DO NOT IMPLEMENT NOW)

### Phase 2 — Tubule Isolation
- Convert segmentation blobs into individual tubule instances  
- Possible watershed or contour-based separation  

### Phase 3 — Intra-Tubule Analysis
- Radial slicing of tubules  
- Stage classification  
- Human tissue support  

These phases depend on Phase 1 being solid first.

---

## 9. Codex Instructions (Mandatory)

When assisting on this project, Codex MUST:
1. Assume offline execution  
2. Avoid introducing unnecessary complexity  
3. Preserve annotation simplicity  
4. Favor clarity over performance tricks  
5. Never reintroduce stage classification unless explicitly asked  
6. Log dataset filtering decisions clearly  
7. Make training runs self-verifying (counts, filters, orientation stats)  

If something is ambiguous, ask the user instead of guessing.

---

## 10. One-Sentence Summary

This project is currently training a binary segmentation model to detect circular cross-section seminiferous tubules only, using polygon annotations, ignoring all other tubule orientations, as the foundation for future spermatogenesis analysis.
