# EyeSpySpermatogenesis

# NOTE: Project now in developement once more! WIll try integrating the EyeSpySC and SCP algorithms into this, and see if that improves things.

## Quick Commands
1. Open .venv: `source .venv/bin/activate` (Windows PowerShell: `.\.venv\Scripts\Activate.ps1`)
2. Open annotation tool: `python -m src.dataset.annotation_tool`
3. Default training command:
   `python -m src.training.seg_train --epochs 30 --batch-size 1 --device cpu --include-yes-only --require-nonempty-annotations --balanced-loss --area-penalty 1`
4. Default eval command:
   `python -m src.training.seg_eval --weights models/exported/seg_model.pt --save-viz`
5. Optional flags to tweak training/eval:
   Training: `--max-size`, `--val-ratio`, `--allowed-orientations`, `--only-high-quality`,
   `--use-negative-training`, `--negative-sample-weight`, `--area-penalty`,
   `--neg-tubule-weight`, `--bg-weight`, `--require-positive-annotations`, `--require-any-annotations`
   Eval: `--threshold`, `--save-viz`, `--postprocess`, `--pp-min-area-frac`, `--pp-max-components`,
   `--pp-fill-holes`, `--pp-max-hole-frac`, `--include-yes-only`, `--require-nonempty-annotations`,
   `--use-negative-training`, `--only-high-quality`

Offline Windows desktop app + training pipeline for Phase 1: seminiferous tubule segmentation in TIFF images. Phase 2 stage classification remains in the repo as legacy/future work. Legacy box-based detection tooling is still available.

## Quickstart (Phase 1: Segmentation)
- Create env: `scripts/make_env.sh` (Linux/macOS) or `scripts/make_env.ps1` (Windows). Uses Poetry; Python 3.10+.
- Index data: place `.tif/.tiff` files under `data/raw/`, then `python -m src.dataset.indexing`.
- Annotate polygons: `python -m src.dataset.annotation_tool` (draw one closed polygon per tubule).
- Train segmenter: `python -m src.training.seg_train --epochs 5 --batch-size 2 --device cpu`.
- Evaluate: `python -m src.training.seg_eval --weights models/exported/seg_model.pt --save-viz`.
- Inference CLI: `python -m src.app.inference --task seg --weights models/exported/seg_model.pt data/raw/example.tif --output-mask-dir outputs/masks`.
- GUI: `python -m src.app.gui`.
- Build app (Windows): `scripts/build_app.ps1` (one-folder PyInstaller bundle).

## Repository Layout
- `data/raw/` raw TIFFs (tracked for moving the project between machines)
- `data/labels/labels.csv` image-level metadata (tracked)
- `data/labels/annotations/` per-image polygon annotations (tracked)
- `data/splits/` phase-2 train/val/test CSVs (ignored)
- `models/exported/seg_model.pt` exported segmentation weights (tracked)
- `models/exported/detector.pt` legacy detector weights, if present
- `models/exported/model.pt` phase-2 classifier weights, if present
- `src/` Python modules (dataset, training, app)
- `scripts/` helper scripts
- `tests/` lightweight pytest checks (synthetic TIFFs)

## Labels and Annotations
`data/labels/labels.csv` columns:
`image_id, rel_path, quality, include, negative_training, notes, labeled_by, labeled_at`

`stage_cluster` is legacy (Phase 2) and ignored by Phase 1 tooling. If present in an existing CSV, it is preserved but unused.
`negative_training` is an image-level tag for explicitly marking negative examples (default `no`).

Per-image polygon annotations live under `data/labels/annotations/<image_id>.json`:
```json
{
  "image_id": "IMG_001",
  "rel_path": "subdir/IMG_001.tif",
  "include": "yes|no|unknown",
  "quality": "high|low|unknown",
  "tubules": [
    {
      "tubule_id": "IMG_001__0",
      "polygon_xy": [[x1, y1], [x2, y2], [x3, y3]],
      "orientation": "cross_section|longitudinal|oblique|unknown",
      "partial": false,
      "notes": ""
    }
  ],
  "labeled_by": "Adam",
  "labeled_at": "ISO-8601"
}
```
Polygon points are in original image pixel coordinates. Empty `tubules` is allowed (no tubules or excluded).
Orientation definitions:
- `cross_section`: round/near-round tubule cross-sections.
- `longitudinal`: elongated cuts along the tubule axis.
- `oblique`: in-between/angled orientations.
- `unknown`: not sure yet.
`partial=true` marks cropped/incomplete cross-sections (currently treated as positives in training).

## Annotation Tool (Polygons)
Launch with `python -m src.dataset.annotation_tool`.
- New tubule: click “New tubule”, then left-click to add vertices.
- Close polygon: right-click or press Enter to close and save that tubule.
- Backspace removes the last vertex while drawing.
- Select existing polygon by clicking inside or near a vertex; Delete removes it.
- Move vertex: click a vertex and drag.
- Save: writes JSON to `data/labels/annotations/` and updates `labels.csv` include/quality metadata.
- Use the top tabs to switch between include=yes/unknown and include=no images (kept separate).

Navigation and metadata:
- `Prev` / `Next` cycle through images (default filter include!=no).
- `I` cycles include; `Q` cycles quality.
- Set per-tubule orientation with the dropdown or press `O` to cycle.
- Toggle partial on the selected tubule with `P` (outline turns orange + “PARTIAL” label).
- Optional args: `--only-unlabeled`, `--include yes|unknown`, `--start <image_id>`.

Legacy `labeling_tool` (stage review UI) is disabled; use the polygon annotation tool for all labeling.

Legacy box tool is still available with `--mode box`.

## Training (Segmentation)
Baseline model: lightweight U-Net (randomly initialized, no downloads).
```bash
python -m src.training.seg_train --epochs 10 --batch-size 2 --device cpu \
  --max-size 1024 --val-ratio 0.2 --include-yes-only
```
Default training command:
```bash
python -m src.training.seg_train \
  --epochs 30 --batch-size 1 --device cpu \
  --include-yes-only \
  --require-nonempty-annotations \
  --balanced-loss \
  --area-penalty 1
```
Loss: BCEWithLogits + Dice. Targets use cross-section tubules (full + partial) as positives; longitudinal/oblique are explicit negatives.
Use `--allowed-orientations cross_section` to ignore longitudinal/oblique polygons (default: all).
Use `--require-nonempty-annotations` to skip images that have zero positive tubules annotated.
`--require-positive-annotations` (default true) enforces this even if `--require-nonempty-annotations` is false.
Use `--require-any-annotations` (default true) to require at least one tubule polygon (positive or negative).
If the model predicts everything, add `--balanced-loss` to penalize the dominant class.
If it still collapses to all-tubule, add `--area-penalty 0.5` (tune 0.1–1.0).
Use `--use-negative-training` to include empty annotations only when `negative_training=yes`.
Use `--use-negative-training` with `seg_eval` to match the same filter during evaluation.
If negatives are underrepresented, use `--negative-sample-weight 5` (or higher) to upsample them during training.
Use `--neg-tubule-weight` to upweight negative tubule pixels vs background (default 2.0).
Use `--bg-weight` to adjust background weighting (default 1.0).
Quality filtering is optional; pass `--only-high-quality` if you want to restrict to quality=high.

Exports best weights to `models/exported/seg_model.pt`.
Each completed training run also archives the exported model to `models/exported/past_models/` with a UTC timestamp in the filename.

## Evaluation (Segmentation)
```bash
python -m src.training.seg_eval --weights models/exported/seg_model.pt --save-viz
```
Postprocessing is the normal step for segmentation outputs (threshold -> clean small components/holes -> keep top K).
Default `K=4` matches the typical 2–4 tubules per image.
Example with postprocessing enabled:
```bash
python -m src.training.seg_eval --weights models/exported/seg_model.pt --save-viz --postprocess
```
Metrics: Dice, IoU, pixel precision/recall (computed on positive tubules only). Overlays are saved to `outputs/seg_viz/` when `--save-viz` is set.
Overlays: GT positive outline (green), GT negative tubules (blue), prediction outline (red).

## Inference CLI (Segmentation)
```bash
python -m src.app.inference --task seg --weights models/exported/seg_model.pt data/raw/example.tif \
  --output-mask-dir outputs/masks --output-json outputs/seg_results.json
```
Outputs a binary mask PNG per image and JSON with instance boxes + confidence.
Use `--split-instances` to separate connected tubule regions into per-tubule instances (optionally save a label map with `--output-instance-mask-dir`).

## Desktop GUI (Segmentation)
Tkinter app: `python -m src.app.gui`
- Load segmentation weights (auto-loads `models/exported/seg_model.pt` if present).
- Select TIFFs, view mask overlays and instance table.
- Export masks + JSON/CSV.

## Packaging (offline, no Python required for end users)
- After training/export, ensure `models/exported/seg_model.pt` exists.
- Build on Windows with PyInstaller (one-folder bundle includes Python + deps):
```powershell
scripts/build_app.ps1
```
Output: `dist/TubuleDetectorApp/`. Zip that folder and share; users unzip and run `TubuleDetectorApp/gui.exe`.

## Legacy Detection (Phase 1 - Boxes)
Legacy detector training/eval/inference still exists under:
- `src/training/detect_train.py`
- `src/training/detect_eval.py`
- `src/app/inference.py --task detect`
- `LegacyDetectorApp` in `src/app/gui.py`

## Phase 2 (Stage Classification - Legacy)
Classification code remains under `src/training/train.py`, `src/training/eval.py`, and `src/app/gui.py` (Phase 2 UI). It is not used by default Phase 1 commands.

## Testing
```bash
poetry run pytest
```
Tests use synthetic TIFFs and synthetic polygons; no real data required.

## VS Code
Launch configs are in `.vscode/launch.json` (annotation tool, segmentation training, GUI).
