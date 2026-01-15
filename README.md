# EyeSpySpermatogenesis

Offline Windows desktop app + training pipeline for Phase 1: seminiferous tubule localization (object detection) in TIFF images. Phase 2 stage classification remains in the repo as legacy/future work.

## Quickstart (Phase 1: Detection)
- Create env: `scripts/make_env.sh` (Linux/macOS) or `scripts/make_env.ps1` (Windows). Uses Poetry; Python 3.10+.
- Index data: place `.tif/.tiff` files under `data/raw/`, then `python -m src.dataset.indexing`.
- Annotate: `python -m src.dataset.annotation_tool` (draw bounding boxes, save JSON).
- Train detector: `python -m src.training.detect_train --epochs 5 --batch-size 2 --device cpu`.
- Evaluate: `python -m src.training.detect_eval --weights models/exported/detector.pt`.
- Inference CLI: `python -m src.app.inference --weights models/exported/detector.pt data/raw/example.tif`.
- GUI: `python -m src.app.gui`.
- Build app (Windows): `scripts/build_app.ps1` (one-folder PyInstaller bundle).

## Repository Layout
- `data/raw/` raw TIFFs (ignored by git)
- `data/labels/labels.csv` image-level metadata (tracked)
- `data/labels/annotations/` per-image box annotations (tracked)
- `data/splits/` phase-2 train/val/test CSVs (ignored)
- `models/exported/detector.pt` exported detector weights (ignored; `.gitkeep` placeholder)
- `models/exported/model.pt` phase-2 classifier weights (ignored; `.gitkeep` placeholder)
- `src/` Python modules (dataset, training, app)
- `scripts/` helper scripts
- `tests/` lightweight pytest checks (synthetic TIFFs)

## Labels and Annotations
`data/labels/labels.csv` columns:
`image_id, rel_path, quality, include, notes, labeled_by, labeled_at`

`stage_cluster` is legacy (Phase 2) and ignored by Phase 1 tooling. If present in an existing CSV, it is preserved but unused.

Per-image annotations live under `data/labels/annotations/<image_id>.json`:
```json
{
  "image_id": "...",
  "rel_path": "...",
  "tissue": "testis_tubule",
  "boxes": [
    {"box_id": "...", "bbox_xywh": [x, y, w, h], "label": "tubule", "notes": ""}
  ],
  "labeled_by": "...",
  "labeled_at": "ISO-8601"
}
```
`bbox_xywh` is always in original image pixel coordinates. Empty `boxes` is allowed (no tubules).

## Annotation Tool
Launch with `python -m src.dataset.annotation_tool`.
- Draw box: click + drag on the image.
- Select box: click in list; delete with Delete key.
- Redraw: select a box, click “Redraw Selected”, then draw replacement.
- Save: writes JSON to `data/labels/annotations/` and updates `labels.csv` quality/include metadata.

## Training (Detection)
Baseline detector: Faster R-CNN (MobileNetV3 FPN), randomly initialized (no downloads).
```bash
python -m src.training.detect_train --epochs 10 --batch-size 2 --device cpu \
  --max-size 1024 --val-ratio 0.2 --include-yes-only
```
Key flags:
- `--only-high-quality` to restrict to `quality=high`.
- `--include-yes-only` to filter to `include=yes`.
- `--require-annotations` to enforce JSON presence.
- `--max-size` to downscale large images for memory.

Exports best weights to `models/exported/detector.pt`.

## Evaluation (Detection)
```bash
python -m src.training.detect_eval --weights models/exported/detector.pt --save-viz
```
Metrics: precision/recall @ IoU 0.5, mean IoU, and count error (predicted vs. true tubule count). Overlays are saved to `outputs/detect_viz/` when `--save-viz` is set.

## Inference CLI (Detection)
```bash
python -m src.app.inference --weights models/exported/detector.pt data/raw/example.tif \
  --output-json outputs/detect.json --output-csv outputs/detect.csv
```
Outputs per-image detections with `bbox_xywh` and confidence score.

## Desktop GUI (Detection)
Tkinter app: `python -m src.app.gui`
- Load detector weights (auto-loads `models/exported/detector.pt` if present).
- Select TIFFs, view box overlays and confidence table.
- Export detections to JSON/CSV.

## Packaging (offline, no Python required for end users)
- After training/export, ensure `models/exported/detector.pt` exists.
- Build on Windows with PyInstaller (one-folder bundle includes Python + deps):
```powershell
scripts/build_app.ps1
```
Output: `dist/TubuleDetectorApp/`. Zip that folder and share; users unzip and run `TubuleDetectorApp/gui.exe`.

## Phase 2 (Stage Classification - Legacy)
Classification code remains under `src/training/train.py`, `src/training/eval.py`, and `src/app/gui.py` (Phase 2 UI). It is not used by default Phase 1 commands.

## Testing
```bash
poetry run pytest
```
Tests use synthetic TIFFs and synthetic boxes; no real data required.

## VS Code
Launch configs are in `.vscode/launch.json` (annotation tool, detector training, GUI).
