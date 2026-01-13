# EyeSpySpermatogenesis

Offline Windows desktop app + training pipeline to classify mouse testis seminiferous tubule TIFF images into 6 stage clusters: `XII_I`, `II_IV`, `V_VI`, `VII_VIII`, `IX_X`, `XI`.

## Quickstart
- Create env: `scripts/make_env.sh` (Linux/macOS) or `scripts/make_env.ps1` (Windows). Uses Poetry; Python 3.10+.
- Index data: place `.tif/.tiff` files under `data/raw/`, then `python -m src.dataset.indexing`.
- Label: `python -m src.dataset.labeling_tool --list --only-unlabeled` or `python -m src.dataset.labeling_tool --review`.
- Split: `python -m src.dataset.splits --train-ratio 0.7 --val-ratio 0.15`.
- Train: `python -m src.training.train --config configs/train_example.json` (adjust epochs, device).
- Evaluate: `python -m src.training.eval --split-path data/splits/test.csv`.
- GUI: `python -m src.app.gui` (warns if weights missing).
- Build app (Windows): `scripts/build_app.ps1` (one-folder PyInstaller bundle).

## Repository Layout
- `data/raw/` raw TIFFs (ignored by git)
- `data/labels/labels.csv` label + tags (tracked)
- `data/splits/` train/val/test CSVs (ignored)
- `data/processed/` cached tensors (ignored)
- `models/exported/model.pt` exported weights for inference/GUI (ignored; `.gitkeep` placeholder)
- `src/` Python modules (dataset, training, app)
- `scripts/` helper scripts
- `tests/` lightweight pytest checks (synthetic TIFFs)

## Labeling & Tagging
`data/labels/labels.csv` columns: `image_id, rel_path, stage_cluster, quality, include, notes, labeled_by, labeled_at`.

1. Scan for new files: `python -m src.dataset.indexing`.
2. List/filter: `python -m src.dataset.labeling_tool --list --only-unlabeled --include-not-yes`.
3. Update a single file: `python -m src.dataset.labeling_tool --image path/to/file.tif --stage V_VI --quality high --include yes --notes "good" --labeled-by me`.
4. Review UI: `python -m src.dataset.labeling_tool --review` (Tkinter viewer with dropdowns).

`stage_cluster` accepts the 6 clusters or `unknown`. `quality` = `high|low|unknown`. `include` = `yes|no|unknown`. Training defaults to `include=yes` and `quality=high`.

## Data Splits
Create reproducible stratified splits on labeled, included data:
```bash
python -m src.dataset.splits --train-ratio 0.7 --val-ratio 0.15 --seed 42
```
Outputs `train.csv`, `val.csv`, `test.csv` under `data/splits/`.

## Training
Config is JSON (see `configs/train_example.json`). Key flags:
- `include_low_quality` (default false) and `low_quality_weight` (loss weight if low quality is included).
- Class imbalance handled via class weights computed on the training set.

Run:
```bash
python -m src.training.train --config configs/train_example.json --epochs 10 --batch-size 16 --device cpu
```
Best checkpoint saved to `models/checkpoints/best.pt` and exported weights to `models/exported/model.pt`.

## Evaluation
```bash
python -m src.training.eval --split-path data/splits/test.csv --checkpoint models/exported/model.pt
```
Reports accuracy + macro F1 and writes `models/confusion_matrix.csv`.

## Inference CLI
```bash
python -m src.app.inference --weights models/exported/model.pt data/raw/example.tif
```
Shows predicted cluster + confidence + top-3 probabilities per file.

## Desktop GUI
Tkinter app: `python -m src.app.gui`
- Load weights (auto-loads `models/exported/model.pt` if present).
- Select one or multiple TIFFs.
- Shows prediction, confidence, and top-3; can export results to CSV.
- If weights are missing, the GUI stays usable but prompts to load weights.

## Preprocessing (optional cache)
Cache normalized tensors to `data/processed/`:
```bash
python -m src.dataset.preprocessing --image-size 224
```

## Packaging (offline, no Python required for end users)
- After training/export, ensure `models/exported/model.pt` exists.
- Build on Windows with PyInstaller (one-folder bundle includes Python + deps):
```powershell
scripts/build_app.ps1
```
Output: `dist/TubuleStageApp/`. Zip that folder and share; users unzip and run `TubuleStageApp/gui.exe` (no Python install needed). If weights were missing, the app still opens but will prompt for weights.

## Developer Notes
- Reproducibility: `src/utils/seed.py` seeds Python, NumPy, and Torch.
- TIFF I/O: `src/utils/tiff_io.py` handles 8/16-bit, grayscale/RGB, multipage (first page), normalizes to float32 `[0,1]`.
- Augmentations: basic flips/rotations for training; configurable image size.
- Everything runs even with empty `data/raw/` (tools log and exit cleanly).

## Testing
```bash
poetry run pytest
```
Tests use synthetic TIFFs; no real data required.

## VS Code
- Launch configs for training and GUI in `.vscode/launch.json`.
- Default interpreter `.venv/bin/python`; adjust if needed.
