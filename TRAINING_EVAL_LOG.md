# Training / Eval Log

Use this file to track training/eval commands and outcomes.

## Best so far (2026-01-27)

### Training
```bash
python -m src.training.seg_train \
  --allowed-orientations cross_section \
  --epochs 30 \
  --batch-size 1 \
  --device cpu \
  --include-yes-only \
  --require-nonempty-annotations \
  --balanced-loss \
  --area-penalty 0 \
  --ignore-band-px 32
```

### Eval
```bash
python -m src.training.seg_eval \
  --weights models/exported/seg_model.pt \
  --save-viz \
  --allowed-orientations cross_section \
  --metrics-ignore \
  --threshold 0.95
```

### Notes / Results
- Metrics:
- Visuals:
- Observations:

## Hard negatives update (2026-01-29)

### Training
```bash
python -m src.training.seg_train \
  --epochs 30 \
  --batch-size 1 \
  --device cpu \
  --include-yes-only \
  --no-require-positive-annotations \
  --require-any-annotations \
  --balanced-loss \
  --neg-tubule-weight 2.0 \
  --bg-weight 1.0 \
  --area-penalty 0 \
  --ignore-band-px 32
```

### Eval (best visual threshold so far)
```bash
python -m src.training.seg_eval \
  --weights models/exported/seg_model.pt \
  --save-viz \
  --metrics-ignore \
  --postprocess \
  --threshold 0.8
```

### Notes / Results
- Threshold sweep 0.4–0.9; ~0.8 looked best visually.

## Best so far with +40 training images (2026-01-28)

### Training
```bash
python -m src.training.seg_train \
  --allowed-orientations cross_section \
  --epochs 30 \
  --batch-size 1 \
  --device cpu \
  --include-yes-only \
  --require-nonempty-annotations \
  --balanced-loss \
  --area-penalty 0 \
  --ignore-band-px 32
```

### Eval
```bash
python -m src.training.seg_eval \
  --weights models/exported/seg_model.pt \
  --save-viz \
  --allowed-orientations cross_section \
  --metrics-ignore \
  --threshold 0.66
```

### Notes / Results
- Best so far after adding ~40 training images; threshold tuned to 0.66.

## Template

### Training
```bash
python -m src.training.seg_train \
  --allowed-orientations cross_section \
  --epochs 30 \
  --batch-size 1 \
  --device cpu \
  --include-yes-only \
  --require-nonempty-annotations \
  --balanced-loss \
  --area-penalty 0 \
  --ignore-band-px 32
```

### Eval
```bash
python -m src.training.seg_eval \
  --weights models/exported/seg_model.pt \
  --save-viz \
  --allowed-orientations cross_section \
  --metrics-ignore \
  --threshold 0.95
```

### Notes / Results
- Metrics:
- Visuals:
- Observations:
