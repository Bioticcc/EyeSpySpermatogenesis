#!/usr/bin/env bash
set -e
poetry run python -m src.dataset.indexing
poetry run python -m src.dataset.splits --train-ratio 0.7 --val-ratio 0.15 --seed 42
poetry run python -m src.training.train --config configs/train_example.json
