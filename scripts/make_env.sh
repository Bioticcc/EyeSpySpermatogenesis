#!/usr/bin/env bash
set -e
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install poetry
poetry install --with dev
echo "Environment ready. Activate with 'source .venv/bin/activate'."
