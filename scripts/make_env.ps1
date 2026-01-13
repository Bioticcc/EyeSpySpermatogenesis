$ErrorActionPreference = "Stop"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install poetry
poetry install --with dev
Write-Host "Environment ready. Activate with .\.venv\Scripts\Activate.ps1"
