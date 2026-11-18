#!/usr/bin/env bash
# Sync this workstation with the hosted annotation page (decision 028).
# Usage: scripts/sync.sh setup|deploy [--now [--force]|--cancel]|status|pull|checkout|checkin [--discard]|push-files
# Settings: hosting.toml in the repository root (Git-ignored; see docs/HOSTING.md).
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m eyespy_spermatogenesis.staging.annotate.hosting "$@"
