#!/usr/bin/env python3
"""List Yeongseok staging inputs that are not on the annotation page yet.

Usage: .venv/bin/python scripts/check_new_staging_inputs.py [WORKSPACE]

Checks (read-only):
  * TIFFs in ``Inputs/YeongSeok_Staging`` whose contents (sha256) are not a
    section in the workspace: they need segmenting and ``prepare``.
  * Photos/scans in ``Inputs/Staging annotations`` not used by any
    ``outputs/paper-alignment-*/**/summary.json``, and not listed as a known
    duplicate in ``outputs/paper-alignment-*/duplicates.json``: new paper
    sheets to align, transcribe and import (docs/ANNOTATION.md).
Step 1 of the staging plan trains only on these labels, so new ones may
justify a retrain: tell the owner (docs/STAGING_STRATEGY.md).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TIFFS = ROOT / "Inputs" / "YeongSeok_Staging"
SHEETS = ROOT / "Inputs" / "Staging annotations"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(workspace: Path) -> int:
    db = sqlite3.connect(f"file:{workspace / 'labels.sqlite'}?mode=ro", uri=True)
    known = {r[0] for r in db.execute("SELECT DISTINCT source_sha256 FROM tubules")}
    db.close()
    new_tiffs = [p.name for p in sorted(TIFFS.glob("*.tif*")) if sha256(p) not in known]

    used = set()
    for s in ROOT.glob("outputs/paper-alignment-*/**/summary.json"):
        used.add(Path(json.loads(s.read_text())["scan"]).name)
    for d in ROOT.glob("outputs/paper-alignment-*/duplicates.json"):
        used.update(json.loads(d.read_text()))
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".heic"}
    new_sheets = [
        p.name
        for p in sorted(SHEETS.iterdir())
        if p.suffix.lower() in exts and p.name not in used
    ]

    print(f"New TIFFs in {TIFFS.relative_to(ROOT)}: {new_tiffs or 'none'}")
    print(f"New sheets in {SHEETS.relative_to(ROOT)}: {new_sheets or 'none'}")
    return 1 if new_tiffs or new_sheets else 0


if __name__ == "__main__":
    ws = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "outputs" / "staging-annotation-2026-11-06-01"
    )
    sys.exit(main(ws))
