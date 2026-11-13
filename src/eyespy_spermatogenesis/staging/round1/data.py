"""Label table and native crops for staging round 1.

Usage::

    python -m eyespy_spermatogenesis.staging.round1.data WORKSPACE OUT_DIR [--jobs 6] \
        [--origin paper|web] [--annotator Yeongseok]

Reads staging labels of one origin (default ``paper``; ``web`` = labels made
on the annotation page, used by EyeSpy from 2026-11-13, decision 027) from
``WORKSPACE/labels.sqlite``
and writes, for every labelled candidate, the native-resolution crop
(``crops/<id>.png``, lossless) and its v0.5 outline mask (``masks/<id>.png``)
read from the original TIFF, plus ``labels.csv``. Originals are only read.
Existing crops are reused, so the command is safe to repeat.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image

STAGES = ["I", "II-III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"]

COLUMNS = [
    "key",
    "tubule_id",
    "section",
    "source_folder",
    "label",
    "recommended",
    "stage",
    "needs_check",
    "transcribed_by",
    "source_bounds",
]


def crop_key(tubule_id: str) -> str:
    """Short, file-safe, stable name for a candidate."""
    return hashlib.sha1(tubule_id.encode()).hexdigest()[:16]


def load_labels(
    workspace: Path, annotator: str = "Yeongseok", origin: str = "paper"
) -> list[dict]:
    """One row per labelled candidate of ``origin`` (any value, incl. none)."""
    db = sqlite3.connect(f"file:{workspace / 'labels.sqlite'}?mode=ro", uri=True)
    db.row_factory = sqlite3.Row
    rows = db.execute(
        """SELECT t.tubule_id, t.source_name, t.source_folder, t.label,
                  t.recommended, t.source_bounds, l.value, l.detail
           FROM labels l JOIN tubules t USING (tubule_id)
           WHERE l.origin = ? AND l.mode = 'staging' AND l.annotator = ?
           ORDER BY t.seq""",
        (origin, annotator),
    ).fetchall()
    db.close()
    out = []
    for r in rows:
        d = json.loads(r["detail"] or "{}")
        out.append(
            {
                "key": crop_key(r["tubule_id"]),
                "tubule_id": r["tubule_id"],
                "section": Path(r["source_name"]).stem,
                "source_folder": r["source_folder"],
                "label": r["label"],
                "recommended": int(r["recommended"]),
                "stage": r["value"] or "",
                "needs_check": int(bool(d.get("needs_check"))),
                "transcribed_by": d.get("transcribed_by", ""),
                "source_bounds": r["source_bounds"],
            }
        )
    return out


def run_dir(workspace: Path, folder: str) -> Path:
    """Run source folder of a workspace section (see annotate.prepare layout)."""
    info = json.loads((workspace / "workspace.json").read_text())
    if "/" not in folder:
        return Path(info["run_parent"]) / folder
    parent, src = folder.split("/", 1)
    for p in info.get("run_parents", []):
        if Path(p).name == parent:
            return Path(p) / src
    raise KeyError(f"No run parent for {folder}")


def _export_section(args):
    from eyespy_spermatogenesis.io.regions import OverviewGeometry, TiffRegionReader

    run, out_dir, items = args
    run, out_dir = Path(run), Path(out_dir)
    manifest = json.loads((run / "manifest.json").read_text())
    labels = tifffile.imread(run / "overview_labels.tif")
    g = manifest["geometry"]
    geom = OverviewGeometry(
        g["source_width"],
        g["source_height"],
        g["level_width"],
        g["level_height"],
        g["factor"],
    )
    done = 0
    with TiffRegionReader(
        manifest["source"]["path"], series=int(manifest["config"]["series"])
    ) as native:
        for key, lab, bounds in items:
            cp, mp = out_dir / "crops" / f"{key}.png", out_dir / "masks" / f"{key}.png"
            if cp.exists() and mp.exists():
                continue
            crop = native.read_region(tuple(bounds))
            if crop.dtype != np.uint8:
                crop = (crop >> 8).astype(np.uint8)
            Image.fromarray(np.ascontiguousarray(crop[..., :3])).save(cp)
            mask = geom.project_labels(labels, tuple(bounds)) == lab
            Image.fromarray(mask.astype(np.uint8) * 255).save(mp)
            done += 1
    return run.name, done


def export(
    workspace: Path,
    out_dir: Path,
    jobs: int = 6,
    origin: str = "paper",
    annotator: str = "Yeongseok",
) -> list[dict]:
    rows = load_labels(workspace, annotator, origin)
    (out_dir / "crops").mkdir(parents=True, exist_ok=True)
    (out_dir / "masks").mkdir(parents=True, exist_ok=True)
    by_run = defaultdict(list)
    for r in rows:
        by_run[str(run_dir(workspace, r["source_folder"]))].append(
            (r["key"], r["label"], json.loads(r["source_bounds"]))
        )
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for name, n in pool.map(
            _export_section, [(k, str(out_dir), v) for k, v in by_run.items()]
        ):
            print(f"{name}: {n} new crops", flush=True)
    with open(out_dir / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "data.json").write_text(
        json.dumps(
            {
                "workspace": str(workspace.resolve()),
                "rows": len(rows),
                "origin": origin,
                "annotator": annotator,
                "note": "Native-resolution crops (lossless PNG) and v0.5 outline "
                "masks; originals unchanged. Labels are unchecked "
                + ("transcriptions." if origin == "paper" else "on-screen labels."),
            },
            indent=2,
        )
    )
    return rows


def read_table(out_dir: Path) -> list[dict]:
    with open(out_dir / "labels.csv", newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("label", "recommended", "needs_check"):
            r[k] = int(r[k])
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("workspace", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--jobs", type=int, default=6)
    p.add_argument("--origin", choices=("paper", "web"), default="paper")
    p.add_argument("--annotator", default="Yeongseok")
    a = p.parse_args(argv)
    rows = export(a.workspace, a.out_dir, a.jobs, a.origin, a.annotator)
    print(f"{len(rows)} labelled candidates -> {a.out_dir}")


if __name__ == "__main__":
    sys.exit(main())
