"""Create an annotation workspace from a batch of ``overview_pipeline`` runs.

Usage::

    python -m eyespy_spermatogenesis.staging.annotate.prepare \
        outputs/tubule-review-2026-11-06-05 outputs/staging-annotation-2026-11-06-01

For every candidate in every ``*/manifest.json`` it reads the candidate's
``source_bounds`` from the original TIFF (read only) and writes a display JPEG
plus a transparent outline PNG at the same size. Display images keep native
pixels unless the long side exceeds ``--max-side``; the scale is recorded.
These JPEGs are for viewing only; analysis must use the original pixels.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image
from scipy import ndimage as ndi

from eyespy_spermatogenesis.io.regions import OverviewGeometry, TiffRegionReader
from eyespy_spermatogenesis.staging.annotate.store import Store

OUTLINE_RGBA = (255, 214, 0, 255)


def _source_job(args):
    run_dir, out_dir, max_side, quality, folder = args
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    labels = tifffile.imread(run_dir / "overview_labels.tif")
    g = manifest["geometry"]
    geometry = OverviewGeometry(
        g["source_width"],
        g["source_height"],
        g["level_width"],
        g["level_height"],
        g["factor"],
    )
    img_dir = out_dir / "images" / folder
    img_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with TiffRegionReader(
        manifest["source"]["path"], series=int(manifest["config"]["series"])
    ) as native:
        for c in manifest["candidates"]:
            bounds = tuple(int(v) for v in c["source_bounds"])
            lab = c["label"]
            name_img, name_out = f"c_{lab:05d}.jpg", f"o_{lab:05d}.png"
            w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
            scale = min(1.0, max_side / max(w, h))
            size = (max(1, round(w * scale)), max(1, round(h * scale)))
            if not (img_dir / name_img).exists() or not (img_dir / name_out).exists():
                crop = native.read_region(bounds)
                if crop.dtype != np.uint8:
                    crop = (crop >> 8).astype(np.uint8)
                im = Image.fromarray(crop)
                if scale < 1:
                    im = im.resize(size, Image.LANCZOS)
                im.save(img_dir / name_img, quality=quality)
                mask = geometry.project_labels(labels, bounds) == lab
                m = Image.fromarray(mask.astype(np.uint8) * 255)
                if scale < 1:
                    m = m.resize(size, Image.NEAREST)
                mk = np.asarray(m) > 0
                edge = mk & ~ndi.binary_erosion(mk, iterations=3)
                rgba = np.zeros(mk.shape + (4,), np.uint8)
                rgba[edge] = OUTLINE_RGBA
                Image.fromarray(rgba, "RGBA").save(img_dir / name_out, optimize=True)
            rows.append(
                {
                    "tubule_id": c["instance_id"],
                    "source_folder": folder,
                    "source_name": Path(manifest["source"]["path"]).name,
                    "source_sha256": manifest["source"].get("sha256"),
                    "label": lab,
                    "recommended": int(bool(c.get("crop_recommended"))),
                    "flags": ";".join(c.get("flags", [])),
                    "source_bounds": json.dumps(list(bounds)),
                    "image": f"images/{folder}/{name_img}",
                    "outline": f"images/{folder}/{name_out}",
                    "display_scale": round(scale, 5),
                }
            )
    return folder, rows


def prepare(run_parent: Path, workspace: Path, *, max_side=1600, quality=90, jobs=4):
    """Add every completed run under ``run_parent`` to ``workspace``.

    Safe to repeat and to call with several run parents: images go to
    ``images/<run_parent>/<source>/`` (except the first workspace, which used
    ``images/<source>/``), new tubules are ordered after existing ones, and
    labels are never touched.
    """
    runs = []
    for mp in sorted(run_parent.glob("*/manifest.json")):
        status = json.loads(mp.read_text()).get("status")
        if status != "complete_needs_review":
            print(f"Skipping {mp.parent.name}: {status}")
            continue
        runs.append(mp.parent)
    if not runs:
        raise SystemExit(f"No completed runs under {run_parent}")
    workspace.mkdir(parents=True, exist_ok=True)
    store = Store(workspace / "labels.sqlite")
    existing = {t["tubule_id"]: t for t in store.tubules()}
    info_path = workspace / "workspace.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    legacy_parent = info.get("run_parent")  # first workspace layout, kept as is
    prefix = "" if legacy_parent == str(run_parent.resolve()) else run_parent.name + "/"
    # Candidate ids are "<source sha>/series-N/candidate-<label>". Re-segmenting
    # a source changes what each label means, so a source may come from one
    # run folder only; otherwise old labels would point at different tubules.
    owner = {}
    for t in existing.values():
        owner.setdefault(t["source_sha256"], t["source_folder"])
    keep, seen = [], {}
    for r in runs:
        src = json.loads((r / "manifest.json").read_text())["source"]
        sha = src.get("sha256")
        name = Path(src["path"]).name
        if not sha:
            store.close()
            raise SystemExit(f"{r}: manifest has no source sha256; cannot check")
        if sha in owner and owner[sha] != prefix + r.name:
            print(f"Skipping {r.name} ({name}): already in workspace as {owner[sha]}")
            continue
        if sha in seen:
            print(f"Skipping {r.name} ({name}): same file as {seen[sha]} in this run")
            continue
        seen[sha] = r.name
        keep.append(r)
    runs = keep
    if not runs:
        store.close()
        print("Nothing new to add.")
        return []
    jobs_args = [
        (str(r), str(workspace), max_side, quality, prefix + r.name) for r in runs
    ]
    results = {}
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for folder, rows in pool.map(_source_job, jobs_args):
            results[folder] = rows
            print(f"{folder}: {len(rows)} candidates", flush=True)
    seq = 1 + max((t["seq"] for t in existing.values()), default=-1)
    all_rows = []
    for folder in sorted(results):
        rows = sorted(results[folder], key=lambda r: (not r["recommended"], r["label"]))
        for r in rows:
            if r["tubule_id"] in existing:
                r["seq"] = existing[r["tubule_id"]]["seq"]
            else:
                r["seq"] = seq
                seq += 1
        all_rows += rows
    store.add_tubules(all_rows)
    tubules = store.tubules()
    info.setdefault("run_parent", str(run_parent.resolve()))
    added = info.setdefault("run_parents", [info["run_parent"]])
    if str(run_parent.resolve()) not in added:
        added.append(str(run_parent.resolve()))
    info.update(
        {
            "sources": sorted({t["source_folder"] for t in tubules}),
            "candidates": len(tubules),
            "recommended": sum(t["recommended"] for t in tubules),
            "max_side": max_side,
            "jpeg_quality": quality,
            "note": "Display JPEGs are for viewing only; originals are unchanged.",
        }
    )
    info_path.write_text(json.dumps(info, indent=2))
    store.close()
    return all_rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run_parent", type=Path)
    p.add_argument("workspace", type=Path)
    p.add_argument("--max-side", type=int, default=1600)
    p.add_argument("--quality", type=int, default=90)
    p.add_argument("--jobs", type=int, default=4)
    a = p.parse_args(argv)
    rows = prepare(
        a.run_parent, a.workspace, max_side=a.max_side, quality=a.quality, jobs=a.jobs
    )
    print(f"{len(rows)} candidates, {sum(r['recommended'] for r in rows)} recommended")


if __name__ == "__main__":
    sys.exit(main())
