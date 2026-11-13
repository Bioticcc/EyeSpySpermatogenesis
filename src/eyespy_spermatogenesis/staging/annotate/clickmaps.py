"""Click data for the annotation page's Overview panels (owner, 2026-11-13).

Clicking a tubule on an Overview panel opens a grid of that section's tubules
with the same stage (by that panel), and from there one tubule full size. The
page needs, per section, which tubule is under each pixel and, per panel, the
stage of every tubule:

* ``WORKSPACE/review/maps/<section>.png``: the v0.5 overview label image,
  candidate label encoded as red + 256 * green (0 = background). Lossless.
* ``WORKSPACE/review/<run>/<section>/stages.json``: ``{"truth": {label:
  stage}, "models": {model name: {label: stage}}}`` for a Method overview run
  (written by ``overlay.write_overview``; older runs by ``backfill``).

* ``WORKSPACE/review/sections/<section>.jpg``: display copy of the overview,
  for the staging Progress page's section map (coloured in the browser from
  the live labels).

Usage (once, for runs made before 2026-11-13; ``sections`` also runs at the
end of ``annotate.prepare``)::

    python -m eyespy_spermatogenesis.staging.annotate.clickmaps backfill WORKSPACE
    python -m eyespy_spermatogenesis.staging.annotate.clickmaps sections WORKSPACE

``backfill`` rebuilds each old run's per-tubule stages from its saved
predictions and checks them against the per-section scores already in its
``overview.json``; it stops on any mismatch. Display only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

from eyespy_spermatogenesis.staging.round1.data import STAGES, read_table, run_dir

ROUND1 = Path("/mnt/20TB_A/Adam/EyeSpySpermatogenesisRebooted/staging-round1")
REPO = Path(__file__).resolve().parents[4]


def slug(section: str) -> str:
    return section.lstrip("#")


def tubule_index(workspace: Path) -> dict[str, tuple[str, str, int]]:
    """tubule_id -> (section, source_folder, candidate label)."""
    db = sqlite3.connect(f"file:{workspace / 'labels.sqlite'}?mode=ro", uri=True)
    rows = db.execute(
        "SELECT tubule_id, source_name, source_folder, label FROM tubules"
    ).fetchall()
    db.close()
    return {t: (Path(n).stem, f, int(lab)) for t, n, f, lab in rows}


def write_map(workspace: Path, section: str, folder: str) -> Path:
    """Label image of one section, encoded for the browser (see module doc)."""
    import tifffile
    from PIL import Image

    out = workspace / "review" / "maps" / f"{slug(section)}.png"
    if out.exists():
        return out
    labels = tifffile.imread(run_dir(workspace, folder) / "overview_labels.tif")
    if labels.max() >= 1 << 16:
        raise ValueError(f"{section}: more than 65535 candidates")
    rgb = np.zeros((*labels.shape, 3), np.uint8)
    rgb[..., 0] = labels & 255
    rgb[..., 1] = labels >> 8
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out, optimize=True)
    return out


def write_stages(out_dir: Path, rows, pred_of: dict, section: str):
    """``stages.json`` for one section of a Method overview run.

    rows: that section's label-table rows (tubule_id, label, stage); pred_of:
    model name -> {tubule_id: stage}. Matches what the panels show: truth =
    every row with a stage; a model = every tubule it staged."""
    truth = {int(r["label"]): r["stage"] for r in rows if r["stage"] in STAGES}
    models = {
        name: {
            int(r["label"]): pm[r["tubule_id"]] for r in rows if pm.get(r["tubule_id"])
        }
        for name, pm in pred_of.items()
    }
    d = out_dir / slug(section)
    d.mkdir(parents=True, exist_ok=True)
    (d / "stages.json").write_text(
        json.dumps({"section": section, "truth": truth, "models": models})
    )


# ---------------------------------------------------------------- backfill
def preds_round1(results_dir: Path):
    from eyespy_spermatogenesis.staging.round1.review import model_predictions

    results = json.loads((results_dir / "results.json").read_text())
    rows, idx, _y, _g, preds = model_predictions(
        ROUND1 / "data", ROUND1 / "embeddings", results, None
    )
    return {
        name: {rows[i]["tubule_id"]: STAGES[k] for i, k in zip(idx, p)}
        for name, p in preds.items()
    }


def preds_round2(results_dir: Path, names: list[str]):
    from eyespy_spermatogenesis.staging.round2 import CONF

    with open(results_dir / "predictions.csv", newline="") as f:
        rs = list(csv.DictReader(f))
    plain, conf = names
    if f">= {CONF:.2f}" not in conf:
        raise SystemExit(f"Round 2 confidence model not recognised: {conf}")
    # Unrounded confidence (the CSV rounds to 3 decimals): max of the 12-class
    # probabilities at the chosen setting, as in round2.evaluate.
    ch = json.loads((results_dir / "results.json").read_text())["chosen"]
    pz = np.load(results_dir / "probs.npz")
    p12 = pz[f"{ch['side_px']}px_{ch['contrast']}"]
    confidence = dict(zip(pz["tubule_id"].tolist(), p12.max(1)))
    for r in rs:
        r["confidence"] = confidence[r["tubule_id"]]
    return {
        plain: {r["tubule_id"]: r["predicted"] for r in rs},
        conf: {
            r["tubule_id"]: r["predicted"] for r in rs if float(r["confidence"]) >= CONF
        },
    }


def preds_round3(results_dir: Path, names: list[str]):
    import ast

    d = np.load(results_dir / "ensemble_predictions.npz")
    src = (results_dir / "overview.py").read_text()
    # The NAMES mapping (npz key -> panel name) as written in overview.py.
    start = src.index("{", src.index("NAMES = {"))
    names_map = ast.literal_eval(src[start : src.index("}", start) + 1])
    out = {}
    for key, name in names_map.items():
        out[name] = {t: STAGES[k] for t, k in zip(d["tubule_id"], d[key])}
    missing = set(names) - set(out)
    if missing:
        raise SystemExit(f"Round 3 models not found: {missing}")
    return out


def check(ov: dict, run_dir_: Path):
    """Per-section n and exact from stages.json must equal overview.json."""
    bad = []
    for sec in ov["sections"]:
        st = json.loads((run_dir_ / slug(sec["section"]) / "stages.json").read_text())
        truth = st["truth"]
        for name, m in sec["models"].items():
            pm = st["models"].get(name, {})
            hits = [pm[k] == truth[k] for k in pm if k in truth]
            n = len(hits)
            exact = float(np.mean(hits)) if n else None
            if n != m["n"] or (exact is not None and abs(exact - m["exact"]) > 1e-9):
                bad.append((sec["section"], name, n, m["n"], exact, m["exact"]))
    if bad:
        raise SystemExit(f"stages.json does not match overview.json: {bad[:5]}")


def backfill(workspace: Path, outputs: Path = REPO / "outputs"):
    rows = read_table(ROUND1 / "data")
    by_sec: dict[str, list] = {}
    for r in rows:
        by_sec.setdefault(r["section"], []).append(r)
    for ovp in sorted((workspace / "review").glob("*/overview.json")):
        ov = json.loads(ovp.read_text())
        run = ovp.parent
        names = [m["name"] for m in ov["models"]]
        res = outputs / run.name
        if "round1" in run.name:
            pred_of = preds_round1(res)
        elif "round2" in run.name:
            pred_of = preds_round2(res, names)
        elif "round3" in run.name:
            pred_of = preds_round3(res, names)
        else:
            print(f"{run.name}: unknown run, skipped")
            continue
        pred_of = {n: pred_of[n] for n in names}
        for sec in ov["sections"]:
            write_stages(run, by_sec[sec["section"]], pred_of, sec["section"])
        check(ov, run)
        print(f"{run.name}: stages.json written and checked", flush=True)
    write_maps(workspace)


def write_maps(workspace: Path):
    """Click maps for every section shown in any overview or the EyeSpy tab."""
    folders = {sec: f for sec, f, _ in tubule_index(workspace).values()}
    sections = set()
    for ovp in (workspace / "review").glob("*/overview.json"):
        sections |= {s["section"] for s in json.loads(ovp.read_text())["sections"]}
    es = workspace / "review" / "eyespy" / "index.json"
    if es.exists():
        sections |= {s["section"] for s in json.loads(es.read_text())["samples"]}
    for sec in sorted(sections):
        write_map(workspace, sec, folders[sec])
    print(f"{len(sections)} click maps in {workspace / 'review' / 'maps'}")


def write_overview_image(workspace: Path, section: str, folder: str) -> Path:
    """Display copy of a section's v0.5 overview (Progress section map)."""
    from PIL import Image

    out = workspace / "review" / "sections" / f"{slug(section)}.jpg"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(run_dir(workspace, folder) / "overview.png").convert("RGB")
    im.save(out, quality=85)
    return out


def write_sections(workspace: Path):
    """Overview image + click map for every section in the workspace, for the
    staging Progress page's section map (owner, 2026-11-13). Existing files are
    kept, so this is cheap to repeat after adding sections."""
    folders = {sec: f for sec, f, _ in tubule_index(workspace).values()}
    for sec, folder in sorted(folders.items()):
        write_map(workspace, sec, folder)
        write_overview_image(workspace, sec, folder)
    print(f"{len(folders)} section maps in {workspace / 'review'}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("backfill", help="stages.json for old runs + click maps")
    b.add_argument("workspace", type=Path)
    m = sub.add_parser("maps", help="click maps only")
    m.add_argument("workspace", type=Path)
    s = sub.add_parser("sections", help="overview image + click map, every section")
    s.add_argument("workspace", type=Path)
    a = p.parse_args(argv)
    if a.cmd == "backfill":
        backfill(a.workspace)
    elif a.cmd == "sections":
        write_sections(a.workspace)
    else:
        write_maps(a.workspace)


if __name__ == "__main__":
    sys.exit(main())
