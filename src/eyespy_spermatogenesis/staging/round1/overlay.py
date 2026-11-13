"""Whole-section overlays for the annotation page's model-review Overview.

Usage::

    python -m eyespy_spermatogenesis.staging.round1.overlay \
        DATA_DIR EMB_DIR RESULTS_DIR WORKSPACE [--method "Step 1: ..."]

For the best configuration of each feature extractor in ``RESULTS_DIR``
(leave-one-section-out predictions, regularization by nested
leave-one-section-out) and each labelled section, writes JPEG panels on the
v0.5 overview (one colour per stage, the numeral at each tubule): the paper
stage, each model's stage, and each model's agreement with the paper (green
exact, amber one stage off, red further; grey = no stage or not evaluated).
Output: ``WORKSPACE/review/<run>/`` with ``overview.json`` (statistics, image
paths) served to the page. Display only.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi

from eyespy_spermatogenesis.staging.annotate.clickmaps import write_map, write_stages
from eyespy_spermatogenesis.staging.round1.data import STAGES, run_dir
from eyespy_spermatogenesis.staging.round1.evaluate import cyc_dist

GREY = (150, 150, 150)
AGREE = {0: (40, 160, 90), 1: (240, 170, 30), 2: (210, 50, 50)}


def stage_colours() -> dict[str, tuple[int, int, int]]:
    """Evenly spaced hues around the cycle, alternating lightness so
    neighbouring stages stay distinguishable."""
    out = {}
    for k, s in enumerate(STAGES):
        h = k / len(STAGES)
        v = 0.95 if k % 2 == 0 else 0.72
        r, g, b = colorsys.hsv_to_rgb(h, 0.8, v)
        out[s] = (round(r * 255), round(g * 255), round(b * 255))
    return out


def _font(size):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSerif-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def panel(overview, labels, colour_of, text_of, scale=2):
    base = overview.astype(np.float32)
    grey = base.mean(2, keepdims=True)
    img = 0.35 * base + 0.65 * grey  # faded tissue so colours read clearly
    lut = np.zeros((labels.max() + 1, 3), np.float32)
    has = np.zeros(labels.max() + 1, bool)
    for lab, c in colour_of.items():
        lut[lab], has[lab] = c, True
    m = has[labels]
    img[m] = 0.45 * img[m] + 0.55 * lut[labels][m]
    edge = labels != ndi.grey_erosion(labels, size=3)
    img[edge & (labels > 0)] = (255, 255, 255)
    im = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    im = im.resize((im.width * scale, im.height * scale), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    f = _font(13 * scale // 2)
    centres = ndi.center_of_mass(np.ones_like(labels), labels, list(text_of))
    for (lab, txt), (cy, cx) in zip(text_of.items(), centres):
        d.text(
            (cx * scale, cy * scale),
            txt,
            fill=(0, 0, 0),
            font=f,
            anchor="mm",
            stroke_width=2,
            stroke_fill=(255, 255, 255),
        )
    return im


def build(data_dir, emb_dir, results_dir, workspace, method, models=None):
    from eyespy_spermatogenesis.staging.round1.evaluate import metrics
    from eyespy_spermatogenesis.staging.round1.review import model_predictions

    results = json.loads((results_dir / "results.json").read_text())
    rows, idx, y, groups, preds = model_predictions(data_dir, emb_dir, results, models)
    pred_of = {
        name: {rows[i]["tubule_id"]: STAGES[k] for i, k in zip(idx, p)}
        for name, p in preds.items()
    }
    models = []
    for name, p in preds.items():
        m = metrics(y, p)
        models.append(
            {
                "name": name,
                "overall": {k: m[k] for k in ("n", "acc", "balanced_acc", "within1")},
                "recall": m["recall"],
            }
        )
    write_overview(workspace, results_dir.name, method, rows, pred_of, models)


def write_overview(workspace, run, method, rows, pred_of, models):
    """Panels + overview.json for one method (``run``) in WORKSPACE/review/.

    rows: label table rows (tubule_id, label, section, source_folder, stage);
    pred_of: model name -> {tubule_id: stage} (absent = not evaluated);
    models: [{"name", "overall": {n, acc, balanced_acc, within1, ...},
    "recall": {stage: r}}] for the statistics table."""
    out = workspace / "review" / run
    out.mkdir(parents=True, exist_ok=True)
    colours = stage_colours()
    by_sec = defaultdict(list)
    for r in rows:
        by_sec[(r["section"], r["source_folder"])].append(r)
    sections = []
    for (section, folder), rs in sorted(by_sec.items()):
        rd = run_dir(workspace, folder)
        overview = np.asarray(Image.open(rd / "overview.png").convert("RGB"))
        labels = tifffile.imread(rd / "overview_labels.tif")
        slug = section.lstrip("#")
        (out / slug).mkdir(exist_ok=True)
        paper_c = {r["label"]: colours[r["stage"]] for r in rs if r["stage"] in STAGES}
        paper_t = {r["label"]: short(r["stage"]) for r in rs if r["stage"] in STAGES}
        panel(overview, labels, paper_c, paper_t).save(
            out / slug / "paper.jpg", quality=85
        )
        entry = {"section": section, "paper": f"{slug}/paper.jpg", "models": {}}
        # Click data for the page (tubule grid; owner 2026-11-13).
        write_stages(out, rs, pred_of, section)
        write_map(workspace, section, folder)
        for mi, (name, pm) in enumerate(pred_of.items()):
            mc, mt, ac, dist = {}, {}, {}, []
            for r in rs:
                p = pm.get(r["tubule_id"])
                if p:
                    mc[r["label"]], mt[r["label"]] = colours[p], short(p)
                    d = int(cyc_dist(STAGES.index(p), STAGES.index(r["stage"])))
                    ac[r["label"]] = AGREE[min(d, 2)]
                    dist.append(d)
                else:
                    ac[r["label"]] = GREY
            panel(overview, labels, mc, mt).save(
                out / slug / f"m{mi}_stage.jpg", quality=85
            )
            panel(overview, labels, ac, {}).save(
                out / slug / f"m{mi}_agree.jpg", quality=85
            )
            dist = np.asarray(dist)
            entry["models"][name] = {
                "stage": f"{slug}/m{mi}_stage.jpg",
                "agree": f"{slug}/m{mi}_agree.jpg",
                "n": int(len(dist)),
                "exact": float((dist == 0).mean()) if len(dist) else None,
                "within1": float((dist <= 1).mean()) if len(dist) else None,
            }
        sections.append(entry)
        print(f"{section}: done", flush=True)
    (out / "overview.json").write_text(
        json.dumps(
            {
                "run": run,
                "method": method,
                "models": models,
                "sections": sections,
                "colours": {s: hexc(c) for s, c in colours.items()},
                "agree_colours": {
                    "exact": hexc(AGREE[0]),
                    "one stage off": hexc(AGREE[1]),
                    "two or more off": hexc(AGREE[2]),
                    "not evaluated / no stage": hexc(GREY),
                },
            },
            indent=1,
        )
    )
    print(f"-> {out}")


def hexc(c):
    return "#{:02x}{:02x}{:02x}".format(*c)


def short(stage):
    return stage.replace("II-III", "II")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", type=Path)
    p.add_argument("emb_dir", type=Path)
    p.add_argument("results_dir", type=Path)
    p.add_argument("workspace", type=Path)
    p.add_argument(
        "--method",
        default="Step 1: Yeongseok paper labels; frozen features + classifier",
    )
    p.add_argument("--models", nargs="+", help="extractors to show (default all)")
    a = p.parse_args(argv)
    build(a.data_dir, a.emb_dir, a.results_dir, a.workspace, a.method, a.models)


if __name__ == "__main__":
    sys.exit(main())
