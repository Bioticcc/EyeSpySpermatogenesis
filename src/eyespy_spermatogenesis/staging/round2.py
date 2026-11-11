"""Staging round 2 (SATINN): pretrained SATINN stager, no training on our labels.

Usage::

    python -m eyespy_spermatogenesis.staging.round2 \
        DATA_DIR WORKSPACE NET_NPZ OUT_DIR [--calib 300]

DATA_DIR is the round 1 data folder (``labels.csv``); NET_NPZ the converted
``bf_tnetr-*.npz`` (``staging.satinn``). SATINN's bright-field preprocessing
is undocumented, so it is chosen **without labels**: for every box size
(native pixels around the tubule centre, resized to the network's 500x500)
and contrast setting, the network runs on ``--calib`` randomly chosen tubules
and the setting whose activations best match the training statistics stored
in its batch-norm layers is selected. Only then are predictions scored against
the paper stages (same tubules and metrics as round 1). Scores for every
setting are reported too, as a sensitivity table, but play no part in the
choice.

Contrast settings: ``raw`` (pixels as read), or the image's mean colour moved
to SATINN's training mean and deviations multiplied by a gain.

Outputs in OUT_DIR: ``results.json``, ``summary.md``, ``predictions.csv``,
``confusion_loso.png`` (name kept for symmetry; no training is involved),
``probs.npz`` (12-class probabilities for every labelled candidate at the
chosen and all other settings, for round 4). Overview panels go to
``WORKSPACE/review/<OUT_DIR name>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from eyespy_spermatogenesis.staging.round1.data import STAGES, read_table, run_dir

SIDES = (330, 500, 650, 800, 1050, 1330, 1600)
GAINS = ("raw", 1.0, 2.0, 2.5, 3.0, 3.5)
IN = 500
CONF = 0.80  # SATINN discards predictions below this confidence
METHOD = "SATINN (step 2): pretrained, no training on our labels"


def to_ours(p12: np.ndarray) -> np.ndarray:
    """12 SATINN classes (II and III separate) -> our 11 (II-III merged)."""
    return np.concatenate(
        [p12[:, :1], p12[:, 1:3].sum(1, keepdims=True), p12[:, 3:]], 1
    )


def _read_section(args):
    from eyespy_spermatogenesis.io.regions import TiffRegionReader

    run, items, side = args
    manifest = json.loads((Path(run) / "manifest.json").read_text())
    W, H = manifest["geometry"]["source_width"], manifest["geometry"]["source_height"]
    out = {}
    with TiffRegionReader(
        manifest["source"]["path"], series=int(manifest["config"]["series"])
    ) as native:
        for i, b in items:
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
            box = np.full((side, side, 3), 255, np.uint8)  # outside the slide: white
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(W, x0 + side), min(H, y0 + side)
            if cx1 > cx0 and cy1 > cy0:
                a = native.read_region((cx0, cy0, cx1, cy1))
                if a.dtype != np.uint8:
                    a = (a >> 8).astype(np.uint8)
                box[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0] = a[..., :3]
            out[i] = np.asarray(Image.fromarray(box).resize((IN, IN), Image.BICUBIC))
    return out


def read_boxes(workspace: Path, rows, side: int, jobs=8) -> np.ndarray:
    by_run = defaultdict(list)
    for i, r in enumerate(rows):
        by_run[str(run_dir(workspace, r["source_folder"]))].append(
            (i, json.loads(r["source_bounds"]))
        )
    arr = np.empty((len(rows), IN, IN, 3), np.uint8)
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for got in pool.map(_read_section, [(k, v, side) for k, v in by_run.items()]):
            for i, a in got.items():
                arr[i] = a
    return arr


def preprocess(x, gain, mean):
    """x: torch uint8 (N, 500, 500, 3) on device -> float (N, 3, 500, 500)."""
    x = x.permute(0, 3, 1, 2).float()
    if gain == "raw":
        return x
    m = mean.view(1, 3, 1, 1)
    return (m + gain * (x - x.mean((2, 3), keepdim=True))).clamp(0, 255)


def run_net(net, arr, gain, idx, batch=16, bn_acc=None):
    """Probabilities (len(idx), 12); optionally accumulate batch-norm stats."""
    import torch

    mean = net.g("input", "mean").reshape(3)
    taps = None
    if bn_acc is not None:

        def tap(n):
            def f(h):
                a = bn_acc.setdefault(n, [0.0, 0.0, 0])
                h = h.float()
                a[0] = a[0] + h.sum((0, 2, 3))
                a[1] = a[1] + (h * h).sum((0, 2, 3))
                a[2] += h[:, 0].numel()

            return f

        taps = {n: tap(n) for n in bn_layers(net)}
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for s in range(0, len(idx), batch):
            x = torch.from_numpy(arr[idx[s : s + batch]]).cuda()
            out.append(net(preprocess(x, gain, mean), taps).float().cpu().numpy())
    return np.concatenate(out)


def bn_layers(net):
    return sorted({k.split("__")[0] for k in net.p if k.endswith("__var")})


def bn_mismatch(net, acc) -> float:
    """Mean over batch-norm layers of median |log var ratio| + median |mean
    shift| in training standard deviations. 0 = activations match training."""
    import torch

    scores = []
    for n, (s1, s2, c) in acc.items():
        m = s1 / c
        v = (s2 / c - m * m).clamp_min(1e-8)
        tm, tv = net.g(n, "mean").float(), net.g(n, "var").float().clamp_min(1e-8)
        lv = torch.log(v / tv).abs().median()
        dm = ((m - tm).abs() / tv.sqrt()).median()
        scores.append((lv + dm).item())
    return float(np.mean(scores))


def evaluate(data_dir, workspace, net_npz, out_dir, calib=300, seed=0):
    from eyespy_spermatogenesis.staging import satinn
    from eyespy_spermatogenesis.staging.round1.evaluate import (
        cyc_dist,
        metrics,
        plot_confusion,
    )
    from eyespy_spermatogenesis.staging.round1.overlay import write_overview

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_table(data_dir)
    ev = [
        i
        for i, r in enumerate(rows)
        if r["recommended"] and r["stage"] in STAGES and not r["needs_check"]
    ]
    y = np.array([STAGES.index(rows[i]["stage"]) for i in ev])
    rng = np.random.default_rng(seed)
    cal = rng.choice(len(rows), min(calib, len(rows)), replace=False)  # labels unused
    net, _ = satinn.load(net_npz, "cuda")
    all_idx = np.arange(len(rows))
    grid, probs = [], {}
    for side in SIDES:
        arr = read_boxes(workspace, rows, side)
        for gain in GAINS:
            acc = {}
            run_net(net, arr, gain, cal, bn_acc=acc)
            score = bn_mismatch(net, acc)
            p12 = run_net(net, arr, gain, all_idx)
            key = f"{side}px_{gain}"
            probs[key] = p12.astype(np.float32)
            p = to_ours(p12[ev]).argmax(1)
            conf = p12[ev].max(1) >= CONF
            m = metrics(y, p)
            mc = metrics(y[conf], p[conf]) if conf.sum() else {"acc": None}
            grid.append(
                {
                    "side_px": side,
                    "contrast": gain,
                    "bn_mismatch": score,
                    "acc": m["acc"],
                    "balanced_acc": m["balanced_acc"],
                    "within1": m["within1"],
                    "coverage_conf80": float(conf.mean()),
                    "acc_conf80": mc["acc"],
                }
            )
            print(
                f"box {side:4d} contrast {gain!s:4}: mismatch {score:.3f} | "
                f"acc {m['acc']:.3f} bal {m['balanced_acc']:.3f} "
                f"w1 {m['within1']:.3f} | conf>=0.8 {conf.mean():.2f}",
                flush=True,
            )
        del arr
    best = min(grid, key=lambda g: g["bn_mismatch"])
    key = f"{best['side_px']}px_{best['contrast']}"
    p12 = probs[key][ev]
    pred = to_ours(p12).argmax(1)
    conf12 = p12.max(1)
    keep = conf12 >= CONF
    m_all = metrics(y, pred)
    m_conf = metrics(y[keep], pred[keep]) if keep.any() else None
    np.savez(
        out_dir / "probs.npz",
        tubule_id=np.array([r["tubule_id"] for r in rows]),
        **probs,
    )
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "tubule_id",
                "section",
                "paper_stage",
                "predicted",
                "confidence",
                "satinn_12class",
                "cyclic_distance",
            ]
        )
        for j, i in enumerate(ev):
            w.writerow(
                [
                    rows[i]["tubule_id"],
                    rows[i]["section"],
                    STAGES[y[j]],
                    STAGES[pred[j]],
                    f"{conf12[j]:.3f}",
                    satinn.SATINN_STAGES[int(p12[j].argmax())],
                    int(cyc_dist(y[j], pred[j])),
                ]
            )
    name = (
        f"SATINN {Path(net_npz).stem} "
        f"(box {best['side_px']} px, contrast {best['contrast']})"
    )
    plot_confusion(
        y,
        pred,
        out_dir / "confusion_loso.png",
        f"{name}\nno training on our labels (counts; colour = row fraction)",
    )
    groups = np.array([rows[i]["section"] for i in ev])
    m_all["per_section_acc"] = {
        s: float((pred[groups == s] == y[groups == s]).mean())
        for s in sorted(set(groups))
    }
    res = {
        "net": str(net_npz),
        "n_tubules": int(len(y)),
        "calibration_tubules": int(len(cal)),
        "selection": "lowest batch-norm mismatch on calibration tubules (no labels)",
        "chosen": best,
        "metrics": m_all,
        "metrics_conf80": m_conf,
        "coverage_conf80": float(keep.mean()),
        "grid": grid,
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    write_summary(out_dir, res, name)
    # Overview panels on the annotation page
    pred_of = {
        name: {rows[i]["tubule_id"]: STAGES[pred[j]] for j, i in enumerate(ev)},
        f"{name}, confidence >= {CONF:.2f} only": {
            rows[i]["tubule_id"]: STAGES[pred[j]] for j, i in enumerate(ev) if keep[j]
        },
    }
    models = [
        {
            "name": name,
            "overall": {k: m_all[k] for k in ("n", "acc", "balanced_acc", "within1")},
            "recall": m_all["recall"],
        }
    ]
    if m_conf:
        models.append(
            {
                "name": f"{name}, confidence >= {CONF:.2f} only",
                "overall": {
                    k: m_conf[k] for k in ("n", "acc", "balanced_acc", "within1")
                },
                "recall": m_conf["recall"],
            }
        )
    write_overview(workspace, out_dir.name, METHOD, rows, pred_of, models)
    return res


def write_summary(out_dir, res, name):
    m, mc, b = res["metrics"], res["metrics_conf80"], res["chosen"]
    L = [
        "# Staging round 2 (SATINN): pretrained, no training on our labels",
        "",
        f"{res['n_tubules']} accepted tubules with an unflagged paper stage "
        "(same set as round 1). Preprocessing chosen without labels "
        "(lowest batch-norm mismatch on "
        f"{res['calibration_tubules']} tubules): **box {b['side_px']} px, contrast "
        f"{b['contrast']}**. Model: {name}.",
        "",
        "| Metric | All tubules | Confidence >= 0.80 only |",
        "|---|---|---|",
    ]
    for k, lab in (
        ("acc", "Accuracy"),
        ("balanced_acc", "Balanced accuracy"),
        ("within1", "Within one stage"),
    ):
        L.append(
            f"| {lab} | {m[k]:.3f} | {mc[k]:.3f} |"
            if mc
            else f"| {lab} | {m[k]:.3f} | – |"
        )
    L.append(f"| Coverage | 1.000 | {res['coverage_conf80']:.3f} |")
    L += ["", "## Per stage", "", "| Stage | Recall |", "|---|---|"]
    L += [f"| {s} | {r:.2f} |" for s, r in m["recall"].items()]
    L += ["", "## Per section", "", "| Section | Accuracy |", "|---|---|"]
    L += [f"| {s} | {a:.2f} |" for s, a in m["per_section_acc"].items()]
    L += [
        "",
        "## Sensitivity (all settings; scores were not used to choose)",
        "",
        "| Box px | Contrast | BN mismatch | Accuracy | Balanced | Within one "
        "| Coverage >=0.8 |",
        "|---|---|---|---|---|---|---|",
    ]
    for g in sorted(res["grid"], key=lambda g: g["bn_mismatch"]):
        L.append(
            f"| {g['side_px']} | {g['contrast']} | {g['bn_mismatch']:.3f} "
            f"| {g['acc']:.3f} | {g['balanced_acc']:.3f} | {g['within1']:.3f} "
            f"| {g['coverage_conf80']:.2f} |"
        )
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", type=Path)
    p.add_argument("workspace", type=Path)
    p.add_argument("net_npz", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--calib", type=int, default=300)
    a = p.parse_args(argv)
    evaluate(a.data_dir, a.workspace, a.net_npz, a.out_dir, a.calib)


if __name__ == "__main__":
    sys.exit(main())
