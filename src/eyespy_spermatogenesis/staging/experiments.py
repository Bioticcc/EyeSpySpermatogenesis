"""Opt-in staging experiments on frozen embeddings (owner, 2026-11-11).

Usage::

    python -m eyespy_spermatogenesis.staging.experiments DATA_DIR EMB_DIR OUT_DIR \\
        --member h-optimus-0:s05_raw:mean --member uni2-h:s1_raw:inside \\
        [--balanced off on] [--weights equal nested] [--smooth off nested]

Nothing here changes rounds 1-3: it reads the same embeddings and writes only
to OUT_DIR. Each experiment is a switch whose "off" setting reproduces the
plain method, so any of them can be dropped again.

* **Members**: ``model:view:pool`` (model may be ``name_d4`` for embeddings made
  with ``embed --tta d4``). Each member is a cyclic ridge head (round 1) whose
  alpha is chosen per held-out section by inner leave-one-section-out.
* ``--balanced on``: rare stages weighted up (sample weight = 1 / stage count,
  normalised to mean 1) when fitting the head.
* ``--weights nested``: ensemble weights on a grid, chosen per held-out
  section from inner out-of-fold predictions of the training sections only;
  ``equal`` averages members (round 3).
* ``--smooth nested``: each tubule's (cos, sin) output is blended with the mean
  of its k nearest tubules in the same section (centres from source bounds):
  ``v' = (1 - lam) v + lam mean(v_neighbours)``. k and lam (lam = 0 = off) are
  chosen per held-out section the same nested way. Uses only predictions,
  never labels, of neighbours, so it is valid at test time. It assumes
  neighbouring tubules tend to share stages (synchronised zones); this is
  for the owner to confirm biologically.

Every variant is scored leave-one-section-out with a section bootstrap (95%
interval) of the difference from the first member, plain. Outputs:
``results.json``, ``summary.md``, ``predictions.npz``.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from eyespy_spermatogenesis.staging.round1.data import STAGES, read_table
from eyespy_spermatogenesis.staging.round1.evaluate import (
    GRID_ALPHA,
    Cyclic,
    K,
    features,
    metrics,
    research_only,
)

WEIGHT_GRID = (0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0)
SMOOTH_K = (4, 6, 8, 12)
SMOOTH_LAM = (0.0, 0.15, 0.3, 0.45, 0.6)


# ---------------------------------------------------------------- pieces
def stage_weights(y: np.ndarray) -> np.ndarray:
    """Per-sample weights 1 / stage count, normalised to mean 1."""
    counts = np.bincount(y, minlength=K).astype(float)
    w = 1.0 / counts[y]
    return w / w.mean()


def sample_weights(y, balanced: bool, sw=None):
    """Stage weighting and/or per-tubule weights ``sw``, mean 1 (None = off)."""
    if not balanced and sw is None:
        return None
    w = stage_weights(y) if balanced else np.ones(len(y))
    if sw is not None:
        w = w * sw
    return w / w.mean()


def fit_vectors(xtr, ytr, xte, alpha, balanced: bool, sw=None) -> np.ndarray:
    """Fit a cyclic head and return its raw (cos, sin) outputs on ``xte``.

    ``sw``: optional per-tubule weights of the training rows (EyeSpy
    experiments, 2026-11-13); None reproduces the 2026-11-11 heads."""
    c = Cyclic(alpha)
    t = np.c_[np.cos(c.ang[ytr]), np.sin(c.ang[ytr])]
    w = sample_weights(ytr, balanced, sw)
    kw = {"ridge__sample_weight": w} if w is not None else {}
    c.m.fit(xtr, t, **kw)
    return c.m.predict(xte)


def decode(v: np.ndarray) -> np.ndarray:
    a = np.arctan2(v[:, 1], v[:, 0]) % (2 * np.pi)
    return np.round(a / (2 * np.pi / K)).astype(int) % K


def oof_vectors(x, y, g, alpha, balanced, sw=None) -> np.ndarray:
    """Leave-one-section-out (cos, sin) outputs with a fixed alpha."""
    v = np.zeros((len(y), 2))
    for s in np.unique(g):
        te = g == s
        v[te] = fit_vectors(
            x[~te], y[~te], x[te], alpha, balanced, None if sw is None else sw[~te]
        )
    return v


def neighbours(centres: np.ndarray, groups: np.ndarray, k: int) -> list[np.ndarray]:
    """Indices of the k nearest tubules in the same section (self excluded)."""
    out: list[np.ndarray] = [np.empty(0, int)] * len(centres)
    for s in np.unique(groups):
        ii = np.nonzero(groups == s)[0]
        kk = min(k, len(ii) - 1)
        if kk < 1:
            continue
        _, nb = cKDTree(centres[ii]).query(centres[ii], kk + 1)
        for row, i in zip(nb, ii):
            out[i] = ii[row[1:]]
    return out


def smooth(v: np.ndarray, nbrs: list[np.ndarray], lam: float) -> np.ndarray:
    if lam == 0:
        return v
    m = np.array([v[n].mean(0) if len(n) else v[i] for i, n in enumerate(nbrs)])
    return (1 - lam) * v + lam * m


def weight_grid(n: int):
    """Convex weights for n members on a grid (n = 1, 2 or 3)."""
    if n == 1:
        return [(1.0,)]
    if n == 2:
        return [(w, 1 - w) for w in WEIGHT_GRID]
    return [
        (a, b, round(1 - a - b, 6))
        for a, b in product(WEIGHT_GRID, repeat=2)
        if a + b <= 1 + 1e-9
    ]


def centres_of(rows, idx) -> np.ndarray:
    b = np.array([json.loads(rows[i]["source_bounds"]) for i in idx], float)
    return np.c_[(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2]


# ---------------------------------------------------------------- core
class Member:
    """One cyclic head; nested alpha; outer and inner out-of-fold vectors."""

    def __init__(self, spec, x, y, g, balanced, sw=None):
        self.spec, self.balanced = spec, balanced
        secs = np.unique(g)
        self.outer = np.zeros((len(y), 2))
        self.inner: dict[str, np.ndarray] = {}  # outer section -> train OOF
        self.alpha: dict[str, float] = {}
        for s in secs:
            te = g == s
            xt, yt, gt = x[~te], y[~te], g[~te]
            swt = None if sw is None else sw[~te]
            best, best_v = None, None
            for a in GRID_ALPHA:
                v = oof_vectors(xt, yt, gt, a, balanced, swt)
                score = metrics(yt, decode(v))["balanced_acc"]
                if best is None or score > best[0]:
                    best, best_v = (score, a), v
            self.alpha[str(s)] = best[1]
            self.inner[str(s)] = best_v
            self.outer[te] = fit_vectors(xt, yt, x[te], best[1], balanced, swt)


def combine(members, y, g, nbrs_by_k, weights: str, smoothing: str):
    """Outer predictions for one variant, with per-section nested choices."""
    secs = np.unique(g)
    grid_w = weight_grid(len(members)) if weights == "nested" else None
    eq = tuple([1 / len(members)] * len(members))
    smooth_grid = (
        [(k, lam) for k in SMOOTH_K for lam in SMOOTH_LAM]
        if smoothing == "nested"
        else [(SMOOTH_K[0], 0.0)]
    )
    out, chosen = np.zeros((len(y), 2)), {}
    for s in secs:
        te = g == s
        tr = ~te
        inner = [m.inner[str(s)] for m in members]
        best = None
        for w in grid_w or [eq]:
            vw = sum(wi * vi for wi, vi in zip(w, inner))
            for k, lam in smooth_grid:
                vs = smooth_sub(vw, nbrs_by_k[k], tr, lam) if lam else vw
                sc = metrics(y[tr], decode(vs))["balanced_acc"]
                if best is None or sc > best[0] + 1e-12:
                    best = (sc, w, k, lam)
        _, w, k, lam = best
        chosen[str(s)] = {"weights": list(w), "k": k, "lam": lam}
        v = sum(wi * m.outer for wi, m in zip(w, members))
        v = smooth(v, nbrs_by_k[k], lam)
        out[te] = v[te]
    return out, chosen


def smooth_sub(v_sub, nbrs_full, mask, lam):
    """Smooth vectors defined on ``mask`` rows (training sections only).

    Neighbours are within one section, so a training tubule's neighbours are
    all in the training subset."""
    pos = -np.ones(len(mask), int)
    pos[mask] = np.arange(mask.sum())
    sub = [pos[nbrs_full[i]] for i in np.nonzero(mask)[0]]
    return smooth(v_sub, sub, lam)


def bootstrap_ci(y, g, p, ref, n=2000, seed=0):
    secs = np.unique(g)
    rng = np.random.default_rng(seed)
    da, db = [], []
    for _ in range(n):
        i = np.concatenate([np.nonzero(g == s)[0] for s in rng.choice(secs, len(secs))])
        da.append((p[i] == y[i]).mean() - (ref[i] == y[i]).mean())
        db.append(
            metrics(y[i], p[i])["balanced_acc"] - metrics(y[i], ref[i])["balanced_acc"]
        )
    return [float(np.percentile(da, q)) for q in (2.5, 97.5)], [
        float(np.percentile(db, q)) for q in (2.5, 97.5)
    ]


def run(data_dir, emb_dir, out_dir, specs, balanced, weights, smoothing, boot=2000):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_table(data_dir)
    idx = [
        i
        for i, r in enumerate(rows)
        if r["recommended"] and r["stage"] in STAGES and not r["needs_check"]
    ]
    y = np.array([STAGES.index(rows[i]["stage"]) for i in idx])
    g = np.array([rows[i]["section"] for i in idx])
    cen = centres_of(rows, idx)
    nbrs_by_k = {k: neighbours(cen, g, k) for k in SMOOTH_K}

    members = {}
    for spec in specs:
        model, view, how = spec.split(":")
        x = features(emb_dir, model, view, how, len(rows))[idx]
        for bal in balanced:
            members[(spec, bal)] = Member(spec, x, y, g, bal == "on")
            print(f"member {spec} balanced={bal} done", flush=True)

    variants = {}
    for bal in balanced:
        groups = [[s] for s in specs] + ([specs] if len(specs) > 1 else [])
        for grp in groups:
            ms = [members[(s, bal)] for s in grp]
            for w in weights if len(grp) > 1 else ["equal"]:
                for sm in smoothing:
                    name = (
                        " + ".join(grp)
                        + f" | balanced={bal}"
                        + (f" | weights={w}" if len(grp) > 1 else "")
                        + f" | smooth={sm}"
                    )
                    if name in variants:
                        continue
                    v, chosen = combine(ms, y, g, nbrs_by_k, w, sm)
                    variants[name] = {
                        "pred": decode(v),
                        "chosen": chosen,
                        "members": grp,
                        "balanced": bal,
                        "weights": w,
                        "smooth": sm,
                    }
                    print(f"variant {name} done", flush=True)

    ref_name = f"{specs[0]} | balanced={balanced[0]} | smooth={smoothing[0]}"
    ref = variants[ref_name]["pred"]
    res = {"n": int(len(y)), "reference": ref_name, "variants": {}}
    for name, v in variants.items():
        m = metrics(y, v["pred"])
        ci_a, ci_b = bootstrap_ci(y, g, v["pred"], ref, boot)
        models = [s.split(":")[0] for s in v["members"]]
        res["variants"][name] = {
            **{k: v[k] for k in ("members", "balanced", "weights", "smooth", "chosen")},
            **m,
            "vs_reference_acc_ci": ci_a,
            "vs_reference_bal_ci": ci_b,
            "research_only": bool(research_only(models)),
        }
    res["alpha"] = {f"{s} | balanced={b}": m.alpha for (s, b), m in members.items()}
    res["research_only"] = research_only([s.split(":")[0] for s in specs])
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    np.savez(
        out_dir / "predictions.npz",
        tubule_id=np.array([rows[i]["tubule_id"] for i in idx]),
        y=y,
        section=g,
        names=np.array(list(variants)),
        preds=np.stack([v["pred"] for v in variants.values()]),
    )
    write_summary(out_dir, res)
    return res


def write_summary(out_dir: Path, res: dict):
    L = ["# Staging experiments: leave-one-section-out", ""]
    for name, note in res["research_only"].items():
        L += [
            f"> **⚠ RESEARCH ONLY: NOT FOR RELEASE ({name}).** {note} Rows marked "
            "⚠ use it and are a proof of concept only (decision 025).",
            "",
        ]
    L += [
        f"{res['n']} tubules. Reference for the intervals: `{res['reference']}`. "
        "Intervals: section bootstrap, 95%. All choices (alpha, ensemble weights, "
        "smoothing) are made per held-out section from its training sections only.",
        "",
        "| Variant | Exact | Balanced | Within one | Δ exact (95% CI) "
        "| Δ balanced (95% CI) |",
        "|---|---|---|---|---|---|",
    ]
    order = sorted(res["variants"].items(), key=lambda kv: -kv[1]["acc"])
    for name, v in order:
        a, b = v["vs_reference_acc_ci"], v["vs_reference_bal_ci"]
        flag = "⚠ " if v["research_only"] else ""
        L.append(
            f"| {flag}{name} | {v['acc']:.3f} | {v['balanced_acc']:.3f} | "
            f"{v['within1']:.3f} | {a[0]:+.3f} to {a[1]:+.3f} | "
            f"{b[0]:+.3f} to {b[1]:+.3f} |"
        )
    L += ["", "## Per-stage recall", ""]
    L.append("| Variant | " + " | ".join(STAGES) + " |")
    L.append("|---" * (len(STAGES) + 1) + "|")
    for name, v in order:
        flag = "⚠ " if v["research_only"] else ""
        L.append(
            f"| {flag}{name} | "
            + " | ".join(f"{v['recall'].get(s, float('nan')):.2f}" for s in STAGES)
            + " |"
        )
    L += ["", "Per-section choices: `results.json` (`chosen`, `alpha`).", ""]
    (out_dir / "summary.md").write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", type=Path)
    p.add_argument("emb_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument(
        "--member",
        action="append",
        required=True,
        help="model:view:pool (repeat; first = reference)",
    )
    p.add_argument("--balanced", nargs="+", choices=("off", "on"), default=["off"])
    p.add_argument(
        "--weights", nargs="+", choices=("equal", "nested"), default=["equal"]
    )
    p.add_argument("--smooth", nargs="+", choices=("off", "nested"), default=["off"])
    p.add_argument("--boot", type=int, default=2000)
    a = p.parse_args(argv)
    if len(a.member) > 3:
        raise SystemExit("At most three members")
    run(
        a.data_dir,
        a.emb_dir,
        a.out_dir,
        a.member,
        a.balanced,
        a.weights,
        a.smooth,
        a.boot,
    )


if __name__ == "__main__":
    sys.exit(main())
