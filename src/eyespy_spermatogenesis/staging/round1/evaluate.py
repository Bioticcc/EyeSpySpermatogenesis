"""Leave-one-section-out evaluation of classifiers on frozen embeddings.

Usage::

    python -m eyespy_spermatogenesis.staging.round1.evaluate DATA_DIR EMB_DIR OUT_DIR \
        [--models uni2-h h-optimus-0 uni2-h+h-optimus-0] [--title "Staging round 3"]

``--models`` limits the extractors (default: every folder in EMB_DIR); ``a+b``
concatenates two extractors' pooled features (same view and pooling).
Extractors listed in ``embed.RESEARCH_ONLY`` get a licence notice in the summary.

Uses accepted (unflagged) crops whose paper stage is set and not flagged for
checking. Each tubule is predicted by a model trained without its section
(= animal). Two heads on standardized, pooled patch embeddings:

* ``logreg``: multinomial logistic regression, class-balanced weights.
* ``cyclic``: ridge regression onto the stage's position on the cycle
  (cos, sin of 2*pi*k/11, XII wraps to I), decoded to the nearest stage.

Step 1 screens every (model, view, pooling, head) with fixed regularization.
Step 2 re-runs the best configurations with regularization chosen by nested
leave-one-section-out on the training sections only. A random stratified
5-fold split is reported as an optimistic reference (same sections in train
and test). Outputs: ``results.json``, ``summary.md``, ``predictions.csv`` and
confusion matrices for the best configuration.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eyespy_spermatogenesis.staging.round1.data import STAGES, read_table
from eyespy_spermatogenesis.staging.round1.embed import RESEARCH_ONLY

K = len(STAGES)
POOLINGS = ("mean", "inside")
SCREEN_C, SCREEN_ALPHA = 0.05, 100.0
GRID_C = (0.005, 0.02, 0.05, 0.2, 1.0)
GRID_ALPHA = (10.0, 100.0, 1000.0, 10000.0)


def cyc_dist(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % K
    return np.minimum(d, K - d)


def pool(npz, n_crops: int, how: str) -> np.ndarray:
    emb, crop, frac = npz["emb"].astype(np.float32), npz["crop"], npz["frac"]
    out = np.zeros((n_crops, emb.shape[1]), np.float32)
    for c in range(n_crops):
        sel = crop == c
        if how == "inside":
            inside = sel & (frac >= 0.25)
            sel = inside if inside.any() else sel
        out[c] = emb[sel].mean(0)
    return out


class Cyclic:
    """Ridge onto (cos, sin) of the cycle position; nearest stage on decode."""

    def __init__(self, alpha):
        self.m = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        self.ang = 2 * np.pi * np.arange(K) / K

    def fit(self, x, y):
        self.m.fit(x, np.c_[np.cos(self.ang[y]), np.sin(self.ang[y])])
        return self

    def predict(self, x):
        p = self.m.predict(x)
        a = np.arctan2(p[:, 1], p[:, 0]) % (2 * np.pi)
        return np.round(a / (2 * np.pi / K)).astype(int) % K


def make(head, reg, dim):
    if head == "cyclic":
        return Cyclic(reg)
    steps = [StandardScaler()]
    if dim > 512:
        steps.append(PCA(256, random_state=0))
    steps.append(
        LogisticRegression(C=reg, class_weight="balanced", max_iter=3000, tol=1e-3)
    )
    return make_pipeline(*steps)


def loso_predict(x, y, groups, head, reg):
    pred = np.empty_like(y)
    for g in np.unique(groups):
        te = groups == g
        pred[te] = make(head, reg, x.shape[1]).fit(x[~te], y[~te]).predict(x[te])
    return pred


def nested_loso_predict(x, y, groups, head):
    grid = GRID_ALPHA if head == "cyclic" else GRID_C
    pred, chosen = np.empty_like(y), {}
    for g in np.unique(groups):
        te = groups == g
        xt, yt, gt = x[~te], y[~te], groups[~te]
        scores = [
            metrics(yt, loso_predict(xt, yt, gt, head, r))["balanced_acc"] for r in grid
        ]
        best = grid[int(np.argmax(scores))]
        chosen[str(g)] = best
        pred[te] = make(head, best, x.shape[1]).fit(xt, yt).predict(x[te])
    return pred, chosen


def random_cv_predict(x, y, head, reg):
    pred = np.empty_like(y)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(x, y):
        pred[te] = make(head, reg, x.shape[1]).fit(x[tr], y[tr]).predict(x[te])
    return pred


def metrics(y, p) -> dict:
    present = np.unique(y)
    rec = {STAGES[k]: float((p[y == k] == k).mean()) for k in present}
    return {
        "n": int(len(y)),
        "acc": float((p == y).mean()),
        "balanced_acc": float(np.mean(list(rec.values()))),
        "macro_f1": float(f1_score(y, p, labels=present, average="macro")),
        "within1": float((cyc_dist(p, y) <= 1).mean()),
        "recall": rec,
    }


def plot_confusion(y, p, path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y, p, labels=range(K))
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.imshow(rown, cmap="Greens", vmin=0, vmax=1)
    for i in range(K):
        for j in range(K):
            if cm[i, j]:
                ax.text(
                    j,
                    i,
                    cm[i, j],
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if rown[i, j] > 0.5 else "black",
                )
    ax.set_xticks(range(K), STAGES, rotation=45)
    ax.set_yticks(range(K), STAGES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Paper stage")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def available_models(emb_dir: Path, models=None) -> list[str]:
    """Extractor names to evaluate; ``a+b`` = concatenated features."""
    have = sorted(p.name for p in emb_dir.iterdir() if (p / "embed.json").exists())
    if not models:
        return have
    for m in models:
        missing = [part for part in m.split("+") if part not in have]
        if missing:
            raise SystemExit(f"No embeddings for {missing} in {emb_dir}")
    return list(models)


def features(emb_dir: Path, model: str, view: str, how: str, n_rows: int):
    """Pooled features (n_rows x dim) for one extractor or an ``a+b`` concat."""
    return np.hstack(
        [
            pool(np.load(emb_dir / m / f"{view}.npz"), n_rows, how)
            for m in model.split("+")
        ]
    )


def research_only(models) -> dict[str, str]:
    return {
        part: RESEARCH_ONLY[part]
        for m in models
        for part in m.split("+")
        if part in RESEARCH_ONLY
    }


def evaluate(
    data_dir: Path,
    emb_dir: Path,
    out_dir: Path,
    top: int = 4,
    models=None,
    title: str = "Staging round 1",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_table(data_dir)
    idx = [
        i
        for i, r in enumerate(rows)
        if r["recommended"] and r["stage"] in STAGES and not r["needs_check"]
    ]
    y = np.array([STAGES.index(rows[i]["stage"]) for i in idx])
    groups = np.array([rows[i]["section"] for i in idx])
    sections = sorted(set(groups))
    # Training examples left for each stage when the section holding most of
    # that stage is held out: the real limit of leave-one-section-out.
    coverage = {}
    for k, s in enumerate(STAGES):
        per = Counter(groups[y == k])
        main = per.most_common(1)[0][0] if per else None
        coverage[s] = {
            "total": int((y == k).sum()),
            "sections": dict(per),
            "left_when_main_section_held_out": int((y == k).sum() - per.get(main, 0)),
        }

    models = available_models(emb_dir, models)
    configs = []
    for model in models:
        view_names = sorted(
            f.stem for f in (emb_dir / model.split("+")[0]).glob("*.npz")
        )
        for view in view_names:
            for how in POOLINGS if not view.startswith("fit") else ("mean",):
                x = features(emb_dir, model, view, how, len(rows))[idx]
                configs.append(((model, view, how), x))

    screen = []
    for (model, view, how), x in configs:
        for head, reg in (("logreg", SCREEN_C), ("cyclic", SCREEN_ALPHA)):
            m = metrics(y, loso_predict(x, y, groups, head, reg))
            screen.append(
                {"model": model, "view": view, "pool": how, "head": head, **m}
            )
            print(
                f"{model:24} {view:10} {how:6} {head:6} "
                f"bal {m['balanced_acc']:.3f} acc {m['acc']:.3f} "
                f"w1 {m['within1']:.3f}",
                flush=True,
            )
    screen.sort(key=lambda r: -r["balanced_acc"])

    xs = {k: x for k, x in configs}
    final = []
    for r in screen[:top]:
        x = xs[(r["model"], r["view"], r["pool"])]
        p, chosen = nested_loso_predict(x, y, groups, r["head"])
        m = metrics(y, p)
        m["per_section_acc"] = {
            s: float((p[groups == s] == y[groups == s]).mean()) for s in sections
        }
        reg = SCREEN_ALPHA if r["head"] == "cyclic" else SCREEN_C
        m["random_5fold"] = metrics(y, random_cv_predict(x, y, r["head"], reg))
        final.append(
            {k: r[k] for k in ("model", "view", "pool", "head")}
            | {"nested": m, "chosen_reg": chosen, "_pred": p}
        )
        print(
            f"nested {r['model']} {r['view']} {r['pool']} {r['head']}: "
            f"bal {m['balanced_acc']:.3f} acc {m['acc']:.3f}",
            flush=True,
        )
    final.sort(key=lambda r: -r["nested"]["balanced_acc"])
    best = final[0]
    p = best.pop("_pred")
    for r in final[1:]:
        r.pop("_pred")

    # Best model per extractor (for the comparison table)
    by_model = {}
    for r in screen:
        by_model.setdefault(r["model"], r)

    name = f"{best['model']} {best['view']} {best['pool']} {best['head']}"
    plot_confusion(
        y,
        p,
        out_dir / "confusion_loso.png",
        f"Leave-one-section-out, {name}\n(counts; colour = row fraction)",
    )
    with open(out_dir / "predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["tubule_id", "section", "paper_stage", "predicted", "cyclic_distance"]
        )
        for i, yy, pp in zip(idx, y, p):
            w.writerow(
                [
                    rows[i]["tubule_id"],
                    rows[i]["section"],
                    STAGES[yy],
                    STAGES[pp],
                    int(cyc_dist(yy, pp)),
                ]
            )
    majority = Counter(y).most_common(1)[0][1] / len(y)
    res = {
        "n_tubules": int(len(y)),
        "sections": sections,
        "stage_counts": {s: int((y == k).sum()) for k, s in enumerate(STAGES)},
        "coverage": coverage,
        "majority_class_acc": majority,
        "chance_balanced_acc": 1 / len(np.unique(y)),
        "screen": screen,
        "final": final,
        "best": {k: best[k] for k in ("model", "view", "pool", "head")},
        "models": models,
        "title": title,
        "research_only": research_only(models),
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=2))
    write_summary(out_dir, res, by_model)
    return res


def write_summary(out_dir: Path, res: dict, by_model: dict):
    b = res["final"][0]
    m = b["nested"]
    L = [f"# {res.get('title', 'Staging round 1')}: leave-one-section-out results", ""]
    for name, note in res.get("research_only", {}).items():
        L += [
            f"> **RESEARCH ONLY: NOT FOR RELEASE ({name}).** {note} Any result "
            "below that uses it is a proof of concept and must not ship in a "
            "public build.",
            "",
        ]
    L += [
        f"{res['n_tubules']} accepted tubules with an unflagged paper stage, "
        f"{len(res['sections'])} sections. Each tubule is predicted by a model "
        "that never saw its section. Labels are unchecked transcriptions.",
        "",
        f"Best: **{b['model']}, view {b['view']}, pooling {b['pool']}, "
        f"head {b['head']}** "
        "(regularization chosen by nested leave-one-section-out).",
        "",
        "| Metric | Leave-one-section-out | Random 5-fold (optimistic) |",
        "|---|---|---|",
    ]
    r5 = m["random_5fold"]
    for k, lab in (
        ("balanced_acc", "Balanced accuracy (mean per-stage recall)"),
        ("acc", "Accuracy"),
        ("macro_f1", "Macro F1"),
        ("within1", "Within one stage (cyclic)"),
    ):
        L.append(f"| {lab} | {m[k]:.3f} | {r5[k]:.3f} |")
    L += [
        f"| Majority-class accuracy | {res['majority_class_acc']:.3f} | |",
        f"| Chance balanced accuracy | {res['chance_balanced_acc']:.3f} | |",
        "",
        "## Per stage (leave-one-section-out)",
        "",
        "| Stage | n | Recall | Random 5-fold recall "
        "| Left in training when its main section is held out |",
        "|---|---|---|---|---|",
    ]
    for s in STAGES:
        c = res["coverage"][s]
        if not c["total"]:
            continue
        L.append(
            f"| {s} | {c['total']} | {m['recall'].get(s, float('nan')):.2f} | "
            f"{r5['recall'].get(s, float('nan')):.2f} | "
            f"{c['left_when_main_section_held_out']} |"
        )
    L += ["", "## Per held-out section", "", "| Section | Accuracy |", "|---|---|"]
    L += [f"| {s} | {a:.2f} |" for s, a in m["per_section_acc"].items()]
    L += [
        "",
        "## Best screening result per feature extractor (fixed regularization)",
        "",
        "| Extractor | View | Pooling | Head | Balanced acc | Accuracy | Within one |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in by_model.values():
        L.append(
            f"| {r['model']} | {r['view']} | {r['pool']} | {r['head']} | "
            f"{r['balanced_acc']:.3f} | {r['acc']:.3f} | {r['within1']:.3f} |"
        )
    L += [
        "",
        "Full screen and nested results: `results.json`. Confusion matrix: "
        "`confusion_loso.png`. Per-tubule predictions: `predictions.csv`.",
        "",
    ]
    (out_dir / "summary.md").write_text("\n".join(L))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", type=Path)
    p.add_argument("emb_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--top", type=int, default=4)
    p.add_argument("--models", nargs="+", help="extractors; a+b concatenates")
    p.add_argument("--title", default="Staging round 1")
    a = p.parse_args(argv)
    evaluate(a.data_dir, a.emb_dir, a.out_dir, a.top, a.models, a.title)


if __name__ == "__main__":
    sys.exit(main())
