"""Add a model-review sample to the annotation page (mode "Model review").

Usage::

    python -m eyespy_spermatogenesis.staging.round1.review \
        DATA_DIR EMB_DIR RESULTS_DIR WORKSPACE [--per-stage 15] \
        [--models uni2-h h-optimus-0]

For the best screening configuration of each feature extractor in
``RESULTS_DIR/results.json`` it computes leave-one-section-out predictions
(regularization chosen by nested leave-one-section-out, as in ``evaluate``),
samples up to ``--per-stage`` tubules per paper stage spread across sections,
and stores them in ``WORKSPACE/labels.sqlite`` (table ``model_predictions``,
run name = ``RESULTS_DIR`` folder name). ``--models`` picks the two extractors
when the results hold more (``a+b`` = concatenated features). The two models
appear as "A" and "B", randomly assigned per tubule, in a shuffled order.
Re-running replaces that run's predictions; reviewers' answers are kept.
Restart the server afterwards.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from eyespy_spermatogenesis.staging.annotate.store import Store
from eyespy_spermatogenesis.staging.round1.data import STAGES, read_table
from eyespy_spermatogenesis.staging.round1.evaluate import (
    features,
    nested_loso_predict,
)


def model_predictions(data_dir, emb_dir, results, models=None):
    rows = read_table(data_dir)
    idx = [
        i
        for i, r in enumerate(rows)
        if r["recommended"] and r["stage"] in STAGES and not r["needs_check"]
    ]
    y = np.array([STAGES.index(rows[i]["stage"]) for i in idx])
    groups = np.array([rows[i]["section"] for i in idx])
    best = {}
    for r in results["screen"]:  # sorted best first
        if not models or r["model"] in models:
            best.setdefault(r["model"], r)
    if models and set(models) - set(best):
        raise SystemExit(f"Not in results: {sorted(set(models) - set(best))}")
    preds = {}
    for model, r in best.items():
        x = features(emb_dir, model, r["view"], r["pool"], len(rows))[idx]
        p, _ = nested_loso_predict(x, y, groups, r["head"])
        name = f"{model} ({r['view']}, {r['pool']}, {r['head']})"
        preds[name] = p
        print(f"{name}: accuracy {(p == y).mean():.3f}", flush=True)
    return rows, idx, y, groups, preds


def sample(y, groups, per_stage, rng):
    chosen = []
    for k in range(len(STAGES)):
        cand = np.nonzero(y == k)[0]
        if len(cand) <= per_stage:
            chosen += cand.tolist()
            continue
        # round-robin over sections so one section does not dominate
        by_sec = {}
        for i in rng.permutation(cand):
            by_sec.setdefault(groups[i], []).append(int(i))
        lists = list(by_sec.values())
        while len([c for c in chosen if y[c] == k]) < per_stage:
            for lst in lists:
                if lst and len([c for c in chosen if y[c] == k]) < per_stage:
                    chosen.append(lst.pop())
    return [int(c) for c in rng.permutation(chosen)]


def build(
    data_dir: Path,
    emb_dir: Path,
    results_dir: Path,
    workspace: Path,
    per_stage=15,
    seed=0,
    models=None,
):
    results = json.loads((results_dir / "results.json").read_text())
    rows, idx, y, groups, preds = model_predictions(data_dir, emb_dir, results, models)
    names = list(preds)
    if len(names) != 2:
        raise SystemExit(f"Expected two feature extractors, found {names}")
    rng = np.random.default_rng(seed)
    pick = sample(y, groups, per_stage, rng)
    out = []
    for order, j in enumerate(pick):
        tid = rows[idx[j]]["tubule_id"]
        pair = names if rng.integers(2) == 0 else names[::-1]
        for slot, model in zip("AB", pair):
            out.append(
                {
                    "tubule_id": tid,
                    "ord": order,
                    "slot": slot,
                    "model": model,
                    "value": STAGES[preds[model][j]],
                }
            )
    store = Store(workspace / "labels.sqlite")
    known = {t["tubule_id"] for t in store.tubules()}
    missing = {r["tubule_id"] for r in out} - known
    if missing:
        store.close()
        raise SystemExit(f"{len(missing)} tubules are not in the workspace")
    store.add_predictions(results_dir.name, out)
    store.close()
    print(f"{len(pick)} tubules -> review run {results_dir.name} in {workspace}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", type=Path)
    p.add_argument("emb_dir", type=Path)
    p.add_argument("results_dir", type=Path)
    p.add_argument("workspace", type=Path)
    p.add_argument("--per-stage", type=int, default=15)
    p.add_argument("--models", nargs=2, help="the two extractors to review")
    a = p.parse_args(argv)
    build(
        a.data_dir, a.emb_dir, a.results_dir, a.workspace, a.per_stage, models=a.models
    )


if __name__ == "__main__":
    sys.exit(main())
