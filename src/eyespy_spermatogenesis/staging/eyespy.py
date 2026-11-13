"""EyeSpy: the primary staging strategy (decision 027, owner 2026-11-13).

Frozen H-optimus-0 + DINOv2-L features and our own classifier trained on the
lab's labels from the annotation page (paper transcriptions and, from v2,
on-screen labels). Three members, equal weights (the best releasable model of
the 2026-11-11 experiments):

* H-optimus-0, 2x downsampled patches, mean pooled;
* H-optimus-0, same view, rotation/flip averaged (``embed --tta d4``);
* DINOv2-L, native patches, rotation/flip averaged.

Each member is a cyclic ridge head (round 1); the ensemble averages their
(cos, sin) outputs on the stage cycle.

Usage::

    python -m eyespy_spermatogenesis.staging.eyespy train --version 1 \\
        --source DATA_DIR:EMB_DIR [--source DATA_DIR2:EMB_DIR2] \\
        --workspace WORKSPACE [--checkpoints Model_Checkpoints] \\
        [--samples "#112_WR2_42d_2-1" "#92_WR2_47d_7-1" "#91_WR2_47d_6-1"]

    python -m eyespy_spermatogenesis.staging.eyespy index --workspace WORKSPACE

``train`` saves three models of one version and never overwrites one:

* ``EyeSpy_vN``: the plain model (headline numbers);
* ``EyeSpy_vN - Smoothed (k=.., λ=..)``: the same heads, with neighbour
  smoothing at prediction time (opt-in, decision 026);
* ``EyeSpy_vN - Class weighted``: comparison only (heads fitted with
  1 / stage-count weights), kept to see whether it ever helps.

Scores are leave-one-section-out, with every choice (ridge alpha, smoothing
k and strength) made per held-out section from its training sections only, as
on 2026-11-11. The saved checkpoint is then fitted on all sections, with alpha
and smoothing chosen by leave-one-section-out on all of them. The sample
overlays come from the leave-one-section-out models (never the checkpoint,
which has seen every section).

Files: ``CHECKPOINTS/registry.json`` (history of every model and its
settings), ``CHECKPOINTS/eyespy_vN/heads.npz`` (classifier weights; feature
extractor weights stay in the Hugging Face cache, revision recorded),
``CHECKPOINTS/eyespy_vN/predictions.npz`` (leave-one-section-out stages), and
``WORKSPACE/review/eyespy/`` (sample panels and ``index.json`` for the
annotation page's Overview > EyeSpy tab).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from eyespy_spermatogenesis.staging.experiments import (
    SMOOTH_K,
    SMOOTH_LAM,
    Member,
    bootstrap_ci,
    centres_of,
    combine,
    decode,
    neighbours,
    oof_vectors,
    sample_weights,
    smooth,
)
from eyespy_spermatogenesis.staging.round1.data import STAGES, read_table
from eyespy_spermatogenesis.staging.round1.evaluate import (
    GRID_ALPHA,
    K,
    cyc_dist,
    features,
    metrics,
)

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINTS = REPO / "Model_Checkpoints"
HF_CACHE = Path("/mnt/20TB_A/Adam/EyeSpySpermatogenesisRebooted/models/hf-cache")
DEFAULT_SAMPLES = ("#112_WR2_42d_2-1", "#92_WR2_47d_7-1", "#91_WR2_47d_6-1")

MEMBERS = (
    "h-optimus-0:s05_raw:mean",
    "h-optimus-0_d4:s05_raw:mean",
    "dinov2-l_d4:s1_raw:mean",
)
EXTRACTORS = {
    "h-optimus-0": ("hf-hub:bioptimus/H-optimus-0", "bioptimus/H-optimus-0"),
    "dinov2-l": (
        "hf-hub:timm/vit_large_patch14_dinov2.lvd142m",
        "timm/vit_large_patch14_dinov2.lvd142m",
    ),
}
SMOOTH_NOTE = (
    "k and λ chosen per held-out section from its training sections (nested), "
    f"over k in {list(SMOOTH_K)} and λ in {list(SMOOTH_LAM)}"
)
STRATEGY = (
    "EyeSpy: frozen H-optimus-0 + DINOv2-L features (Apache-2.0, releasable), "
    "our cyclic ridge classifier trained on the lab's labels; three members "
    "(H-optimus-0, H-optimus-0 rotation/flip averaged, DINOv2-L rotation/flip "
    "averaged), equal-weight mean of their (cos, sin) stage-cycle outputs."
)


# ---------------------------------------------------------------- data
class Dataset:
    """Evaluated tubules from one or more (data_dir, emb_dir[, options]) sources.

    Options (experiments, owner 2026-11-13; default off): ``shift`` moves every
    label of the source by that many stages on the cycle (-1 = one stage
    earlier; II-III counts as one stage), ``weight`` multiplies its tubules'
    weight in training (others 1). Unshifted labels stay in ``y_orig``."""

    def __init__(self, sources):
        self.sources, xs, rows_all = [], {m: [] for m in MEMBERS}, []
        shifts, weights = [], []
        for src in sources:
            data_dir, emb_dir = src[0], src[1]
            opts = src[2] if len(src) > 2 else {}
            shift, weight = int(opts.get("shift", 0)), float(opts.get("weight", 1))
            rows = read_table(data_dir)
            info = json.loads((data_dir / "data.json").read_text())
            idx = [
                i
                for i, r in enumerate(rows)
                if r["recommended"] and r["stage"] in STAGES and not r["needs_check"]
            ]
            for spec in MEMBERS:
                model, view, how = spec.split(":")
                xs[spec].append(features(emb_dir, model, view, how, len(rows))[idx])
            origin = info.get("origin", "paper")
            sel = [dict(rows[i], origin=origin) for i in idx]
            rows_all += sel
            shifts += [shift] * len(sel)
            weights += [weight] * len(sel)
            self.sources.append(
                {
                    "data_dir": str(data_dir),
                    "emb_dir": str(emb_dir),
                    "origin": origin,
                    "annotator": info.get("annotator", "Yeongseok"),
                    "labels_sha256": sha256(data_dir / "labels.csv"),
                    "n_evaluated": len(sel),
                    "sections": sorted({r["section"] for r in sel}),
                    "label_shift": shift,
                    "weight": weight,
                }
            )
        ids = [r["tubule_id"] for r in rows_all]
        dup = [t for t, n in Counter(ids).items() if n > 1]
        if dup:
            raise SystemExit(
                f"{len(dup)} tubules are labelled in more than one source "
                f"(e.g. {dup[0]}); decide which label wins before training."
            )
        self.rows = rows_all
        self.x = {m: np.vstack(v) for m, v in xs.items()}
        self.y_orig = np.array([STAGES.index(r["stage"]) for r in rows_all])
        self.y = (self.y_orig + np.array(shifts, int)) % K
        w = np.array(weights, float)
        self.w = None if np.all(w == 1) else w
        self.g = np.array([r["section"] for r in rows_all])
        self.origin = np.array([r["origin"] for r in rows_all])
        self.ids = np.array(ids)
        self.centres = centres_of(rows_all, range(len(rows_all)))

    def summary(self) -> dict:
        return {
            "sources": self.sources,
            "n": int(len(self.y)),
            "per_stage": {STAGES[k]: int((self.y == k).sum()) for k in range(K)},
            "per_section": dict(sorted(Counter(self.g.tolist()).items())),
            "rule": "accepted (unflagged) crops with a stage, readings not "
            "flagged to check; 'can't stage' and unlabelled tubules excluded",
        }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- heads
def fit_head(x, y, alpha, balanced, sw=None) -> dict:
    """Standardize + ridge onto (cos, sin); returned as plain arrays."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler().fit(x)
    ang = 2 * np.pi * np.arange(K) / K
    t = np.c_[np.cos(ang[y]), np.sin(ang[y])]
    w = sample_weights(y, balanced, sw)
    r = Ridge(alpha=alpha).fit(sc.transform(x), t, sample_weight=w)
    return {
        "mean": sc.mean_,
        "scale": sc.scale_,
        "coef": r.coef_,
        "intercept": r.intercept_,
    }


def head_vectors(head: dict, x: np.ndarray) -> np.ndarray:
    return ((x - head["mean"]) / head["scale"]) @ head["coef"].T + head["intercept"]


def choose_alpha(x, y, g, balanced, sw=None) -> float:
    """Alpha with the best leave-one-section-out balanced accuracy."""
    scores = [
        metrics(y, decode(oof_vectors(x, y, g, a, balanced, sw)))["balanced_acc"]
        for a in GRID_ALPHA
    ]
    return float(GRID_ALPHA[int(np.argmax(scores))])


def choose_smoothing(v, y, g, centres) -> tuple[int, float]:
    """(k, lam) with the best balanced accuracy on out-of-fold vectors."""
    best = None
    for k in SMOOTH_K:
        nb = neighbours(centres, g, k)
        for lam in SMOOTH_LAM:
            sc = metrics(y, decode(smooth(v, nb, lam)))["balanced_acc"]
            if best is None or sc > best[0] + 1e-12:
                best = (sc, k, lam)
    return best[1], best[2]


class EyeSpyModel:
    """A saved checkpoint: one head per member, optional smoothing.

    ``predict(feats, centres, sections)``: ``feats`` maps each member spec to
    its pooled features (n x dim), ``centres`` are tubule centres in source
    pixels, ``sections`` the section of each tubule (smoothing never crosses
    sections). Smoothing uses neighbours' predictions, never labels."""

    def __init__(self, heads: list[dict], members, k: int = 0, lam: float = 0.0):
        self.heads, self.members, self.k, self.lam = heads, list(members), k, lam

    @classmethod
    def load(cls, path: Path, k: int = 0, lam: float = 0.0) -> EyeSpyModel:
        z = np.load(path)
        members = [str(m) for m in z["members"]]
        heads = [
            {f: z[f"m{i}_{f}"] for f in ("mean", "scale", "coef", "intercept")}
            for i in range(len(members))
        ]
        return cls(heads, members, k, lam)

    def save(self, path: Path, **meta):
        arrays = {"members": np.array(self.members)}
        for i, h in enumerate(self.heads):
            arrays.update({f"m{i}_{f}": v for f, v in h.items()})
        np.savez(path, **arrays, **{k: np.array(v) for k, v in meta.items()})

    def vectors(self, feats, centres=None, sections=None) -> np.ndarray:
        v = np.mean(
            [head_vectors(h, feats[m]) for h, m in zip(self.heads, self.members)], 0
        )
        if self.lam:
            v = smooth(
                v,
                neighbours(np.asarray(centres), np.asarray(sections), self.k),
                self.lam,
            )
        return v

    def predict(self, feats, centres=None, sections=None) -> np.ndarray:
        return decode(self.vectors(feats, centres, sections))


# ---------------------------------------------------------------- training
def evaluate_variant(ds: Dataset, balanced: bool, smoothing: str, members=None):
    """Leave-one-section-out stages with nested choices (as 2026-11-11)."""
    if members is None:
        members = [Member(s, ds.x[s], ds.y, ds.g, balanced, ds.w) for s in MEMBERS]
    nbrs = {k: neighbours(ds.centres, ds.g, k) for k in SMOOTH_K}
    v, chosen = combine(members, ds.y, ds.g, nbrs, "equal", smoothing)
    return decode(v), chosen, members


def fit_final(ds: Dataset, balanced: bool) -> tuple[EyeSpyModel, dict, np.ndarray]:
    """Heads on every section; alpha per member by leave-one-section-out."""
    heads, alphas, oof = [], {}, []
    for spec in MEMBERS:
        a = choose_alpha(ds.x[spec], ds.y, ds.g, balanced, ds.w)
        alphas[spec] = a
        heads.append(fit_head(ds.x[spec], ds.y, a, balanced, ds.w))
        oof.append(oof_vectors(ds.x[spec], ds.y, ds.g, a, balanced, ds.w))
    return EyeSpyModel(heads, MEMBERS), alphas, np.mean(oof, 0)


def scores(ds: Dataset, pred: np.ndarray, samples) -> dict:
    out = {"all": metrics(ds.y, pred)}
    for origin in ("paper", "web"):
        m = ds.origin == origin
        out[f"{origin}_sections"] = metrics(ds.y[m], pred[m]) if m.any() else None
    if (ds.y != ds.y_orig).any():
        m = ds.origin == "web"
        out["web_sections_original_labels"] = (
            metrics(ds.y_orig[m], pred[m]) if m.any() else None
        )
    out["per_section"] = {
        s: section_score(ds.y[ds.g == s], pred[ds.g == s]) for s in np.unique(ds.g)
    }
    out["samples"] = {s: out["per_section"].get(s) for s in samples}
    return out


def section_score(y, p) -> dict:
    d = cyc_dist(p, y)
    return {
        "n": int(len(y)),
        "exact": float((d == 0).mean()),
        "within1": float((d <= 1).mean()),
    }


def git_commit() -> str:
    try:
        rev = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return rev + ("+uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hf_revision(repo_id: str) -> str | None:
    ref = HF_CACHE / f"models--{repo_id.replace('/', '--')}" / "refs" / "main"
    return ref.read_text().strip() if ref.exists() else None


def extractor_info() -> list[dict]:
    out = []
    for spec in MEMBERS:
        model, view, how = spec.split(":")
        base, tta = (model.split("_") + ["none"])[:2]
        timm_id, repo_id = EXTRACTORS[base]
        out.append(
            {
                "member": spec,
                "extractor": base,
                "timm_id": timm_id,
                "hf_revision": hf_revision(repo_id),
                "licence": "Apache-2.0",
                "view": view,
                "pooling": how,
                "rotation_flip_averaging": tta == "d4",
            }
        )
    return out


def load_registry(ckpt: Path) -> dict:
    p = ckpt / "registry.json"
    if p.exists():
        return json.loads(p.read_text())
    return {
        "schema": 1,
        "about": "History of every saved EyeSpy staging model and its settings. "
        "Entries are appended, never overwritten. Scores are leave-one-section-"
        "out; 'paper_sections' compares versions on the same tubules.",
        "strategy": STRATEGY,
        "samples": list(DEFAULT_SAMPLES),
        "models": [],
    }


def previous_primary(reg: dict, version: int) -> dict | None:
    prev = [
        m
        for m in reg["models"]
        if m["variant"] == "plain"
        and m["version"] < version
        and m.get("role") != "experiment"
    ]
    return max(prev, key=lambda m: m["version"]) if prev else None


def compare_to(ds: Dataset, pred, ckpt: Path, entry: dict | None, variant: str):
    """Paired section-bootstrap interval vs the previous version's same variant
    on the tubules both were scored on (paper sections)."""
    if not entry:
        return None
    z = np.load(ckpt / entry["folder"] / "predictions.npz")
    names = [str(n) for n in z["variants"]]
    if variant not in names:
        return None
    prev = dict(zip(z["tubule_id"].tolist(), z["preds"][names.index(variant)]))
    m = np.array([t in prev for t in ds.ids])
    if not m.any():
        return None
    ref = np.array([prev[t] for t in ds.ids[m]])
    ca, cb = bootstrap_ci(ds.y[m], ds.g[m], pred[m], ref)
    return {
        "against": f"{entry['name'].split(' - ')[0]} ({variant})",
        "n": int(m.sum()),
        "exact": float((pred[m] == ds.y[m]).mean()),
        "exact_reference": float((ref == ds.y[m]).mean()),
        "delta_exact_ci95": ca,
        "delta_balanced_ci95": cb,
    }


def train(
    version: int,
    sources,
    workspace: Path,
    ckpt: Path = DEFAULT_CHECKPOINTS,
    samples=None,
    note: str = "",
    replace: bool = False,
    experiment: str | None = None,
):
    """Save one version (plain, smoothed, class weighted), or with
    ``experiment`` (a short slug, e.g. ``web-shift-1``) a single plain model
    tagged as an experiment, named ``EyeSpy_vN - Experiment: <slug>``."""
    ckpt.mkdir(parents=True, exist_ok=True)
    reg = load_registry(ckpt)
    if samples:
        reg["samples"] = list(samples)
    samples = reg["samples"]
    base = f"EyeSpy_v{version}"
    folder = f"eyespy_v{version}"
    if experiment:
        base = f"EyeSpy_v{version} - Experiment: {experiment}"
        folder = f"eyespy_v{version}_{experiment}"

    def mine(m):
        if experiment:
            return m["name"] == base
        return m["version"] == version and m.get("role") != "experiment"

    if any(mine(m) for m in reg["models"]):
        if not replace:
            raise SystemExit(f"{base} is already in the registry; pick a new name.")
        reg["models"] = [m for m in reg["models"] if not mine(m)]
    (ckpt / folder).mkdir(exist_ok=True)

    ds = Dataset(sources)
    missing = [s for s in samples if s not in set(ds.g)]
    if missing:
        raise SystemExit(f"Sample sections without labels in the sources: {missing}")
    print(f"{base}: {len(ds.y)} tubules, {len(set(ds.g))} sections", flush=True)

    # Leave-one-section-out scores (nested choices, as 2026-11-11).
    p_plain, ch_plain, mem = evaluate_variant(ds, False, "off")
    print("plain evaluated", flush=True)
    alpha_nested = {m.spec: m.alpha for m in mem}
    model, alphas, oof = fit_final(ds, False)
    model.save(ckpt / folder / "heads.npz", alpha=[alphas[m] for m in MEMBERS])
    if not experiment:
        p_smooth, ch_smooth, _ = evaluate_variant(ds, False, "nested", mem)
        print("smoothed evaluated", flush=True)
        p_bal, ch_bal, mem_b = evaluate_variant(ds, True, "off")
        print("class weighted evaluated", flush=True)
        alpha_nested_b = {m.spec: m.alpha for m in mem_b}
        k, lam = choose_smoothing(oof, ds.y, ds.g, ds.centres)
        model_b, alphas_b, _ = fit_final(ds, True)
        model_b.save(
            ckpt / folder / "heads_class_weighted.npz",
            alpha=[alphas_b[m] for m in MEMBERS],
        )
        print(f"checkpoints saved; smoothing k={k}, lam={lam}", flush=True)

    variants = [
        (
            "plain",
            base,
            p_plain,
            ch_plain,
            "experiment" if experiment else "primary",
            None,
            "heads.npz",
            {"on": False},
            False,
            alphas,
            alpha_nested,
        )
    ]
    if not experiment:
        variants += [
            (
                "smoothed",
                f"{base} - Smoothed (k={k}, λ={lam:.2f})",
                p_smooth,
                ch_smooth,
                "settings variant",
                base,
                "heads.npz",
                {"on": True, "k": k, "lam": lam, "scored_with": SMOOTH_NOTE},
                False,
                alphas,
                alpha_nested,
            ),
            (
                "class_weighted",
                f"{base} - Class weighted",
                p_bal,
                ch_bal,
                "comparison",
                base,
                "heads_class_weighted.npz",
                {"on": False},
                True,
                alphas_b,
                alpha_nested_b,
            ),
        ]
    np.savez(
        ckpt / folder / "predictions.npz",
        tubule_id=ds.ids,
        y=ds.y,
        y_orig=ds.y_orig,
        section=ds.g,
        origin=ds.origin,
        variants=np.array([v[0] for v in variants]),
        names=np.array([v[1] for v in variants]),
        preds=np.stack([v[2] for v in variants]),
    )

    prev = previous_primary(reg, version)
    created = datetime.now().astimezone().isoformat(timespec="seconds")
    commit = git_commit()
    entries = []
    for (
        variant,
        name,
        pred,
        chosen,
        role,
        parent,
        heads,
        sm,
        bal,
        alph,
        alph_n,
    ) in variants:
        sc = scores(ds, pred, samples)
        ci_parent = None
        if parent:
            ca, cb = bootstrap_ci(ds.y, ds.g, pred, p_plain)
            ci_parent = {"delta_exact_ci95": ca, "delta_balanced_ci95": cb}
        entries.append(
            {
                "name": name,
                "version": version,
                "variant": variant,
                "experiment": experiment,
                "role": role,
                "parent": parent,
                "created": created,
                "code_commit": commit,
                "folder": folder,
                "checkpoint": f"{folder}/{heads}",
                "strategy": {
                    "name": "EyeSpy",
                    "description": STRATEGY,
                    "extractors": extractor_info(),
                    "head": "StandardScaler + ridge onto (cos, sin) of the "
                    "stage's angle on the 11-stage cycle; nearest stage",
                    "ensemble": "equal weights",
                },
                "settings": {
                    "smoothing": sm,
                    "class_weighting": bal,
                    "ensemble_weights": "equal",
                    "alpha_checkpoint": alph,
                    "alpha_scored_per_section": alph_n,
                    "smoothing_scored_per_section": (
                        {s: [c["k"], c["lam"]] for s, c in chosen.items()}
                        if sm["on"]
                        else None
                    ),
                },
                "training_data": ds.summary(),
                "evaluation": {
                    "method": "leave-one-section-out (each tubule predicted by a "
                    "model that never saw its section); 95% intervals by "
                    "section bootstrap, 2000 resamples",
                    **sc,
                    "vs_parent": ci_parent,
                    "vs_previous_version": compare_to(ds, pred, ckpt, prev, variant),
                },
                "note": note,
            }
        )
    reg["models"] += entries
    (ckpt / "registry.json").write_text(json.dumps(reg, indent=1, ensure_ascii=False))
    write_panels(ds, workspace, samples, folder, variants)
    write_index(workspace, ckpt)
    print(f"-> {ckpt / 'registry.json'}", flush=True)
    return entries


# ---------------------------------------------------------------- display
def slug(section: str) -> str:
    return section.lstrip("#")


def write_panels(ds: Dataset, workspace: Path, samples, folder, variants):
    """Ground truth + each variant's stage and agreement panels per sample."""
    import tifffile
    from PIL import Image

    from eyespy_spermatogenesis.staging.round1.data import run_dir
    from eyespy_spermatogenesis.staging.round1.overlay import (
        AGREE,
        panel,
        short,
        stage_colours,
    )

    out = workspace / "review" / "eyespy"
    colours = stage_colours()
    for sec in samples:
        ii = np.nonzero(ds.g == sec)[0]
        rd = run_dir(workspace, ds.rows[ii[0]]["source_folder"])
        overview = np.asarray(Image.open(rd / "overview.png").convert("RGB"))
        labels = tifffile.imread(rd / "overview_labels.tif")
        lab = [ds.rows[i]["label"] for i in ii]
        truth = out / "truth" / f"{slug(sec)}.jpg"
        truth.parent.mkdir(parents=True, exist_ok=True)
        if not truth.exists():
            panel(
                overview,
                labels,
                {lb: colours[STAGES[ds.y[i]]] for lb, i in zip(lab, ii)},
                {lb: short(STAGES[ds.y[i]]) for lb, i in zip(lab, ii)},
            ).save(truth, quality=85)
        for variant, _name, pred, *_ in variants:
            d = out / folder / variant
            d.mkdir(parents=True, exist_ok=True)
            panel(
                overview,
                labels,
                {lb: colours[STAGES[pred[i]]] for lb, i in zip(lab, ii)},
                {lb: short(STAGES[pred[i]]) for lb, i in zip(lab, ii)},
            ).save(d / f"{slug(sec)}_stage.jpg", quality=85)
            panel(
                overview,
                labels,
                {
                    lb: AGREE[min(int(cyc_dist(pred[i], ds.y[i])), 2)]
                    for lb, i in zip(lab, ii)
                },
                {},
            ).save(d / f"{slug(sec)}_agree.jpg", quality=85)
        print(f"{sec}: panels done", flush=True)


def write_index(workspace: Path, ckpt: Path = DEFAULT_CHECKPOINTS):
    """``WORKSPACE/review/eyespy/index.json`` from the registry (for the page)."""
    from eyespy_spermatogenesis.staging.annotate.clickmaps import (
        tubule_index,
        write_maps,
    )
    from eyespy_spermatogenesis.staging.round1.overlay import (
        AGREE,
        GREY,
        hexc,
        stage_colours,
    )

    reg = load_registry(ckpt)
    out = workspace / "review" / "eyespy"
    out.mkdir(parents=True, exist_ok=True)
    # Per-tubule stages for the page's click-through grid, keyed by candidate
    # label within each sample section.
    where = tubule_index(workspace)
    npz: dict[str, dict] = {}

    def stages_of(folder, variant=None, truth=False):
        if folder not in npz:
            z = np.load(ckpt / folder / "predictions.npz")
            npz[folder] = {k: z[k] for k in z.files}
        z = npz[folder]
        vals = (
            z.get("y_orig", z["y"])
            if truth
            else z["preds"][[str(v) for v in z["variants"]].index(variant)]
        )
        out_: dict[str, dict] = {}
        for t, v in zip(z["tubule_id"].tolist(), vals):
            sec, _f, lab = where[t]
            out_.setdefault(sec, {})[lab] = STAGES[int(v)]
        return out_

    samples = []
    for sec in reg["samples"]:
        truth = truth_info(reg, sec)
        src = next(
            (m for m in reg["models"] if sec in m["evaluation"]["per_section"]), None
        )
        samples.append(
            {
                "section": sec,
                "truth": f"truth/{slug(sec)}.jpg",
                "truth_label": truth["label"],
                "stages": truth["stages"],
                "truth_stages": (
                    stages_of(src["folder"], truth=True).get(sec, {}) if src else {}
                ),
            }
        )
    models = []
    for m in reg["models"]:
        ev = m["evaluation"]
        prefix = f"{m['folder']}/{m['variant']}"
        st = stages_of(m["folder"], m["variant"])
        models.append(
            {
                k: m[k]
                for k in (
                    "name",
                    "version",
                    "variant",
                    "role",
                    "parent",
                    "created",
                    "code_commit",
                    "checkpoint",
                    "note",
                )
            }
            | {
                "settings": m["settings"],
                "training": {
                    "n": m["training_data"]["n"],
                    "per_stage": m["training_data"]["per_stage"],
                    "sources": [
                        {
                            k: s[k]
                            for k in ("origin", "annotator", "n_evaluated", "sections")
                        }
                        for s in m["training_data"]["sources"]
                    ],
                },
                "all": pick(ev["all"]),
                "paper_sections": pick(ev["paper_sections"]),
                "web_sections": pick(ev["web_sections"]),
                "web_sections_original_labels": pick(
                    ev.get("web_sections_original_labels")
                ),
                "experiment": m.get("experiment"),
                "vs_parent": ev["vs_parent"],
                "vs_previous_version": ev["vs_previous_version"],
                "panels": {
                    s["section"]: {
                        "stage": f"{prefix}/{slug(s['section'])}_stage.jpg",
                        "agree": f"{prefix}/{slug(s['section'])}_agree.jpg",
                        "stages": st.get(s["section"], {}),
                        **(ev["per_section"].get(s["section"]) or {}),
                    }
                    for s in samples
                },
            }
        )
    (out / "index.json").write_text(
        json.dumps(
            {
                "strategy": reg["strategy"],
                "checkpoints": str(ckpt),
                "samples": samples,
                "models": models,
                "colours": {s: hexc(c) for s, c in stage_colours().items()},
                "agree_colours": {
                    "exact": hexc(AGREE[0]),
                    "one stage off": hexc(AGREE[1]),
                    "two or more off": hexc(AGREE[2]),
                    "uncoloured: no ground truth, not scored": hexc(GREY),
                },
            },
            indent=1,
            ensure_ascii=False,
        )
    )
    write_maps(workspace)


def pick(m):
    if not m:
        return None
    return {k: m[k] for k in ("n", "acc", "balanced_acc", "within1", "recall")}


def truth_info(reg: dict, section: str) -> dict:
    """Ground-truth source and main stages of a sample, from its newest model."""
    for m in reversed(reg["models"]):
        for s in m["training_data"]["sources"]:
            if section in s["sections"]:
                with open(Path(s["data_dir"]) / "labels.csv", newline="") as f:
                    rows = list(csv.DictReader(f))
                st = Counter(
                    r["stage"]
                    for r in rows
                    if r["section"] == section and r["stage"] in STAGES
                )
                n = sum(st.values())
                main = [x for x in STAGES if st[x] >= max(3, 0.05 * n)]
                who = s["annotator"]
                kind = "paper sheet" if s["origin"] == "paper" else "on-screen labels"
                return {"label": f"Ground truth: {who}, {kind}", "stages": main}
    return {"label": "Ground truth", "stages": []}


# ---------------------------------------------------------------- review run
def review_run(
    workspace: Path,
    names: list[str],
    ckpt: Path = DEFAULT_CHECKPOINTS,
    run: str | None = None,
    per_stage: int = 15,
    seed: int = 0,
) -> str:
    """Blind A/B review run on the annotation page for two saved models.

    Samples the paper-labelled tubules exactly as ``round1.review`` does (same
    seed, same order), so it holds the same tubules as the earlier review runs
    and reviewers' answers carry over. Predictions are the models'
    leave-one-section-out stages. The page uses the newest run."""
    from eyespy_spermatogenesis.staging.annotate.store import Store
    from eyespy_spermatogenesis.staging.round1.review import sample

    reg = load_registry(ckpt)
    by_name = {m["name"]: m for m in reg["models"]}
    missing = [n for n in names if n not in by_name]
    if len(names) != 2 or missing:
        raise SystemExit(f"Need two saved model names; not found: {missing}")
    preds = {}
    for n in names:
        m = by_name[n]
        z = np.load(ckpt / m["folder"] / "predictions.npz")
        v = [str(x) for x in z["variants"]].index(m["variant"])
        preds[n] = dict(zip(z["tubule_id"].tolist(), z["preds"][v]))
    # Paper tubules in round-1 order (the first source of every version).
    first = next(m for m in reg["models"] if m["version"] == 1)
    z = np.load(ckpt / first["folder"] / "predictions.npz")
    paper = z["origin"] == "paper"
    ids, y, groups = z["tubule_id"][paper], z["y"][paper], z["section"][paper]
    rng = np.random.default_rng(seed)
    pick = sample(y, groups, per_stage, rng)
    out = []
    for order, j in enumerate(pick):
        tid = str(ids[j])
        pair = names if rng.integers(2) == 0 else names[::-1]
        for slot, n in zip("AB", pair):
            out.append(
                {
                    "tubule_id": tid,
                    "ord": order,
                    "slot": slot,
                    "model": n,
                    "value": STAGES[int(preds[n][tid])],
                }
            )
    run = run or "eyespy-" + "-vs-".join(
        n.replace("EyeSpy_", "").replace(" ", "") for n in names
    )
    store = Store(workspace / "labels.sqlite")
    known = {t["tubule_id"] for t in store.tubules()}
    if {r["tubule_id"] for r in out} - known:
        store.close()
        raise SystemExit("Some sampled tubules are not in the workspace")
    store.add_predictions(run, out)
    store.close()
    print(f"{len(pick)} tubules -> review run {run}")
    return run


# ---------------------------------------------------------------- CLI
def parse_source(s: str):
    """``DATA_DIR:EMB_DIR[:shift=-1][:weight=3]``."""
    d, e, *opts = s.split(":")
    o = dict(x.split("=", 1) for x in opts)
    bad = set(o) - {"shift", "weight"}
    if bad:
        raise argparse.ArgumentTypeError(f"Unknown source options {sorted(bad)}")
    return Path(d), Path(e), o


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="evaluate and save one EyeSpy version")
    t.add_argument("--version", type=int, required=True)
    t.add_argument(
        "--source",
        action="append",
        required=True,
        type=parse_source,
        help="DATA_DIR:EMB_DIR[:shift=N][:weight=W] (repeat)",
    )
    t.add_argument("--workspace", type=Path, required=True)
    t.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    t.add_argument(
        "--samples", nargs="+", help="sample sections (kept in the registry)"
    )
    t.add_argument("--note", default="")
    t.add_argument(
        "--experiment",
        help="save one plain model tagged as an experiment under this slug",
    )
    t.add_argument(
        "--replace",
        action="store_true",
        help="replace this version's entries (only for a failed run)",
    )
    i = sub.add_parser("index", help="rebuild the page's index.json from the registry")
    i.add_argument("--workspace", type=Path, required=True)
    i.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    r = sub.add_parser("review", help="blind A/B review run of two saved models")
    r.add_argument("--workspace", type=Path, required=True)
    r.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    r.add_argument("--models", nargs=2, required=True, metavar="NAME")
    r.add_argument("--run", help="run name (default eyespy-<a>-vs-<b>)")
    a = p.parse_args(argv)
    if a.cmd == "review":
        review_run(a.workspace, a.models, a.checkpoints, a.run)
        return
    if a.cmd == "train":
        train(
            a.version,
            a.source,
            a.workspace,
            a.checkpoints,
            a.samples,
            a.note,
            a.replace,
            a.experiment,
        )
    else:
        write_index(a.workspace, a.checkpoints)


if __name__ == "__main__":
    sys.exit(main())
