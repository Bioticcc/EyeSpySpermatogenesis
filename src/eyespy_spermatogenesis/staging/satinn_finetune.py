"""Fine-tune SATINN's bright-field stager on our paper labels (opt-in).

Usage::

    python -m eyespy_spermatogenesis.staging.satinn_finetune \\
        DATA_DIR WORKSPACE NET_NPZ OUT_DIR [--epochs 8] [--lr 1e-4] \\
        [--batch 12] [--side 500] [--fill black] [--balanced]

Starts from the converted SATINN network (``staging.satinn``, CC BY 4.0
weights, so a fine-tuned model is releasable) with round 2's best label-free
input: a ``--side`` px native box around the tubule centre, resized to 500 px,
everything outside the v0.5 outline set to ``--fill``, raw contrast.

Leave-one-section-out: for each held-out section the network is fine-tuned
on the other sections only. Hyper-parameters are fixed in advance (not tuned
on held-out scores). The 12-class head is kept; II and III probabilities are
summed for our II-III class, so the loss is the negative log of the merged
probability. Batch-norm statistics stay frozen during training (small
batches); two label-free variants are also scored:

* ``pretrained``: the network as published (round 2 setting);
* ``adabn``: batch-norm statistics re-estimated on the training sections'
  images (no labels), weights unchanged.

Augmentation: random 90-degree rotations and flips (stage does not depend on
orientation). Outputs: ``results.json``, ``summary.md``, ``predictions.npz``
(11-class probabilities per variant), ``train.log``. Fine-tuned weights are
not kept unless ``--save-weights`` is given (8 x 95 MB).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from eyespy_spermatogenesis.staging.round1.data import STAGES, read_table
from eyespy_spermatogenesis.staging.round1.evaluate import metrics
from eyespy_spermatogenesis.staging.round2 import IN, read_boxes

FILLS = ("none", "white", "black")


def box_masks(data_dir: Path, rows, side: int) -> np.ndarray:
    """Tubule outline in each box's 500x500 frame (box centred as in round 2)."""
    out = np.zeros((len(rows), IN, IN), bool)
    for i, r in enumerate(rows):
        b = json.loads(r["source_bounds"])
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
        m = np.asarray(Image.open(data_dir / "masks" / f"{r['key']}.png")) > 0
        full = np.zeros((side, side), bool)
        ox, oy = b[0] - x0, b[1] - y0
        sx0, sy0 = max(0, -ox), max(0, -oy)
        dx0, dy0 = max(0, ox), max(0, oy)
        w = min(m.shape[1] - sx0, side - dx0)
        h = min(m.shape[0] - sy0, side - dy0)
        if w > 0 and h > 0:
            full[dy0 : dy0 + h, dx0 : dx0 + w] = m[sy0 : sy0 + h, sx0 : sx0 + w]
        out[i] = np.asarray(Image.fromarray(full).resize((IN, IN), Image.NEAREST))
    return out


def merge_probs(p12):
    """(N, 12) SATINN probabilities -> (N, 11) ours (II + III summed). Torch."""
    import torch

    return torch.cat([p12[:, :1], p12[:, 1:3].sum(1, keepdim=True), p12[:, 3:]], 1)


def bn_names(net):
    return sorted({k.split("__")[0] for k in net.p if k.endswith("__var")})


class _Stop(Exception):
    pass


def adapt_bn(net, arr, idx, batch=16, n_sub=256, seed=0):
    """Re-estimate every batch-norm layer's mean/var (AdaBN). No labels used.

    Layers are estimated one at a time in network order, each from inputs
    produced by the already re-estimated layers before it (estimating all
    layers in one pass diverges on this network). Each pass stops at the
    layer being estimated. Uses a fixed random subset of ``n_sub`` crops."""
    import torch

    names = set(bn_names(net))
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(idx, min(n_sub, len(idx)), replace=False))
    x0 = torch.from_numpy(arr[sub[:1]]).cuda().permute(0, 3, 1, 2).float()
    order = []

    def rec(n):
        return lambda h: order.append(n)

    with torch.no_grad():  # discover network order of the batch-norm layers
        net(x0, {n: rec(n) for n in names})
    for n in order:
        acc = [0.0, 0.0, 0]

        def tap(h):
            h = h.double()
            acc[0] = acc[0] + h.sum((0, 2, 3))
            acc[1] = acc[1] + (h * h).sum((0, 2, 3))
            acc[2] += h[:, 0].numel()
            raise _Stop

        # fp32: fp16 activations overflow in stack3 once statistics change.
        with torch.no_grad():
            for s in range(0, len(sub), batch):
                x = torch.from_numpy(arr[sub[s : s + batch]]).cuda()
                try:
                    net(x.permute(0, 3, 1, 2).float(), {n: tap})
                except _Stop:
                    pass
        m = acc[0] / acc[2]
        v = (acc[1] / acc[2] - m * m).clamp_min(0)
        net.g(n, "mean").data.copy_(m.to(net.g(n, "mean").dtype))
        net.g(n, "var").data.copy_(v.to(net.g(n, "var").dtype))


def predict(net, arr, idx, batch=16):
    import torch

    out = []
    with torch.no_grad():  # fp32 (see adapt_bn)
        for s in range(0, len(idx), batch):
            x = torch.from_numpy(arr[idx[s : s + batch]]).cuda()
            p = net(x.permute(0, 3, 1, 2).float())
            out.append(merge_probs(p.float()).cpu().numpy())
    out = np.concatenate(out)
    if not np.isfinite(out).all():  # NaN would argmax to stage I silently
        raise FloatingPointError("non-finite SATINN probabilities")
    return out


def augment(x, rng):
    """Random 90-degree rotation and horizontal flip per batch. (N,H,W,3)."""
    k = int(rng.integers(4))
    x = np.rot90(x, k, axes=(1, 2))
    if rng.integers(2):
        x = x[:, :, ::-1]
    return np.ascontiguousarray(x)


def finetune(net, arr, idx, y, epochs, lr, batch, balanced, seed, log):
    import torch

    trainable = [
        p
        for k, p in net.p.items()
        if not (k.endswith("__mean") or k.endswith("__var"))
        and not k.startswith("input")
    ]
    for p in trainable:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-4)
    steps = epochs * int(np.ceil(len(idx) / batch))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)
    rng = np.random.default_rng(seed)
    if balanced:
        cnt = np.bincount(y, minlength=len(STAGES)).astype(float)
        w = 1 / cnt[y]
        w /= w.sum()
    yt = torch.from_numpy(y).cuda()
    for ep in range(epochs):
        order = (
            rng.choice(len(idx), len(idx), p=w)
            if balanced
            else rng.permutation(len(idx))
        )
        tot, n = 0.0, 0
        for s in range(0, len(order), batch):
            b = order[s : s + batch]
            x = torch.from_numpy(augment(arr[idx[b]], rng)).cuda()
            # bf16: fp16's range overflows in this network (no scaler needed)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p12 = net(x.permute(0, 3, 1, 2).float())
            p = merge_probs(p12.float()).clamp_min(1e-7)
            loss = torch.nn.functional.nll_loss(p.log(), yt[b])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at epoch {ep + 1}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot, n = tot + loss.item() * len(b), n + len(b)
        log(f"  epoch {ep + 1}/{epochs} loss {tot / n:.3f}")
    for p in trainable:
        p.requires_grad_(False)


def run(
    data_dir,
    workspace,
    net_npz,
    out_dir,
    epochs=8,
    lr=1e-4,
    batch=12,
    side=500,
    fill="black",
    balanced=False,
    seed=0,
    save_weights=False,
    sections=None,
):
    import torch

    from eyespy_spermatogenesis.staging import satinn

    out_dir.mkdir(parents=True, exist_ok=True)
    logf = open(out_dir / "train.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    rows = read_table(data_dir)
    ev = np.array(
        [
            i
            for i, r in enumerate(rows)
            if r["recommended"] and r["stage"] in STAGES and not r["needs_check"]
        ]
    )
    y = np.array([STAGES.index(rows[i]["stage"]) for i in ev])
    g = np.array([rows[i]["section"] for i in ev])
    t0 = time.time()
    arr = read_boxes(workspace, rows, side)
    if fill != "none":
        arr[~box_masks(data_dir, rows, side)] = 255 if fill == "white" else 0
    log(f"read {len(rows)} boxes in {time.time() - t0:.0f} s")

    variants = ("pretrained", "adabn", "finetuned")
    probs = {v: np.zeros((len(ev), len(STAGES)), np.float32) for v in variants}
    secs = sections or sorted(set(g))
    for s in secs:
        te, tr = g == s, g != s
        log(f"held out {s}: train {tr.sum()} test {te.sum()}")
        net, _ = satinn.load(net_npz, "cuda")
        probs["pretrained"][te] = predict(net, arr, ev[te])
        adapt_bn(net, arr, ev[tr])
        probs["adabn"][te] = predict(net, arr, ev[te])
        torch.manual_seed(seed)
        finetune(net, arr, ev[tr], y[tr], epochs, lr, batch, balanced, seed, log)
        probs["finetuned"][te] = predict(net, arr, ev[te])
        for v in variants:
            acc = (probs[v][te].argmax(1) == y[te]).mean()
            log(f"  {v}: section accuracy {acc:.3f}")
        if save_weights:
            torch.save(
                {k: p.detach().cpu() for k, p in net.p.items()},
                out_dir / f"weights_{s.lstrip('#')}.pt",
            )
        del net
        torch.cuda.empty_cache()

    done = np.isin(g, secs)
    res = {
        "settings": {
            "net": str(net_npz),
            "epochs": epochs,
            "lr": lr,
            "batch": batch,
            "side": side,
            "fill": fill,
            "balanced": balanced,
            "seed": seed,
        },
        "sections": list(secs),
        "n": int(done.sum()),
        "variants": {v: metrics(y[done], probs[v][done].argmax(1)) for v in variants},
        "seconds": round(time.time() - t0),
    }
    (out_dir / "results.json").write_text(json.dumps(res, indent=1))
    np.savez(
        out_dir / "predictions.npz",
        tubule_id=np.array([rows[i]["tubule_id"] for i in ev]),
        y=y,
        section=g,
        **probs,
    )
    L = [
        "# SATINN fine-tuning: leave-one-section-out",
        "",
        f"{res['n']} tubules, sections {', '.join(secs)}. Settings: "
        + ", ".join(f"{k} {v}" for k, v in res["settings"].items()),
        "",
        "| Variant | Exact | Balanced | Within one |",
        "|---|---|---|---|",
    ]
    for v, m in res["variants"].items():
        L.append(
            f"| {v} | {m['acc']:.3f} | {m['balanced_acc']:.3f} | {m['within1']:.3f} |"
        )
    L += ["", "| Variant | " + " | ".join(STAGES) + " |", "|---" * 12 + "|"]
    for v, m in res["variants"].items():
        L.append(
            f"| {v} | "
            + " | ".join(f"{m['recall'].get(s, float('nan')):.2f}" for s in STAGES)
            + " |"
        )
    (out_dir / "summary.md").write_text("\n".join(L) + "\n")
    log(f"done in {time.time() - t0:.0f} s")
    return res


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", type=Path)
    p.add_argument("workspace", type=Path)
    p.add_argument("net_npz", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--side", type=int, default=500)
    p.add_argument("--fill", choices=FILLS, default="black")
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-weights", action="store_true")
    p.add_argument("--sections", nargs="+", help="only these held-out sections")
    a = p.parse_args(argv)
    run(
        a.data_dir,
        a.workspace,
        a.net_npz,
        a.out_dir,
        a.epochs,
        a.lr,
        a.batch,
        a.side,
        a.fill,
        a.balanced,
        a.seed,
        a.save_weights,
        a.sections,
    )


if __name__ == "__main__":
    sys.exit(main())
