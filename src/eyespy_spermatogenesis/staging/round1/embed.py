"""Frozen feature extraction on 224 px patches of each tubule crop.

Usage::

    HF_HUB_CACHE=... python -m eyespy_spermatogenesis.staging.round1.embed \
        DATA_DIR OUT_DIR --model h-optimus-0 [--batch 32] \
        [--views s05_raw ...] [--tta d4]

For every crop in ``DATA_DIR/labels.csv`` and every view
(scale x masking) it tiles the crop into 224 px patches, runs the frozen model
and stores one embedding per patch in ``OUT_DIR/<model>/<view>.npz``:
``emb`` (float16, n_patches x dim), ``crop`` (row index into labels.csv) and
``frac`` (fraction of the patch inside the tubule outline).

Views: scale ``s1`` (native pixels), ``s05`` (2x downsampled), ``fit`` (whole
crop resized to one patch); ``raw`` (crop as is) or ``masked`` (outside the
outline set to white). Physical pixel size is unverified, so both native and
downsampled patches are kept; evaluation decides.

``--tta d4`` (optional, off by default) runs each patch in its 8 rotations and
flips and stores the mean embedding; output goes to ``OUT_DIR/<model>_d4/`` so
the plain embeddings are never overwritten. ``--views`` limits the views.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from eyespy_spermatogenesis.staging.round1.data import read_table

TILE = 224
SCALES = {"s1": 1.0, "s05": 0.5, "fit": None}
MASKINGS = ("raw", "masked")

MODELS = {
    # name: (timm id, create kwargs, mean, std)
    "h-optimus-0": (
        "hf-hub:bioptimus/H-optimus-0",
        {"init_values": 1e-5, "dynamic_img_size": False},
        (0.707223, 0.578729, 0.703617),
        (0.211883, 0.230117, 0.177517),
    ),
    "dinov2-l": (
        "hf-hub:timm/vit_large_patch14_dinov2.lvd142m",
        {"num_classes": 0, "img_size": TILE},
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
    # RESEARCH ONLY (CC-BY-NC-ND-4.0 + Mahmood lab terms): see RESEARCH_ONLY.
    # Settings copied from the MahmoodLab/UNI2-h model card (2026-11-11);
    # "mlp_layer"/"act_layer" are resolved in load_model (they need timm/torch).
    "uni2-h": (
        "hf-hub:MahmoodLab/UNI2-h",
        {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": "SwiGLUPacked",
            "act_layer": "SiLU",
            "reg_tokens": 8,
            "dynamic_img_size": True,
        },
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225),
    ),
}

# Extractors whose weights, and anything trained on their outputs, may only be
# used for non-commercial academic research and never distributed. A staging
# model built on one of these cannot ship in a released app.
RESEARCH_ONLY = {
    "uni2-h": "UNI2-h (MahmoodLab), CC-BY-NC-ND-4.0: non-commercial academic "
    "research only; the model, models trained on its outputs and datasets made "
    "from it must not be distributed or used commercially.",
}


TTA = ("none", "d4")


def d4(x):
    """The 8 rotations/flips of a (n, c, h, w) batch, stacked: (8n, c, h, w)."""
    import torch

    rots = [torch.rot90(x, k, dims=(2, 3)) for k in range(4)]
    return torch.cat(rots + [torch.flip(r, dims=(3,)) for r in rots])


def out_name(model_name: str, tta: str = "none") -> str:
    return model_name if tta == "none" else f"{model_name}_{tta}"


def tile_starts(length: int, tile: int = TILE) -> list[int]:
    """Evenly spaced tile origins covering ``length`` (overlap if needed)."""
    if length <= tile:
        return [0]
    n = math.ceil(length / tile)
    return [round(i * (length - tile) / (n - 1)) for i in range(n)]


def views(img: np.ndarray, mask: np.ndarray):
    """Yield (view_name, patches[n,224,224,3] uint8, frac[n]) for one crop."""
    for mname in MASKINGS:
        im = img.copy()
        if mname == "masked":
            im[~mask] = 255
        for sname, s in SCALES.items():
            pim, pmk = Image.fromarray(im), Image.fromarray(mask.astype(np.uint8) * 255)
            if s is None:
                size = (TILE, TILE)
            else:
                size = (max(1, round(im.shape[1] * s)), max(1, round(im.shape[0] * s)))
            if size != (im.shape[1], im.shape[0]):
                pim = pim.resize(size, Image.LANCZOS)
                pmk = pmk.resize(size, Image.NEAREST)
            a, m = np.asarray(pim), np.asarray(pmk) > 0
            if a.shape[0] < TILE or a.shape[1] < TILE:  # pad small crops white
                ph, pw = max(0, TILE - a.shape[0]), max(0, TILE - a.shape[1])
                a = np.pad(a, ((0, ph), (0, pw), (0, 0)), constant_values=255)
                m = np.pad(m, ((0, ph), (0, pw)))
            patches, fracs = [], []
            for y in tile_starts(a.shape[0]):
                for x in tile_starts(a.shape[1]):
                    patches.append(a[y : y + TILE, x : x + TILE])
                    fracs.append(m[y : y + TILE, x : x + TILE].mean())
            yield f"{sname}_{mname}", np.stack(patches), np.asarray(fracs, np.float32)


class _Crops:
    def __init__(self, data_dir: Path, rows, keep=None):
        self.d, self.rows, self.keep = data_dir, rows, keep

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        k = self.rows[i]["key"]
        img = np.asarray(Image.open(self.d / "crops" / f"{k}.png").convert("RGB"))
        mask = np.asarray(Image.open(self.d / "masks" / f"{k}.png")) > 0
        vs = views(img, mask)
        return i, [v for v in vs if self.keep is None or v[0] in self.keep]


def load_model(name: str):
    import timm
    import torch

    tid, kw, mean, std = MODELS[name]
    kw = dict(kw)
    if kw.get("mlp_layer") == "SwiGLUPacked":
        kw["mlp_layer"] = timm.layers.SwiGLUPacked
    if kw.get("act_layer") == "SiLU":
        kw["act_layer"] = torch.nn.SiLU
    # fp16 weights: the 1.1B-parameter H-optimus-0 does not fit 8 GB in fp32
    model = timm.create_model(tid, pretrained=True, **kw).eval().half().to("cuda")
    mean_t = torch.tensor(mean, device="cuda").view(1, 3, 1, 1)
    std_t = torch.tensor(std, device="cuda").view(1, 3, 1, 1)
    return model, mean_t, std_t


def embed(
    data_dir: Path,
    out_dir: Path,
    model_name: str,
    batch: int = 32,
    workers=8,
    keep_views=None,
    tta: str = "none",
):
    import torch

    rows = read_table(data_dir)
    model, mean, std = load_model(model_name)
    if tta == "d4":
        batch = max(1, batch // 8)  # each patch becomes 8 images
    loader = torch.utils.data.DataLoader(
        _Crops(data_dir, rows, set(keep_views) if keep_views else None),
        batch_size=None,
        num_workers=workers,
        prefetch_factor=4,
    )
    store: dict[str, dict[str, list]] = {}
    pending: list = []  # (view, crop index, frac, patch)

    def flush():
        if not pending:
            return
        x = torch.from_numpy(np.stack([p[3] for p in pending])).to("cuda")
        x = ((x.permute(0, 3, 1, 2).float() / 255 - mean) / std).half()
        with torch.inference_mode():
            if tta == "d4":
                f = model(d4(x)).float().view(8, len(pending), -1).mean(0)
            else:
                f = model(x).float()
            f = f.cpu().numpy().astype(np.float16)
        for (v, ci, fr, _), e in zip(pending, f):
            s = store.setdefault(v, {"emb": [], "crop": [], "frac": []})
            s["emb"].append(e)
            s["crop"].append(ci)
            s["frac"].append(fr)
        pending.clear()

    t0 = time.time()
    for n, (ci, vs) in enumerate(loader, 1):
        for v, patches, fracs in vs:
            for p, fr in zip(patches, fracs):
                pending.append((v, ci, fr, p))
                if len(pending) >= batch:
                    flush()
        if n % 200 == 0:
            print(f"{n}/{len(rows)} crops, {time.time() - t0:.0f} s", flush=True)
    flush()
    od = out_dir / out_name(model_name, tta)
    od.mkdir(parents=True, exist_ok=True)
    for v, s in store.items():
        np.savez(
            od / f"{v}.npz",
            emb=np.stack(s["emb"]),
            crop=np.asarray(s["crop"], np.int32),
            frac=np.asarray(s["frac"], np.float32),
        )
    (od / "embed.json").write_text(
        json.dumps(
            {
                "model": model_name,
                "timm_id": MODELS[model_name][0],
                "data_dir": str(data_dir),
                "crops": len(rows),
                "views": {v: len(s["crop"]) for v, s in store.items()},
                "seconds": round(time.time() - t0),
                "torch": torch.__version__,
                "research_only": RESEARCH_ONLY.get(model_name),
                "tta": tta,
            },
            indent=2,
        )
    )
    print(f"done in {time.time() - t0:.0f} s -> {od}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("data_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--model", choices=sorted(MODELS), required=True)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--views", nargs="+", help="only these views (default all)")
    p.add_argument("--tta", choices=TTA, default="none")
    a = p.parse_args(argv)
    embed(a.data_dir, a.out_dir, a.model, a.batch, a.workers, a.views, a.tta)


if __name__ == "__main__":
    sys.exit(main())
