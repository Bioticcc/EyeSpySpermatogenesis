"""SATINN v2 bright-field tubule stager, converted from MATLAB to PyTorch.

SATINN (Conrad lab; Yang et al. 2022, 2025) ships its pretrained networks as
MATLAB ``DAGNetwork`` objects on Figshare (CC BY 4.0; record 22853645). The
tubule classifier ``bf_tnetr-*.mat`` is a ResNet-50: 500x500 RGB input,
"zerocenter" normalization (per-channel mean subtracted from 0-255 values),
12 outputs I, II, III, ..., XII.

Usage::

    # once: decode the .mat (needs mat-io) into plain arrays
    python -m eyespy_spermatogenesis.staging.satinn convert NET.mat OUT.npz

``load(OUT.npz)`` then rebuilds the network in PyTorch with no MATLAB and no
mat-io. Layer types supported: image input, 2-D convolution, batch norm,
ReLU, max pooling, addition, global average pooling, fully connected,
softmax, classification output. Anything else raises an error.

MATLAB stores conv weights as (H, W, Cin, Cout) and pads as
[top, bottom, left, right]; both are mapped explicitly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SATINN_STAGES = [
    "I",
    "II",
    "III",
    "IV",
    "V",
    "VI",
    "VII",
    "VIII",
    "IX",
    "X",
    "XI",
    "XII",
]


# ---------------------------------------------------------------- conversion
def _val(x):
    while isinstance(x, np.ndarray) and x.dtype == object and x.size == 1:
        x = x.ravel()[0]
    return x


def _learnable(rec):
    """Weights/Bias are stored as a record with a 'Value' field."""
    v = _val(rec)
    if isinstance(v, np.ndarray) and v.dtype.names and "Value" in v.dtype.names:
        v = _val(v["Value"])
    return np.asarray(v, np.float32)


def _num(x):
    return np.asarray(_val(x), np.float64).ravel().tolist()


def _str(x):
    v = _val(x)
    return str(np.asarray(v).ravel()[0]) if isinstance(v, np.ndarray) else str(v)


def convert(mat_path: Path, out_path: Path) -> dict:
    from matio import load_from_mat  # conversion-time dependency only

    net = load_from_mat(str(mat_path))["net"]
    if net.classname != "DAGNetwork":
        raise ValueError(f"Expected DAGNetwork, got {net.classname}")
    rec = net.properties["any"]
    layers = _val(rec["Layers"]).ravel()
    conns = _val(rec["Connections"])
    arrays, spec = {}, []
    for o in layers:
        a = o.properties["any"]
        f = set(a.dtype.names)
        name = _str(a["Name"])
        if "Normalization" in f:
            nrm = _val(a["Normalization"])
            kind = _str(nrm["Type"])
            if kind != "zerocenter":
                raise ValueError(f"Unsupported input normalization {kind}")
            arrays[f"{name}.mean"] = np.asarray(_val(nrm["AverageImage"]), np.float32)
            spec.append({"name": name, "type": "input", "size": _num(a["InputSize"])})
        elif "FilterSize" in f and "NumFilters" in f:
            arrays[f"{name}.weight"] = _learnable(a["Weights"])
            arrays[f"{name}.bias"] = _learnable(a["Bias"])
            spec.append(
                {
                    "name": name,
                    "type": "conv",
                    "stride": _num(a["Stride"]),
                    "dilation": _num(a["DilationFactor"]),
                    "pad": _num(a["PaddingSize"]),  # top, bottom, left, right
                }
            )
        elif "TrainedMean" in f:
            for k, src in (
                ("mean", "TrainedMean"),
                ("var", "TrainedVariance"),
                ("scale", "Scale"),
                ("offset", "Offset"),
            ):
                v = a[src] if src.startswith("Trained") else a[src]
                arrays[f"{name}.{k}"] = (
                    np.asarray(_val(v), np.float32).ravel()
                    if src.startswith("Trained")
                    else _learnable(v).ravel()
                )
            spec.append({"name": name, "type": "bn", "eps": _num(a["Epsilon"])[0]})
        elif "PoolSize" in f:
            spec.append(
                {
                    "name": name,
                    "type": "maxpool",
                    "size": _num(a["PoolSize"]),
                    "stride": _num(a["Stride"]),
                    "pad": _num(a["PaddingSize"]),
                }
            )
        elif "NumInputs" in f:
            spec.append({"name": name, "type": "add"})
        elif "OutputSize" in f and "Weights" in f:
            arrays[f"{name}.weight"] = _learnable(a["Weights"])
            arrays[f"{name}.bias"] = _learnable(a["Bias"])
            spec.append({"name": name, "type": "fc"})
        elif "Categories" in f:
            cats = [str(np.asarray(c).ravel()[0]) for c in _val(a["Categories"])]
            spec.append({"name": name, "type": "output", "classes": cats})
        elif "VectorFormat" in f:
            spec.append({"name": name, "type": "softmax"})
        elif f == {"Version", "Name"}:
            # relu and global average pooling carry no parameters; MATLAB's
            # default names are used by SATINN ('relu*', 'gap')
            lt = "gap" if name == "gap" or "pool" in name else "relu"
            if lt == "relu" and "relu" not in name:
                raise ValueError(f"Cannot tell the type of parameterless layer {name}")
            spec.append({"name": name, "type": lt})
        else:
            raise ValueError(f"Unsupported layer {name}: {sorted(f)}")
    edges = [
        [str(np.ravel(s)[0]).split("/")[0], str(np.ravel(d)[0]).split("/")[0]]
        for s, d in zip(conns["Source"], conns["Destination"])
    ]
    meta = {"source": str(mat_path), "layers": spec, "edges": edges}
    np.savez(
        out_path, __meta__=np.frombuffer(json.dumps(meta).encode(), np.uint8), **arrays
    )
    return meta


# ---------------------------------------------------------------- PyTorch
def load(npz_path: Path, device="cpu"):
    """Rebuild the converted network as a torch.nn.Module (eval mode)."""
    import torch
    import torch.nn.functional as F

    z = np.load(npz_path)
    meta = json.loads(bytes(z["__meta__"]).decode())
    spec = {s["name"]: s for s in meta["layers"]}
    names = [s["name"] for s in meta["layers"]]
    parents = {n: [] for n in names}
    for s, d in meta["edges"]:
        parents[d].append(s)
    # MATLAB's layer list is not in execution order (shortcut branches come
    # later): sort topologically, keeping the stored order among ready layers.
    order, done = [], set()
    while len(order) < len(names):
        ready = [
            n for n in names if n not in done and all(p in done for p in parents[n])
        ]
        if not ready:
            raise ValueError("Network graph has a cycle")
        order.append(ready[0])
        done.add(ready[0])

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classes = next(
                s["classes"] for s in meta["layers"] if s["type"] == "output"
            )
            self.p = torch.nn.ParameterDict()
            for k in z.files:
                if k == "__meta__":
                    continue
                a = z[k]
                name, what = k.rsplit(".", 1)
                t = spec[name]["type"]
                if t == "conv" and what == "weight":
                    a = a.transpose(3, 2, 0, 1)  # H W Cin Cout -> Cout Cin H W
                elif t == "fc" and what == "weight":
                    a = a.reshape(-1, a.shape[-1]).T  # (1,1,Cin,Cout) -> Cout Cin
                elif t == "input":
                    a = (
                        a.reshape(1, 3, 1, 1)
                        if a.size == 3
                        else a.transpose(2, 0, 1)[None]
                    )
                else:
                    a = a.ravel()
                key = k.replace(".", "__")
                self.p[key] = torch.nn.Parameter(
                    torch.from_numpy(np.ascontiguousarray(a)), requires_grad=False
                )

        def g(self, name, what):
            return self.p[f"{name}__{what}"]

        def forward(self, x, taps=None):
            """x: (N, 3, H, W) float, RGB 0-255. Returns class probabilities.
            ``taps``: optional dict name -> callable, called with each
            listed layer's input (for checking batch-norm statistics)."""
            out = {}
            left = {n: 0 for n in order}  # consumers not yet run, to free memory
            for d, ps in parents.items():
                for p_ in ps:
                    left[p_] += 1
            for n in order:
                s = spec[n]
                ins = [out[p] for p in parents[n]] if parents[n] else [x]
                for p_ in parents[n]:
                    left[p_] -= 1
                    if left[p_] == 0 and p_ != order[-1]:
                        del out[p_]
                h = ins[0]
                if taps is not None and n in taps:
                    taps[n](h.detach())
                t = s["type"]
                if t == "input":
                    h = h - self.g(n, "mean")
                elif t == "conv":
                    tp, bt, lf, rt = (int(v) for v in s["pad"])
                    h = F.pad(h, (lf, rt, tp, bt))
                    h = F.conv2d(
                        h,
                        self.g(n, "weight"),
                        self.g(n, "bias"),
                        stride=[int(v) for v in s["stride"]],
                        dilation=[int(v) for v in s["dilation"]],
                    )
                elif t == "bn":
                    h = F.batch_norm(
                        h,
                        self.g(n, "mean"),
                        self.g(n, "var"),
                        self.g(n, "scale"),
                        self.g(n, "offset"),
                        training=False,
                        eps=s["eps"],
                    )
                elif t == "relu":
                    h = F.relu(h)
                elif t == "maxpool":
                    tp, bt, lf, rt = (int(v) for v in s["pad"])
                    h = F.pad(h, (lf, rt, tp, bt), value=float("-inf"))
                    h = F.max_pool2d(
                        h, [int(v) for v in s["size"]], [int(v) for v in s["stride"]]
                    )
                elif t == "add":
                    h = sum(ins)
                elif t == "gap":
                    h = h.mean((2, 3))
                elif t == "fc":
                    h = F.linear(h.flatten(1), self.g(n, "weight"), self.g(n, "bias"))
                elif t == "softmax":
                    h = torch.softmax(h, 1)
                out[n] = h
            return out[order[-1]]

    return Net().eval().to(device), meta


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("mat", type=Path)
    c.add_argument("out", type=Path)
    a = p.parse_args(argv)
    meta = convert(a.mat, a.out)
    kinds = {}
    for s in meta["layers"]:
        kinds[s["type"]] = kinds.get(s["type"], 0) + 1
    print(
        json.dumps(
            {"layers": len(meta["layers"]), "edges": len(meta["edges"]), "types": kinds}
        )
    )


if __name__ == "__main__":
    sys.exit(main())
