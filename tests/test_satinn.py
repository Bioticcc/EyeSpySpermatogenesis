"""SATINN PyTorch rebuild on a tiny hand-made converted network (no MATLAB)."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from eyespy_spermatogenesis.staging import satinn  # noqa: E402


def _tiny(tmp_path):
    rng = np.random.default_rng(0)
    layers = [
        {"name": "input", "type": "input", "size": [6, 6, 3]},
        {
            "name": "conv1",
            "type": "conv",
            "stride": [1, 1],
            "dilation": [1, 1],
            "pad": [1, 1, 1, 1],
        },
        {"name": "bn1", "type": "bn", "eps": 1e-5},
        {"name": "relu1", "type": "relu"},
        # listed before its input on purpose: MATLAB's order is not execution order
        {"name": "add", "type": "add"},
        {
            "name": "skip",
            "type": "conv",
            "stride": [1, 1],
            "dilation": [1, 1],
            "pad": [0, 0, 0, 0],
        },
        {"name": "gap", "type": "gap"},
        {"name": "fc", "type": "fc"},
        {"name": "softmax", "type": "softmax"},
        {"name": "output", "type": "output", "classes": ["a", "b"]},
    ]
    edges = [
        ["input", "conv1"],
        ["conv1", "bn1"],
        ["bn1", "relu1"],
        ["relu1", "add"],
        ["input", "skip"],
        ["skip", "add"],
        ["add", "gap"],
        ["gap", "fc"],
        ["fc", "softmax"],
        ["softmax", "output"],
    ]
    a = {
        "input.mean": np.array([[[10, 20, 30]]], np.float32),
        "conv1.weight": rng.normal(size=(3, 3, 3, 4)).astype(
            np.float32
        ),  # H W Cin Cout
        "conv1.bias": rng.normal(size=(1, 1, 4)).astype(np.float32),
        "bn1.mean": rng.normal(size=4).astype(np.float32),
        "bn1.var": rng.uniform(0.5, 2, 4).astype(np.float32),
        "bn1.scale": rng.normal(size=4).astype(np.float32),
        "bn1.offset": rng.normal(size=4).astype(np.float32),
        "skip.weight": rng.normal(size=(1, 1, 3, 4)).astype(np.float32),
        "skip.bias": np.zeros((1, 1, 4), np.float32),
        "fc.weight": rng.normal(size=(1, 1, 4, 2)).astype(np.float32),
        "fc.bias": np.array([0.1, -0.1], np.float32),
    }
    meta = {"source": "test", "layers": layers, "edges": edges}
    p = tmp_path / "tiny.npz"
    np.savez(p, __meta__=np.frombuffer(json.dumps(meta).encode(), np.uint8), **a)
    return p, a


def _reference(x, a):
    """Plain NumPy forward pass, MATLAB conventions (H W C, weights H W Cin Cout)."""
    x = x - a["input.mean"][0, 0]
    xp = np.pad(x, ((1, 1), (1, 1), (0, 0)))
    w = a["conv1.weight"]
    h = np.zeros((6, 6, 4))
    for i in range(6):
        for j in range(6):
            h[i, j] = np.einsum("abc,abcd->d", xp[i : i + 3, j : j + 3], w)
    h += a["conv1.bias"][0, 0]
    h = (h - a["bn1.mean"]) / np.sqrt(a["bn1.var"] + 1e-5) * a["bn1.scale"] + a[
        "bn1.offset"
    ]
    h = np.maximum(h, 0)
    h = h + np.einsum("ijc,cd->ijd", x, a["skip.weight"][0, 0])
    g = h.mean((0, 1))
    z = g @ a["fc.weight"][0, 0] + a["fc.bias"]
    e = np.exp(z - z.max())
    return e / e.sum()


def test_rebuild_matches_numpy_reference(tmp_path):
    path, a = _tiny(tmp_path)
    net, meta = satinn.load(path)
    assert net.classes == ["a", "b"]
    x = np.random.default_rng(1).uniform(0, 255, (6, 6, 3)).astype(np.float32)
    got = net(torch.from_numpy(x).permute(2, 0, 1)[None]).detach().numpy()[0]
    np.testing.assert_allclose(got, _reference(x, a), rtol=1e-4, atol=1e-6)


def test_taps_see_layer_inputs(tmp_path):
    path, _ = _tiny(tmp_path)
    net, _ = satinn.load(path)
    seen = []
    net(torch.zeros(2, 3, 6, 6), {"bn1": seen.append})
    assert seen[0].shape == (2, 4, 6, 6)


def test_to_ours_merges_ii_and_iii():
    from eyespy_spermatogenesis.staging import round2

    p = np.zeros((1, 12))
    p[0, 1], p[0, 2], p[0, 11] = 0.3, 0.3, 0.4  # II 0.3 + III 0.3 beats XII 0.4
    q = round2.to_ours(p)
    assert q.shape == (1, 11) and q[0, 1] == pytest.approx(0.6)
    assert int(q.argmax()) == 1  # II-III


def test_preprocess_recentres_and_scales_contrast():
    from eyespy_spermatogenesis.staging import round2

    x = torch.tensor([[[[100, 100, 100]], [[120, 120, 120]]]], dtype=torch.uint8)
    x = x.expand(1, 2, 2, 3).contiguous()  # (N, H, W, C)
    mean = torch.tensor([150.0, 150.0, 150.0])
    raw = round2.preprocess(x, "raw", mean)
    assert raw.shape == (1, 3, 2, 2) and raw[0, 0, 0, 0] == 100
    g = round2.preprocess(x, 2.0, mean)
    # image mean 110 moved to 150; deviations of +-10 doubled
    assert g[0, 0, 0, 0] == pytest.approx(130) and g[0, 0, 1, 0] == pytest.approx(170)
