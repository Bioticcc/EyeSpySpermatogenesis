"""Staging round 1: tiling, cyclic head, metrics and label loading (no GPU)."""

import json

import numpy as np
import pytest

pytest.importorskip("sklearn")

from eyespy_spermatogenesis.staging.annotate.store import PAPER, Store  # noqa: E402
from eyespy_spermatogenesis.staging.round1 import data, embed, evaluate  # noqa: E402


def test_tile_starts_cover_and_stay_inside():
    assert embed.tile_starts(100) == [0]
    assert embed.tile_starts(224) == [0]
    s = embed.tile_starts(800)
    assert s[0] == 0 and s[-1] == 800 - 224 and len(s) == 4
    assert all(b - a <= 224 for a, b in zip(s, s[1:]))  # no gaps


def test_views_shapes_and_mask_fraction():
    img = np.full((500, 460, 3), 100, np.uint8)
    mask = np.zeros((500, 460), bool)
    mask[:, :230] = True
    got = {name: (p, f) for name, p, f in embed.views(img, mask)}
    assert set(got) == {f"{s}_{m}" for s in embed.SCALES for m in embed.MASKINGS}
    p, f = got["s1_raw"]
    assert p.shape == (9, 224, 224, 3)  # 3 x 3 tiles over 500 x 460
    assert f[0] == 1.0 and f[-1] == 0.0
    p, _ = got["fit_masked"]
    assert p.shape == (1, 224, 224, 3)
    p, _ = got["s1_masked"]
    assert p[-1].min() == 255  # right edge tile is outside the outline: white
    p, _ = got["s05_raw"]
    assert p.shape == (4, 224, 224, 3)


def test_cyclic_distance_wraps():
    assert evaluate.cyc_dist(0, 10) == 1  # I next to XII
    assert evaluate.cyc_dist(2, 7) == 5
    assert evaluate.cyc_dist(3, 3) == 0


def test_cyclic_head_recovers_separable_stages():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(evaluate.K), 30)
    ang = 2 * np.pi * y / evaluate.K
    x = np.c_[np.cos(ang), np.sin(ang), rng.normal(size=(len(y), 5))]
    x[:, :2] += rng.normal(scale=0.05, size=(len(y), 2))
    p = evaluate.Cyclic(1.0).fit(x, y).predict(x)
    assert (p == y).mean() > 0.95


def test_metrics_balanced_and_within_one():
    y = np.array([0, 0, 0, 0, 5])
    p = np.array([0, 0, 0, 10, 5])
    m = evaluate.metrics(y, p)
    assert m["acc"] == pytest.approx(0.8)
    assert m["balanced_acc"] == pytest.approx((0.75 + 1.0) / 2)
    assert m["within1"] == 1.0  # XII is one stage from I


def test_load_labels_reads_paper_labels(tmp_path):
    s = Store(tmp_path / "labels.sqlite")
    s.add_tubules(
        [
            {
                "tubule_id": f"sha/series-0/candidate-{i}",
                "seq": i,
                "source_folder": "source_01",
                "source_name": "#1_a.tif",
                "source_sha256": "sha",
                "label": i,
                "recommended": 1,
                "flags": "",
                "source_bounds": json.dumps([0, 0, 10, 10]),
                "image": "x",
                "outline": "y",
                "display_scale": 1.0,
            }
            for i in (1, 2)
        ]
    )
    s.set_label(
        "sha/series-0/candidate-1",
        "staging",
        "Yeongseok",
        "VII",
        origin=PAPER,
        detail={"needs_check": False, "transcribed_by": "R"},
    )
    s.set_label("sha/series-0/candidate-2", "staging", "Adam", "I")  # web: ignored
    s.close()
    rows = data.load_labels(tmp_path)
    assert len(rows) == 1
    r = rows[0]
    assert (r["stage"], r["section"], r["needs_check"], r["transcribed_by"]) == (
        "VII",
        "#1_a",
        0,
        "R",
    )
    assert r["key"] == data.crop_key("sha/series-0/candidate-1")


def _fake_embeddings(tmp_path, names, n_crops=4):
    for i, name in enumerate(names):
        d = tmp_path / name
        d.mkdir()
        (d / "embed.json").write_text("{}")
        crop = np.repeat(np.arange(n_crops), 2)
        emb = np.full((len(crop), 3), i + 1, np.float16) * crop[:, None]
        frac = np.ones(len(crop), np.float32)
        np.savez(d / "s05_raw.npz", emb=emb, crop=crop, frac=frac)


def test_concatenated_features_and_model_selection(tmp_path):
    _fake_embeddings(tmp_path, ["a", "b"])
    x = evaluate.features(tmp_path, "a+b", "s05_raw", "mean", 4)
    assert x.shape == (4, 6)
    np.testing.assert_allclose(x[2], [2, 2, 2, 4, 4, 4])
    assert evaluate.available_models(tmp_path) == ["a", "b"]
    assert evaluate.available_models(tmp_path, ["a+b", "b"]) == ["a+b", "b"]
    with pytest.raises(SystemExit):
        evaluate.available_models(tmp_path, ["a+c"])


def test_research_only_extractors_are_flagged():
    assert "uni2-h" in embed.MODELS
    assert set(evaluate.research_only(["uni2-h+h-optimus-0", "dinov2-l"])) == {"uni2-h"}
    assert evaluate.research_only(["h-optimus-0", "dinov2-l"]) == {}
