import csv
import json
import sqlite3

import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")
tifffile = pytest.importorskip("tifffile")
Image = pytest.importorskip("PIL.Image")

from eyespy_spermatogenesis.staging import experiments as ex  # noqa: E402
from eyespy_spermatogenesis.staging import eyespy as es  # noqa: E402
from eyespy_spermatogenesis.staging.round1.data import COLUMNS, STAGES  # noqa: E402

K = len(STAGES)


def cyc_x(y, rng, noise=0.15, dim=4):
    a = 2 * np.pi * np.asarray(y) / K
    x = np.c_[np.cos(a), np.sin(a), rng.normal(size=(len(y), dim - 2))]
    return x + rng.normal(0, noise, x.shape)


@pytest.mark.parametrize("balanced", [False, True])
def test_saved_head_matches_experiment_head(balanced):
    rng = np.random.default_rng(1)
    y = np.r_[np.repeat(np.arange(K), 6), [0] * 20]
    x = cyc_x(y, rng)
    h = es.fit_head(x, y, 10.0, balanced)
    np.testing.assert_allclose(
        es.head_vectors(h, x), ex.fit_vectors(x, y, x, 10.0, balanced), atol=1e-8
    )


def test_checkpoint_round_trip_with_and_without_smoothing(tmp_path):
    rng = np.random.default_rng(2)
    y = np.repeat(np.arange(K), 5)
    feats = {m: cyc_x(y, rng) for m in es.MEMBERS}
    heads = [es.fit_head(feats[m], y, 1.0, False) for m in es.MEMBERS]
    model = es.EyeSpyModel(heads, es.MEMBERS)
    model.save(tmp_path / "h.npz", alpha=[1.0, 1.0, 1.0])
    centres = rng.uniform(0, 100, (len(y), 2))
    sections = np.array(["a"] * len(y))
    plain = es.EyeSpyModel.load(tmp_path / "h.npz")
    np.testing.assert_array_equal(plain.predict(feats), model.predict(feats))
    sm = es.EyeSpyModel.load(tmp_path / "h.npz", k=4, lam=0.3)
    expect = ex.smooth(model.vectors(feats), ex.neighbours(centres, sections, 4), 0.3)
    np.testing.assert_allclose(sm.vectors(feats, centres, sections), expect)


# ---------------------------------------------------------------- end to end
def make_source(root, sections, origin, rng, run_parent):
    data, emb = root / "data", root / "emb"
    data.mkdir(parents=True)
    rows, y = [], []
    for si, sec in enumerate(sections):
        folder = f"source_{origin}_{si}"
        rd = run_parent / folder
        rd.mkdir(parents=True)
        labels = np.zeros((40, 40), np.int32)
        for k in range(K * 2):
            r, c = divmod(k, 6)
            labels[r * 6 : r * 6 + 5, c * 6 : c * 6 + 5] = k + 1
            stage = STAGES[k % K]
            rows.append(
                {
                    "key": f"{origin}{si}_{k}",
                    "tubule_id": f"{sec}:{k}",
                    "section": sec,
                    "source_folder": folder,
                    "label": k + 1,
                    "recommended": 1,
                    "stage": stage,
                    "needs_check": 0,
                    "transcribed_by": "",
                    "source_bounds": json.dumps(
                        [c * 60, r * 60, c * 60 + 50, r * 60 + 50]
                    ),
                }
            )
            y.append(k % K)
        Image.fromarray(np.full((40, 40, 3), 200, np.uint8)).save(rd / "overview.png")
        tifffile.imwrite(rd / "overview_labels.tif", labels)
    with open(data / "labels.csv", "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        w.writerows(rows)
    (data / "data.json").write_text(
        json.dumps({"origin": origin, "annotator": "Tester", "rows": len(rows)})
    )
    n = len(rows)
    for spec in es.MEMBERS:
        model, view, _ = spec.split(":")
        (emb / model).mkdir(parents=True, exist_ok=True)
        np.savez(
            emb / model / f"{view}.npz",
            emb=cyc_x(y, rng).astype(np.float16),
            crop=np.arange(n),
            frac=np.ones(n),
        )
    return data, emb


@pytest.fixture
def setup(tmp_path):
    rng = np.random.default_rng(3)
    ws = tmp_path / "ws"
    runs = tmp_path / "runs"
    ws.mkdir()
    (ws / "workspace.json").write_text(json.dumps({"run_parent": str(runs)}))
    paper = make_source(
        tmp_path / "paper", ["#1_a", "#2_b", "#3_c"], "paper", rng, runs
    )
    web = make_source(tmp_path / "web", ["#4_d"], "web", rng, runs)
    # The page's tubule table (the index maps tubule ids to candidate labels).
    db = sqlite3.connect(ws / "labels.sqlite")
    db.execute("CREATE TABLE tubules (tubule_id, source_name, source_folder, label)")
    for data, _ in (paper, web):
        with open(data / "labels.csv", newline="") as f:
            db.executemany(
                "INSERT INTO tubules VALUES (?, ?, ?, ?)",
                [
                    (
                        r["tubule_id"],
                        r["section"] + ".tif",
                        r["source_folder"],
                        r["label"],
                    )
                    for r in csv.DictReader(f)
                ],
            )
    db.commit()
    db.close()
    return ws, tmp_path / "ckpt", paper, web


def test_train_writes_registry_panels_and_never_overwrites(setup):
    ws, ckpt, paper, web = setup
    es.train(1, [paper], ws, ckpt, samples=["#1_a", "#3_c"])
    reg = json.loads((ckpt / "registry.json").read_text())
    names = [m["name"] for m in reg["models"]]
    assert names[0] == "EyeSpy_v1" and names[2] == "EyeSpy_v1 - Class weighted"
    assert names[1].startswith("EyeSpy_v1 - Smoothed (k=")
    plain = reg["models"][0]
    assert plain["settings"]["smoothing"] == {"on": False}
    assert reg["models"][1]["checkpoint"] == plain["checkpoint"]  # same heads
    assert plain["training_data"]["n"] == 3 * 2 * K
    assert plain["evaluation"]["web_sections"] is None
    for f in ("heads.npz", "heads_class_weighted.npz", "predictions.npz"):
        assert (ckpt / "eyespy_v1" / f).exists()
    idx = json.loads((ws / "review/eyespy/index.json").read_text())
    assert [s["section"] for s in idx["samples"]] == ["#1_a", "#3_c"]
    for m in idx["models"]:
        for p in m["panels"].values():
            assert (ws / "review/eyespy" / p["stage"]).exists()
            assert (ws / "review/eyespy" / p["agree"]).exists()
    assert (ws / "review/eyespy/truth/1_a.jpg").exists()
    # Click data: per-tubule stages keyed by candidate label, plus click maps.
    s0 = idx["samples"][0]
    assert len(s0["truth_stages"]) == 2 * K and s0["truth_stages"]["1"] == STAGES[0]
    p0 = idx["models"][0]["panels"]["#1_a"]["stages"]
    assert set(p0) == set(s0["truth_stages"])
    assert (ws / "review/maps/1_a.png").exists()
    with pytest.raises(SystemExit):
        es.train(1, [paper], ws, ckpt)

    # v2 adds on-screen labels; compared with v1 on the shared paper tubules.
    es.train(2, [paper, web], ws, ckpt)
    reg = json.loads((ckpt / "registry.json").read_text())
    assert len(reg["models"]) == 6
    v2 = reg["models"][3]
    assert v2["name"] == "EyeSpy_v2"
    assert v2["evaluation"]["web_sections"]["n"] == 2 * K
    cmp = v2["evaluation"]["vs_previous_version"]
    assert cmp["against"] == "EyeSpy_v1 (plain)" and cmp["n"] == 3 * 2 * K
    assert reg["samples"] == ["#1_a", "#3_c"]  # kept from v1
    # The saved checkpoint loads and predicts.
    m = es.EyeSpyModel.load(ckpt / v2["checkpoint"])
    assert m.members == list(es.MEMBERS)


def test_duplicate_tubule_across_sources_is_refused(setup):
    ws, ckpt, paper, _ = setup
    with pytest.raises(SystemExit):
        es.Dataset([paper, paper])


def test_experiment_shift_and_weight_are_recorded(setup):
    ws, ckpt, paper, web = setup
    es.train(1, [paper], ws, ckpt, samples=["#1_a"])
    es.train(
        2, [paper, (*web, {"shift": "-1", "weight": "3"})], ws, ckpt, experiment="x"
    )
    reg = json.loads((ckpt / "registry.json").read_text())
    e = reg["models"][-1]
    assert e["name"] == "EyeSpy_v2 - Experiment: x" and e["role"] == "experiment"
    assert len(reg["models"]) == 4  # one plain model per experiment
    src = e["training_data"]["sources"][1]
    assert src["label_shift"] == -1 and src["weight"] == 3
    assert e["evaluation"]["web_sections_original_labels"] is not None
    assert e["evaluation"]["vs_previous_version"]["against"] == "EyeSpy_v1 (plain)"
    # An experiment never becomes "the previous version" of a real one.
    assert es.previous_primary(reg, 3)["name"] == "EyeSpy_v1"


def test_shifted_dataset_moves_labels_cyclically(setup):
    _, _, paper, web = setup
    ds = es.Dataset([paper, (*web, {"shift": "-1"})])
    w = ds.origin == "web"
    np.testing.assert_array_equal(ds.y[w], (ds.y_orig[w] - 1) % K)
    np.testing.assert_array_equal(ds.y[~w], ds.y_orig[~w])
    assert ds.w is None
    ds2 = es.Dataset([paper, (*web, {"weight": "2"})])
    assert (ds2.w[ds2.origin == "web"] == 2).all() and (
        ds2.w[ds2.origin == "paper"] == 1
    ).all()


def test_section_maps_for_every_workspace_section(setup):
    from eyespy_spermatogenesis.staging.annotate import clickmaps

    ws, _, _, _ = setup
    clickmaps.write_sections(ws)
    for sec in ("1_a", "2_b", "3_c", "4_d"):
        m = np.asarray(Image.open(ws / "review/maps" / f"{sec}.png"))
        ov = Image.open(ws / "review/sections" / f"{sec}.jpg")
        assert m.shape[:2] == (ov.height, ov.width)
        lab = m[..., 0].astype(int) + 256 * m[..., 1].astype(int)
        assert set(np.unique(lab)) == set(range(2 * K + 1))  # background + labels
