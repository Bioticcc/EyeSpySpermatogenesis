import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from eyespy_spermatogenesis.staging import experiments as ex  # noqa: E402
from eyespy_spermatogenesis.staging.round1 import evaluate  # noqa: E402


def test_stage_weights_balance_classes():
    y = np.array([0, 0, 0, 1])
    w = ex.stage_weights(y)
    assert w.mean() == pytest.approx(1)
    assert w[:3].sum() == pytest.approx(w[3])


def test_neighbours_stay_in_section_and_exclude_self():
    c = np.array([[0, 0], [1, 0], [2, 0], [100, 0], [0, 0.5]], float)
    g = np.array(["a", "a", "a", "a", "b"])
    nb = ex.neighbours(c, g, 2)
    assert sorted(nb[0]) == [1, 2]
    assert all(g[j] == g[i] for i, n in enumerate(nb) for j in n)
    assert all(i not in n for i, n in enumerate(nb))
    assert len(nb[4]) == 0  # alone in its section


def test_smooth_off_is_identity_and_blends_when_on():
    v = np.array([[1.0, 0], [0, 1.0]])
    nb = [np.array([1]), np.array([0])]
    assert ex.smooth(v, nb, 0.0) is v
    np.testing.assert_allclose(ex.smooth(v, nb, 0.5), [[0.5, 0.5], [0.5, 0.5]])


def test_unweighted_member_matches_round1_head():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(evaluate.K), 12)
    x = np.c_[np.cos(2 * np.pi * y / evaluate.K), np.sin(2 * np.pi * y / evaluate.K)]
    x = x + rng.normal(0, 0.1, x.shape)
    v = ex.fit_vectors(x, y, x, 1.0, balanced=False)
    np.testing.assert_array_equal(
        ex.decode(v), evaluate.Cyclic(1.0).fit(x, y).predict(x)
    )


def test_weight_grid_is_convex():
    for n in (1, 2, 3):
        for w in ex.weight_grid(n):
            assert len(w) == n and sum(w) == pytest.approx(1) and min(w) >= 0
