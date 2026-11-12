"""Tests for the coverage-aware initialisation (proposed extension)."""
import numpy as np
import pytest
import torch

from src.data import make_sinc, make_trig
from src.fuzzy_network import FuzzyNetwork
from src.initialization import dead_rules, fuzzy_c_means, initialize, uncovered_fraction


def test_fcm_finds_separated_clusters():
    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.05, (100, 2)), rng.normal(1, 0.05, (100, 2))])
    V, U = fuzzy_c_means(X, 2, seed=0)
    assert np.allclose(sorted(V[:, 0]), [0, 1], atol=0.05)
    assert np.allclose(U.sum(axis=1), 1.0)


@pytest.mark.parametrize("method", ["lhs", "fcm", "fcm_xy"])
@pytest.mark.parametrize("mf", ["gauss", "trap"])
def test_initialization_covers_input_space(method, mf):
    ds = make_trig(n=2000, seed=0)
    net = initialize(FuzzyNetwork([[3, 1, 7]], mf, mf, seed=0), ds.x, ds.y, method=method, seed=0)
    assert dead_rules(net, ds.x) == 0
    assert uncovered_fraction(net, ds.x) < 0.05
    assert torch.isfinite(net(ds.x)).all()


@pytest.mark.parametrize("method", ["lhs", "fcm", "fcm_xy"])
def test_initialized_model_starts_better_than_predicting_mean(method):
    ds = make_sinc(n=4000)
    net = initialize(FuzzyNetwork([[1, 1, 20]], "gauss", "gauss", seed=0), ds.x, ds.y, method=method)
    with torch.no_grad():
        mse = torch.mean((net(ds.x) - ds.y) ** 2).item()
    assert mse < torch.var(ds.y).item()


def test_random_is_noop_and_reproducible():
    ds = make_sinc(n=500)
    a = FuzzyNetwork([[1, 1, 5]], "gauss", "gauss", seed=3)
    before = a.layers[0].antes.clone()
    initialize(a, ds.x, ds.y, method="random")
    assert torch.equal(before, a.layers[0].antes)
    b = initialize(FuzzyNetwork([[1, 1, 5]], seed=0), ds.x, ds.y, "fcm_xy", seed=1)
    c = initialize(FuzzyNetwork([[1, 1, 5]], seed=0), ds.x, ds.y, "fcm_xy", seed=1)
    assert torch.equal(b.layers[0].antes, c.layers[0].antes)


def test_random_init_can_leave_rules_dead():
    """Documents the motivating problem: the paper's random init yields dead rules on [0,1] data."""
    ds = make_trig(n=2000, seed=0)
    dead = [dead_rules(FuzzyNetwork([[3, 1, 7]], "trap", "trap", seed=s), ds.x) for s in range(20)]
    assert max(dead) >= 1
