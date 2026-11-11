"""Tests for the NN baseline, the LM optimizers and the training loop."""
import pytest
import torch
from torch import nn

from src.data import make_sinc
from src.fuzzy_network import FuzzyNetwork
from src.nn_baseline import MLP
from src.optim_lm import LevenbergMarquardt
from src.train import cross_validate, summarize, train_model


def test_mlp_param_count_matches_paper():
    assert MLP(1, 20, 1, "tanh").n_params() == 61          # Table 1b


@pytest.mark.parametrize("mode", ["trust_region", "standard"])
def test_lm_solves_linear_least_squares(mode):
    """A linear model is fitted exactly by LM in a few iterations."""
    torch.manual_seed(0)
    x = torch.rand(200, 3)
    y = x @ torch.tensor([[1.5], [-2.0], [0.5]]) + 0.3
    model = nn.Linear(3, 1)
    lm = LevenbergMarquardt(model, mode=mode, gamma_init=1.0)
    for _ in range(30):
        loss = lm.step(x, y)
    assert loss < 1e-8


@pytest.mark.parametrize("mode", ["trust_region", "standard"])
def test_lm_error_never_increases(mode):
    """Eq. (19): a step is accepted only if it lowers the error."""
    ds = make_sinc(n=400)
    net = FuzzyNetwork([[1, 1, 6]], ante="gauss", cons="gauss", seed=1)
    lm = LevenbergMarquardt(net, mode=mode)
    prev = float("inf")
    for _ in range(15):
        loss = lm.step(ds.x, ds.y)
        assert loss <= prev + 1e-12
        prev = loss


@pytest.mark.parametrize("optimizer", ["adam", "lm", "lm_tr"])
def test_train_model_reduces_error(optimizer):
    ds = make_sinc(n=500)
    net = FuzzyNetwork([[1, 1, 8]], ante="gauss", cons="gauss", seed=0)
    with torch.no_grad():
        start = torch.mean((net(ds.x) - ds.y) ** 2).item()
    res = train_model(net, ds.x, ds.y, ds.x, ds.y, optimizer=optimizer, epochs=20)
    assert res.train_mse < start and len(res.history["train_mse"]) >= 1


def test_cross_validate_and_summary():
    ds = make_sinc(n=300)
    res = cross_validate(lambda f: FuzzyNetwork([[1, 1, 4]], ante="gauss", cons="gauss", seed=f),
                         ds, k=3, optimizer="adam", epochs=5)
    s = summarize(res)
    assert s["folds"] == 3 and s["n_params"] == 16
    assert s["val_best"] <= s["val_mean"]


def test_early_stopping_restores_best():
    ds = make_sinc(n=300)
    net = FuzzyNetwork([[1, 1, 4]], ante="gauss", cons="gauss", seed=0)
    res = train_model(net, ds.x, ds.y, ds.x, ds.y, optimizer="adam", epochs=200, patience=3)
    assert res.val_mse == pytest.approx(min(res.history["val_mse"]), rel=1e-6)
