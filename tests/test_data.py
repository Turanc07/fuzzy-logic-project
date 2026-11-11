import numpy as np
import torch

from src.data import kfold_indices, make_sinc, make_trig


def test_sinc_shape_and_peak():
    ds = make_sinc(n=1001)
    assert ds.x.shape == (1001, 1) and ds.y.shape == (1001, 1)
    assert torch.isclose(ds.y[500], torch.tensor([1.0]))  # sinc(0) = 1
    assert ds.x.min() == -1 and ds.x.max() == 1


def test_trig_range_and_reproducible():
    a, b = make_trig(n=500, seed=1), make_trig(n=500, seed=1)
    assert torch.equal(a.x, b.x) and torch.equal(a.y, b.y)
    assert a.x.shape == (500, 3)
    assert a.y.min() >= 0 and a.y.max() <= 1
    assert a.x.min() >= 0 and a.x.max() <= 1


def test_correlated_nuisance_is_correlated():
    ds = make_trig(n=4000, nuisance="correlated", noise_std=0.05)
    r = np.corrcoef(ds.x[:, 0].numpy(), ds.x[:, 1].numpy())[0, 1]
    assert r > 0.8


def test_kfold_partitions_everything_once():
    folds = kfold_indices(103, k=5)
    val = np.concatenate([v for _, v in folds])
    assert sorted(val.tolist()) == list(range(103))
    for tr, v in folds:
        assert len(np.intersect1d(tr, v)) == 0 and len(tr) + len(v) == 103
