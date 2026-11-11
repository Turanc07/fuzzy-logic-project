"""Training loop and k-fold cross-validation for FN and NN models.

Written for this project. One "epoch" = one pass over the training fold in
mini-batches; every mini-batch gets one optimizer step (one Adam step or one
LM iteration), as in the authors' public training code.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from torch import nn

from .data import Dataset, kfold_indices
from .optim_lm import LevenbergMarquardt

OPTIMIZERS = ("adam", "lm", "lm_tr")


@dataclass
class TrainResult:
    train_mse: float
    val_mse: float
    time_s: float
    n_params: int
    history: dict[str, list[float]] = field(default_factory=dict)
    model: nn.Module | None = None


@torch.no_grad()
def mse(model: nn.Module, x: torch.Tensor, y: torch.Tensor, batch: int = 50_000) -> float:
    total = 0.0
    for i in range(0, len(x), batch):
        total += torch.sum((model(x[i:i + batch]) - y[i:i + batch]) ** 2).item()
    return total / y.numel()


def train_model(
    model: nn.Module,
    x_tr: torch.Tensor, y_tr: torch.Tensor,
    x_val: torch.Tensor | None = None, y_val: torch.Tensor | None = None,
    optimizer: str = "adam",
    epochs: int = 100,
    batch_size: int | None = None,
    lr: float = 0.01,
    seed: int = 0,
    patience: int | None = None,
    verbose: bool = False,
) -> TrainResult:
    """Train `model` in place. batch_size=None -> full batch.

    patience: early stopping on validation MSE (None = off). The best-validation
    weights are restored when early stopping is on.
    """
    if optimizer not in OPTIMIZERS:
        raise ValueError(f"optimizer must be one of {OPTIMIZERS}")
    g = torch.Generator().manual_seed(seed)
    n = len(x_tr)
    bs = n if batch_size is None else min(batch_size, n)

    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        lm = LevenbergMarquardt(model, mode="trust_region" if optimizer == "lm_tr" else "standard")

    hist: dict[str, list[float]] = {"train_mse": [], "val_mse": []}
    best_val, best_state, bad_epochs = math.inf, None, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g) if bs < n else torch.arange(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb, yb = x_tr[idx], y_tr[idx]
            if optimizer == "adam":
                opt.zero_grad()
                loss = torch.mean((model(xb) - yb) ** 2)
                loss.backward()
                opt.step()
            else:
                lm.step(xb, yb)
        hist["train_mse"].append(mse(model, x_tr, y_tr))
        if x_val is not None:
            v = mse(model, x_val, y_val)
            hist["val_mse"].append(v)
            if patience is not None:
                if v < best_val - 1e-12:
                    best_val, bad_epochs = v, 0
                    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
                else:
                    bad_epochs += 1
                    if bad_epochs >= patience:
                        break
        if verbose and (ep % 10 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:4d} train {hist['train_mse'][-1]:.3e}"
                  + (f" val {hist['val_mse'][-1]:.3e}" if hist["val_mse"] else ""))
        if optimizer != "adam" and lm.state.converged:
            break
    elapsed = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)

    n_params = model.n_params() if hasattr(model, "n_params") else sum(p.numel() for p in model.parameters())
    return TrainResult(
        train_mse=mse(model, x_tr, y_tr),
        val_mse=mse(model, x_val, y_val) if x_val is not None else float("nan"),
        time_s=elapsed, n_params=n_params, history=hist, model=model,
    )


def cross_validate(
    make_model: Callable[[int], nn.Module],
    ds: Dataset,
    k: int = 5,
    split_seed: int = 0,
    keep_models: bool = False,
    **train_kwargs,
) -> list[TrainResult]:
    """k-fold CV. make_model(fold) must return a freshly initialised model."""
    results = []
    for fold, (tr, va) in enumerate(kfold_indices(len(ds), k=k, seed=split_seed)):
        tr_t, va_t = torch.as_tensor(tr), torch.as_tensor(va)
        model = make_model(fold)
        res = train_model(model, ds.x[tr_t], ds.y[tr_t], ds.x[va_t], ds.y[va_t],
                          seed=split_seed * 1000 + fold, **train_kwargs)
        if not keep_models:
            res.model = None
        results.append(res)
    return results


def summarize(results: list[TrainResult]) -> dict[str, float]:
    """mean +- std over folds and best fold, like the paper's tables."""
    val = np.array([r.val_mse for r in results])
    tr = np.array([r.train_mse for r in results])
    return {
        "val_mean": float(val.mean()), "val_std": float(val.std(ddof=0)), "val_best": float(val.min()),
        "train_mean": float(tr.mean()), "train_std": float(tr.std(ddof=0)), "train_best": float(tr.min()),
        "time_mean_s": float(np.mean([r.time_s for r in results])),
        "n_params": int(results[0].n_params),
        "folds": len(results),
    }
