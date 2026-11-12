"""Coverage-aware initialisation of the first fuzzy layer (proposed extension).

Written for this project. Motivation (Checkpoint 3, E1b): with the paper's
random initialisation (breakpoints ~ U(-1, 2), Gaussian (mu, sigma) ~ U(-1, 1))
some regions of the input space are covered by no rule. A rule that never
fires gets zero gradient, so the optimiser cannot move it there: FN + LM-TR
converged in only 9/20 random initialisations on sinc.

Instead of random placement the antecedents are placed from the training data:

  "lhs"     Latin-hypercube style: for every input dimension the K rule centres
            are the (k + 0.5)/K quantiles of that input, shuffled independently
            per dimension. Guarantees that every 1-D marginal is covered evenly.
  "fcm"     Fuzzy C-Means clustering of the inputs; rule k <- cluster k.
  "fcm_xy"  Fuzzy C-Means in the product space [x, y] (classic product-space
            fuzzy model identification); centres are projected onto the inputs,
            so rules concentrate where the target function changes.

Widths come from the (fuzzy) spread of each cluster / the quantile spacing, and
the consequents are set to the firing-strength-weighted mean (and spread) of y
under the new antecedents, so the network output is already a sensible
piecewise approximation before training starts.

Only the first layer is initialised from data; it is the layer that sees the
raw inputs (all experiments in this project use single-layer FNs).
"""
from __future__ import annotations

import numpy as np
import torch

from .fuzzy_network import FuzzyNetwork

METHODS = ("random", "lhs", "fcm", "fcm_xy")


# ---------------------------------------------------------------- FCM
def fuzzy_c_means(X: np.ndarray, c: int, m: float = 2.0, max_iter: int = 150, tol: float = 1e-5,
                  seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Standard FCM (Bezdek). X: (n, d). Returns centres (c, d) and memberships U (n, c)."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    U = rng.random((n, c))
    U /= U.sum(axis=1, keepdims=True)
    p = 2.0 / (m - 1.0)
    for _ in range(max_iter):
        Um = U ** m
        V = (Um.T @ X) / Um.sum(axis=0)[:, None]                       # centres
        D = np.linalg.norm(X[:, None, :] - V[None, :, :], axis=2)      # (n, c)
        D = np.fmax(D, 1e-12)
        U_new = 1.0 / ((D[:, :, None] / D[:, None, :]) ** p).sum(axis=2)
        if np.abs(U_new - U).max() < tol:
            U = U_new
            break
        U = U_new
    Um = U ** m
    V = (Um.T @ X) / Um.sum(axis=0)[:, None]
    return V, U


# ------------------------------------------------------ antecedent placement
def _centres_and_widths(x: np.ndarray, y: np.ndarray, K: int, method: str, seed: int,
                        max_points: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """Rule centres (K, M) and widths (K, M) in the input space."""
    rng = np.random.default_rng(seed)
    n, M = x.shape
    lo, hi = x.min(axis=0), x.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    # With the min t-norm a sample must fall inside a rule's support in *every*
    # dimension. K rules in M dimensions correspond to a grid of K^(1/M) cells
    # per axis, so widths are scaled to that cell size (= span / K in 1-D).
    cell = span / K ** (1.0 / M)
    min_w = 0.5 * cell                                                  # never narrower than half a cell

    if method == "lhs":
        q = (np.arange(K) + 0.5) / K
        C = np.stack([rng.permutation(np.quantile(x[:, j], q)) for j in range(M)], axis=1)
        W = np.tile(cell, (K, 1))
        return C, np.maximum(W, min_w)

    idx = rng.choice(n, size=min(n, max_points), replace=False)
    xs, ys = x[idx], y[idx]
    # scale every column to [0, 1] so that x and y weigh equally in the distance
    if method == "fcm":
        Z, z_lo, z_span = xs, lo, span
    elif method == "fcm_xy":
        Z = np.hstack([xs, ys])
        z_lo, z_span = Z.min(axis=0), np.maximum(Z.max(axis=0) - Z.min(axis=0), 1e-6)
    else:
        raise ValueError(f"unknown method {method!r}")
    Zn = (Z - z_lo) / z_span
    V, U = fuzzy_c_means(Zn, K, seed=seed)
    V = V * z_span + z_lo
    C = V[:, :M]
    Um = U ** 2
    var = (Um[:, :, None] * (xs[:, None, :] - C[None, :, :]) ** 2).sum(axis=0) / Um.sum(axis=0)[:, None]
    W = np.sqrt(var)
    return C, np.maximum(W, min_w)


def _set_antecedents(layer, C: np.ndarray, W: np.ndarray) -> None:
    C_t, W_t = torch.as_tensor(C, dtype=torch.float32), torch.as_tensor(W, dtype=torch.float32)
    with torch.no_grad():
        if layer.ante_type == "gauss":
            layer.antes.copy_(torch.stack([C_t, W_t], dim=-1))
        else:  # trapezoid: core = centre +- 0.5 w, support = centre +- 2 w
            layer.antes.copy_(torch.stack([C_t - 2 * W_t, C_t - 0.5 * W_t,
                                           C_t + 0.5 * W_t, C_t + 2 * W_t], dim=-1))


def _set_consequents(layer, x: torch.Tensor, y: torch.Tensor) -> None:
    """Consequent of rule k <- firing-strength-weighted mean / spread of y."""
    with torch.no_grad():
        w = layer.firing_strengths(x)                                    # (n, K)
        wsum = w.sum(dim=0)                                              # (K,)
        y_range = (y.max(dim=0).values - y.min(dim=0).values).clamp_min(1e-6)  # (N,)
        mean = (w.T @ y) / wsum[:, None].clamp_min(1e-12)                # (K, N)
        var = (w.T @ y ** 2) / wsum[:, None].clamp_min(1e-12) - mean ** 2
        s = var.clamp_min(0).sqrt().clamp_min(0.02 * y_range)            # (K, N)
        # rules that fire nowhere get the global mean
        dead = wsum < 1e-8
        mean[dead] = y.mean(dim=0)
        if layer.cons_type == "gauss":
            layer.cons.copy_(torch.stack([mean, s], dim=-1))
        else:
            layer.cons.copy_(torch.stack([mean - 2 * s, mean - 0.5 * s, mean + 0.5 * s, mean + 2 * s], dim=-1))


def initialize(net: FuzzyNetwork, x: torch.Tensor, y: torch.Tensor, method: str = "fcm",
               seed: int = 0) -> FuzzyNetwork:
    """Re-initialise the first layer of `net` from training data (x, y). method='random' is a no-op."""
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    if method == "random":
        return net
    layer = net.layers[0]
    if len(net.layers) == 1:
        target = y
    else:  # deeper networks: only place the antecedents of layer 1, use x-only clustering
        target = torch.zeros(len(x), layer.out_dim)
        if method == "fcm_xy":
            method = "fcm"
    C, W = _centres_and_widths(x.numpy().astype(np.float64), target.numpy().astype(np.float64),
                               layer.n_rules, method, seed)
    _set_antecedents(layer, C, W)
    if len(net.layers) == 1:
        _set_consequents(layer, x, y)
    return net


# --------------------------------------------------------------- diagnostics
@torch.no_grad()
def dead_rules(net: FuzzyNetwork, x: torch.Tensor, thr: float = 1e-6) -> int:
    """Number of first-layer rules whose maximum firing strength on x is below thr."""
    w = net.layers[0].firing_strengths(x)
    return int((w.max(dim=0).values < thr).sum())


@torch.no_grad()
def uncovered_fraction(net: FuzzyNetwork, x: torch.Tensor, thr: float = 1e-3) -> float:
    """Fraction of samples on which no first-layer rule fires above thr."""
    w = net.layers[0].firing_strengths(x)
    return float((w.max(dim=1).values < thr).float().mean())
