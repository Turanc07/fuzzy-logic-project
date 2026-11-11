"""Membership functions of the Fuzzy Rule-Based Network.

Written for this project. Formulas follow Eqs. (2)-(3) and (7) of the
reference paper (Lukács et al., Sci. Rep. 16, 34429, 2026).

All functions broadcast: `x` and the parameter tensors only need compatible
shapes; the last dimension of the parameter tensor holds the MF parameters.
"""
from __future__ import annotations

import torch

EPS = 1e-8


def sort_trap(params: torch.Tensor) -> torch.Tensor:
    """Enforce a <= b <= c <= d on the last dimension (differentiable)."""
    return torch.sort(params, dim=-1).values


def trapmf(x: torch.Tensor, params: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Trapezoidal membership degree, Eq. (2)-(3).

    params[..., :] = (a, b, c, d); they are sorted first, so any real values
    are a valid trapezoid. Core = [b, c], support = (a, d).
    """
    p = sort_trap(params)
    a, b, c, d = p[..., 0], p[..., 1], p[..., 2], p[..., 3]
    rise = (x - a) / (b - a + eps)
    fall = (d - x) / (d - c + eps)
    # min(rise, 1, fall) clipped to [0, 1] is exactly the piecewise definition:
    # 0 outside (a, d), linear on the slopes, 1 on the core.
    return torch.clamp(torch.minimum(torch.minimum(rise, fall), torch.ones_like(rise)), min=0.0)


def gaussmf(x: torch.Tensor, params: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Gaussian membership degree, Eq. (7). params[..., :] = (mu, sigma); sigma >= eps."""
    mu = params[..., 0]
    sigma = torch.clamp(params[..., 1].abs(), min=eps)
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


MF = {"trap": trapmf, "gauss": gaussmf}
N_PARAMS = {"trap": 4, "gauss": 2}


def support(params: torch.Tensor, kind: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Support interval used by the coverage ratio (Eq. 14).

    Trapezoid: (a, d). Gaussian: mu +- 3 sigma (as defined in the paper).
    """
    if kind == "trap":
        p = sort_trap(params)
        return p[..., 0], p[..., 3]
    mu, sigma = params[..., 0], params[..., 1].abs()
    return mu - 3 * sigma, mu + 3 * sigma


def core_center(params: torch.Tensor, kind: str) -> torch.Tensor:
    """Center of the core: (b + c) / 2 for trapezoids, mu for Gaussians."""
    if kind == "trap":
        p = sort_trap(params)
        return (p[..., 1] + p[..., 2]) / 2
    return params[..., 0]
