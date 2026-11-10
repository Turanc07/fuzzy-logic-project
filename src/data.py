"""Synthetic datasets used in the reproduction experiments.

Written for this project (not copied from the official repository).
Dataset definitions follow the reference paper [1] and, where the paper is
silent, the authors' public notebook (see docs/checkpoint2_uygulama_plani.md).

[1] Lukács et al., "Fuzzy rule-based network ...", Sci. Rep. 16, 34429 (2026).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Make numpy and torch random generators deterministic."""
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class Dataset:
    x: torch.Tensor  # shape (n, in_dim), float32
    y: torch.Tensor  # shape (n, out_dim), float32
    name: str

    def __len__(self) -> int:
        return self.x.shape[0]


def make_sinc(n: int = 20_000, freq: float = 10.0) -> Dataset:
    """1-D sinc benchmark: y = sinc(freq * x) = sin(pi*freq*x) / (pi*freq*x), x in [-1, 1].

    The paper reports 20 000 points; the official notebook uses an evenly
    spaced grid on [-1, 1] with torch.sinc(10 * x).
    """
    x = torch.linspace(-1.0, 1.0, n, dtype=torch.float32).unsqueeze(1)
    y = torch.sinc(freq * x)
    return Dataset(x=x, y=y, name="sinc")


def trig_target(x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
    """Raw target of Eq. (20): sin(2*pi*x2) * cos(2*pi*x3), range [-1, 1]."""
    return torch.sin(2 * torch.pi * x2) * torch.cos(2 * torch.pi * x3)


def make_trig(
    n: int = 8_000,
    seed: int = 0,
    nuisance: str = "uniform",
    noise_std: float = 0.1,
    normalize_y: bool = True,
) -> Dataset:
    """3-D trigonometric benchmark, Eq. (20) of the paper.

    x2, x3 ~ U[0, 1] carry the signal; x1 is a nuisance feature:
      - nuisance="uniform":    x1 ~ U[0, 1] (pure noise, main experiment)
      - nuisance="correlated": x1 = 0.3 * x2 + eps, eps ~ N(0, noise_std^2) (ablation, Table 4)
    The paper states that inputs and y are normalized to [0, 1]; y is mapped
    with (y + 1) / 2 when normalize_y is True. For the correlated case x1 is
    min-max scaled to [0, 1].
    """
    g = torch.Generator().manual_seed(seed)
    x2 = torch.rand(n, generator=g)
    x3 = torch.rand(n, generator=g)
    if nuisance == "uniform":
        x1 = torch.rand(n, generator=g)
    elif nuisance == "correlated":
        x1 = 0.3 * x2 + noise_std * torch.randn(n, generator=g)
        x1 = (x1 - x1.min()) / (x1.max() - x1.min())
    else:
        raise ValueError("nuisance must be 'uniform' or 'correlated'")

    y = trig_target(x2, x3)
    if normalize_y:
        y = (y + 1.0) / 2.0
    x = torch.stack([x1, x2, x3], dim=1).float()
    return Dataset(x=x, y=y.unsqueeze(1).float(), name=f"trig_{nuisance}")


def kfold_indices(n: int, k: int = 5, seed: int = 0) -> list[tuple[np.ndarray, np.ndarray]]:
    """Shuffled k-fold split -> list of (train_idx, val_idx)."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    folds = np.array_split(perm, k)
    return [(np.concatenate(folds[:i] + folds[i + 1:]), folds[i]) for i in range(k)]


if __name__ == "__main__":
    for ds in (make_sinc(), make_trig(), make_trig(nuisance="correlated", noise_std=0.5)):
        print(f"{ds.name:18s} n={len(ds):6d} x{tuple(ds.x.shape)} "
              f"y[min={ds.y.min():.3f}, max={ds.y.max():.3f}]")
