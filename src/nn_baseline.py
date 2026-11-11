"""Neural-network baseline used for comparison with the FN (paper Table 1b).

Written for this project. Single hidden layer MLP; for 1 input, 20 hidden
neurons and 1 output it has 1*20 + 20 + 20*1 + 1 = 61 parameters, the same
count the paper reports.
"""
from __future__ import annotations

import torch
from torch import nn

ACTIVATIONS = {"tanh": nn.Tanh, "relu": nn.ReLU}


class MLP(nn.Module):
    def __init__(self, in_dim: int = 1, hidden: int = 20, out_dim: int = 1,
                 activation: str = "tanh", seed: int | None = None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), ACTIVATIONS[activation](),
                                 nn.Linear(hidden, out_dim))
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def n_params(self, effective: bool = False) -> int:
        return sum(p.numel() for p in self.parameters())
