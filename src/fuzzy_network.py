"""Fuzzy Rule-Based Network: a cascade of fuzzy layers, y = f_L(...f_1(x)) (Eq. 10).

Written for this project, following the reference paper
(Lukács et al., Sci. Rep. 16, 34429, 2026). No activation functions are
placed between layers: the membership functions provide the non-linearity.
"""
from __future__ import annotations

import torch
from torch import nn

from .fuzzy_layer import FuzzyLayer
from .membership import sort_trap


class FuzzyNetwork(nn.Module):
    """Example: FuzzyNetwork([[3, 1, 7]]) or FuzzyNetwork([[3, 3, 4], [3, 1, 5]], ante="gauss", cons="gauss")."""

    def __init__(
        self,
        layers: list[list[int]],
        ante: str = "trap",
        cons: str = "trap",
        init_range: tuple[float, float] | None = None,
        seed: int | None = None,
    ):
        super().__init__()
        if not layers:
            raise ValueError("at least one layer [in_dim, out_dim, n_rules] is required")
        for prev, nxt in zip(layers, layers[1:]):
            if prev[1] != nxt[0]:
                raise ValueError(f"layer output dim {prev[1]} != next input dim {nxt[0]}")
        g = torch.Generator().manual_seed(seed) if seed is not None else None
        self.layers = nn.ModuleList(
            FuzzyLayer(i, o, k, ante=ante, cons=cons, init_range=init_range, generator=g)
            for i, o, k in layers
        )
        self.architecture = [list(map(int, l)) for l in layers]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def n_params(self, effective: bool = False) -> int:
        return sum(l.n_params(effective) for l in self.layers)

    @torch.no_grad()
    def rules(self, layer: int = 0, decimals: int = 3) -> list[str]:
        """Human-readable rules of one layer with their MF parameters."""
        L = self.layers[layer]
        A = sort_trap(L.antes) if L.ante_type == "trap" else L.antes
        C = sort_trap(L.cons) if L.cons_type == "trap" else L.cons
        fmt = lambda t: "(" + ", ".join(f"{v:.{decimals}f}" for v in t.tolist()) + ")"
        out = []
        for i in range(L.n_rules):
            ifs = " AND ".join(f"x{j + 1} is {L.ante_type}{fmt(A[i, j])}" for j in range(L.in_dim))
            thens = " AND ".join(f"y{n + 1} is {L.cons_type}{fmt(C[i, n])}" for n in range(L.out_dim))
            out.append(f"R{i}: IF {ifs} THEN {thens}")
        return out

    def summary(self) -> str:
        lines = [f"FuzzyNetwork {self.architecture}"]
        for i, l in enumerate(self.layers):
            lines.append(f"  layer {i}: {l.extra_repr()}, params={l.n_params()}")
        lines.append(f"  total params={self.n_params()} (effective={self.n_params(effective=True)})")
        return "\n".join(lines)
