"""One fuzzy rule-based layer of the FRBN.

Written for this project. Implements Section "Proposed model" of the reference
paper (Lukács et al., Sci. Rep. 16, 34429, 2026):

  rule i : IF x_1 is A_i1 AND ... AND x_M is A_iM
           THEN y_1 is C_i1 AND ... AND y_N is C_iN            (Eq. 1)

  firing strength   w_i = min_j lambda_ij(x_j)                   (Eq. 4 / 8)
  trapezoid cons. -> Mamdani + Center of Sums, closed form       (Eq. 5-6)
  Gaussian cons.  -> weighted average of consequent centres      (Eq. 9)

A layer with M inputs, N outputs and K rules holds an antecedent tensor of
shape (K, M, p_a) and a consequent tensor of shape (K, N, p_c). All N output
FRBSs share the same antecedents (MIMO layer).
"""
from __future__ import annotations

import torch
from torch import nn

from .membership import MF, N_PARAMS, sort_trap

EPS = torch.finfo(torch.float32).eps


def cos_defuzz_trap(w: torch.Tensor, cons: torch.Tensor) -> torch.Tensor:
    """Center of Sums defuzzification with clipped trapezoidal consequents, Eq. (5)-(6).

    w    : (B, K)        rule firing strengths in [0, 1]
    cons : (K, N, 4)     consequent trapezoids (sorted inside)
    returns (B, N)

    For a trapezoid (a, b, c, d) clipped at height w the area is
        D = w(d - a) + (w^2 / 2)(c + a - d - b)      (= Eq. 6 denominator / 2)
    and the first moment is N / 6 (Eq. 6 numerator), so y = N / (3 * D_eq6).
    """
    p = sort_trap(cons)                                   # (K, N, 4)
    a, b, c, d = (p[..., i].unsqueeze(0) for i in range(4))  # (1, K, N)
    w = w.unsqueeze(-1)                                   # (B, K, 1)
    num = (3 * w * (d ** 2 - a ** 2) * (1 - w)
           + 3 * w ** 2 * (c * d - a * b)
           + w ** 3 * (c - d + a - b) * (c - d - a + b))
    den = 2 * w * (d - a) + w ** 2 * (c + a - d - b)
    return num.sum(dim=1) / (3 * den.sum(dim=1) + EPS)


def weighted_avg_defuzz(w: torch.Tensor, centres: torch.Tensor) -> torch.Tensor:
    """Eq. (9): y_n = sum_i w_i mu_in / (sum_i w_i + eps).  w:(B,K), centres:(K,N) -> (B,N)."""
    return (w @ centres) / (w.sum(dim=1, keepdim=True) + EPS)


class FuzzyLayer(nn.Module):
    """Fuzzy rule-based layer [in_dim, out_dim, n_rules]."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        n_rules: int,
        ante: str = "trap",
        cons: str = "trap",
        init_range: tuple[float, float] | None = None,
        generator: torch.Generator | None = None,
    ):
        super().__init__()
        if ante not in MF or cons not in MF:
            raise ValueError("ante/cons must be 'trap' or 'gauss'")
        self.in_dim, self.out_dim, self.n_rules = in_dim, out_dim, n_rules
        self.ante_type, self.cons_type = ante, cons

        # Default ranges follow the authors' public code:
        # trapezoid breakpoints ~ U(-1, 2), Gaussian (mu, sigma) ~ U(-1, 1).
        def init(kind: str, shape: tuple[int, ...]) -> torch.Tensor:
            lo, hi = init_range or ((-1.0, 2.0) if kind == "trap" else (-1.0, 1.0))
            t = torch.rand(shape, generator=generator) * (hi - lo) + lo
            if kind == "trap":
                t = torch.sort(t, dim=-1).values
            return t

        self.antes = nn.Parameter(init(ante, (n_rules, in_dim, N_PARAMS[ante])))
        self.cons = nn.Parameter(init(cons, (n_rules, out_dim, N_PARAMS[cons])))

    # ------------------------------------------------------------------ core
    def firing_strengths(self, x: torch.Tensor) -> torch.Tensor:
        """Membership of every input in every antecedent, then min over inputs. (B, M) -> (B, K)."""
        lam = MF[self.ante_type](x.unsqueeze(1), self.antes)   # (B, K, M)
        return lam.min(dim=-1).values

    def defuzzify(self, w: torch.Tensor) -> torch.Tensor:
        if self.cons_type == "trap":
            return cos_defuzz_trap(w, self.cons)
        return weighted_avg_defuzz(w, self.cons[..., 0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.defuzzify(self.firing_strengths(x))

    # ------------------------------------------------------------ utilities
    def n_params(self, effective: bool = False) -> int:
        """Trainable parameter count. effective=True excludes the Gaussian
        consequent widths, which Eq. (9) never uses (they get zero gradient)."""
        cons_p = 1 if (effective and self.cons_type == "gauss") else N_PARAMS[self.cons_type]
        return self.n_rules * (self.in_dim * N_PARAMS[self.ante_type] + self.out_dim * cons_p)

    def extra_repr(self) -> str:
        return (f"[{self.in_dim}, {self.out_dim}, {self.n_rules}], "
                f"ante={self.ante_type}, cons={self.cons_type}")
