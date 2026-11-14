"""P3 + P4 — trigonometric data, 5-fold CV x 2 initialisations per fold,
Trap [3,1,7] x {Adam, LM, LM-TR} x 4 initialisations, with interpretability
diagnostics (dead rules, uncovered samples, rules that ignore the noise
feature x1, mean antecedent coverage ratio rho of Eq. 14).

    python -m experiments.proposed.p3_trig_cv           # ~25 min CPU
    python -m experiments.proposed.p3_trig_cv --quick
"""
from __future__ import annotations

import argparse

import torch

from src.data import make_trig
from src.experiment import Config, run_experiment
from src.fuzzy_network import FuzzyNetwork
from src.membership import MF

from .common import INITS


@torch.no_grad()
def trig_diagnostics(model: FuzzyNetwork, x_tr: torch.Tensor, x_va: torch.Tensor) -> dict:
    L = model.layers[0]
    lo, hi = x_tr.min(dim=0).values, x_tr.max(dim=0).values
    grid = torch.stack([torch.linspace(float(lo[j]), float(hi[j]), 401) for j in range(L.in_dim)], dim=1)  # (G, M)
    # membership of every grid value in every antecedent: (K, M, G)
    lam = torch.stack([torch.stack([MF[L.ante_type](grid[:, j], L.antes[i, j]) for j in range(L.in_dim)])
                       for i in range(L.n_rules)])
    support_frac = (lam > 0).float().mean(dim=2)                  # (K, M): share of U_j inside supp(A_ij)
    rho = support_frac.min(dim=1).values                          # Eq. 14
    x1_ignored = int((lam[:, 0, :].mean(dim=1) > 0.99).sum())     # rules with x1 membership ~1 on all of U_1
    return {"rho_mean": float(rho.mean()), "x1_ignored_rules": x1_ignored}


def configs(epochs: int) -> list[Config]:
    return [Config(f"FN trap {opt} | {init}",
                   lambda s: FuzzyNetwork([[3, 1, 7]], "trap", "trap", seed=s),
                   optimizer=opt, epochs=epochs, init=init, diagnostics=trig_diagnostics,
                   tags={"model": "FN", "variant": "trap"})
            for opt in ("adam", "lm", "lm_tr") for init in INITS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    exp = "p3_trig_cv_quick" if args.quick else "p3_trig_cv"
    run_experiment(exp, configs(3 if args.quick else 100), make_trig(n=8000, seed=0),
                   protocols=("varied", "varied_b"), k=2 if args.quick else 5)


if __name__ == "__main__":
    main()
