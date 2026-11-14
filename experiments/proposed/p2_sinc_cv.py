"""P2 — sinc, 5-fold CV, {Gauss, Trap} x {Adam, LM, LM-TR} x 4 initialisations.

Uses the "varied" protocol (a different initialisation in every fold).

    python -m experiments.proposed.p2_sinc_cv           # ~25 min CPU
    python -m experiments.proposed.p2_sinc_cv --quick
"""
from __future__ import annotations

import argparse

import torch

from src.data import make_sinc
from src.experiment import Config, run_experiment
from src.fuzzy_network import FuzzyNetwork

from .common import INITS


def configs(epochs: int) -> list[Config]:
    return [Config(f"FN {mf} {opt} | {init}",
                   lambda s, mf=mf: FuzzyNetwork([[1, 1, 20]], mf, mf, seed=s),
                   optimizer=opt, epochs=epochs, init=init,
                   tags={"model": "FN", "variant": mf})
            for mf in ("gauss", "trap") for opt in ("adam", "lm", "lm_tr") for init in INITS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    exp = "p2_sinc_cv_quick" if args.quick else "p2_sinc_cv"
    run_experiment(exp, configs(3 if args.quick else 100), make_sinc(n=20_000),
                   protocols=("varied",), k=2 if args.quick else 5,
                   grid=torch.linspace(-1, 1, 1001).unsqueeze(1))


if __name__ == "__main__":
    main()
