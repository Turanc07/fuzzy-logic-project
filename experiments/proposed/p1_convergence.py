"""P1 — convergence rate over 20 initialisations (sinc, Gauss [1,1,20], fold 0).

Same protocol as E1b, now for every initialisation method. Paired by seed:
seed s gives the same random draw to every method before the data-driven
re-initialisation, and the same mini-batch order.

    python -m experiments.proposed.p1_convergence           # ~30 min CPU
    python -m experiments.proposed.p1_convergence --quick
"""
from __future__ import annotations

import argparse

import pandas as pd

from src.data import kfold_indices, make_sinc
from src.experiment import TABLES, Config, run_one
from src.fuzzy_network import FuzzyNetwork

from .common import INITS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    seeds, epochs = (range(2), 3) if args.quick else (range(20), 100)
    exp = "p1_convergence_quick" if args.quick else "p1_convergence"

    ds = make_sinc(n=20_000)
    tr, va = kfold_indices(len(ds), 5, 0)[0]
    rows = []
    for opt in ("lm_tr", "lm", "adam"):
        for init in INITS:
            cfg = Config(f"FN gauss {opt} | {init}",
                         lambda s: FuzzyNetwork([[1, 1, 20]], "gauss", "gauss", seed=s),
                         optimizer=opt, epochs=epochs, init=init,
                         tags={"model": "FN", "variant": "gauss"})
            for s in seeds:
                row, _, _ = run_one(cfg, ds, tr, va, init_seed=s, train_seed=0)
                rows.append(row)
                print(f"[{exp}] {cfg.name:28s} seed {s:2d}: val {row['val_mse']:.3e} "
                      f"(start {row['init_val_mse']:.1e})", flush=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / f"{exp}.csv", index=False)


if __name__ == "__main__":
    main()
