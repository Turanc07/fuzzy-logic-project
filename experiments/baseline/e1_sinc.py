"""E1 — sinc function approximation: FN vs NN with Adam / LM / LM trust-region.

Reproduces Table 1 and Figs. 4-5 of the reference paper.

    python -m experiments.baseline.e1_sinc           # full run (5 folds x 2 protocols, ~25 min CPU)
    python -m experiments.baseline.e1_sinc --quick   # smoke test (2 folds, 5 epochs)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from src.data import make_sinc
from src.experiment import (TABLES, Config, plot_curves, plot_predictions,
                            run_experiment, summary_markdown)
from src.fuzzy_network import FuzzyNetwork
from src.nn_baseline import MLP

# Paper, Table 1 (GradOnly rows): validation MSE mean, std, best fold.
PAPER = {
    "FN gauss adam":  (3.35e-4, 2.60e-4, 2.05e-5),
    "FN gauss lm":    (6.05e-3, 4.00e-3, 8.98e-8),
    "FN gauss lm_tr": (8.07e-7, 5.78e-7, 6.20e-8),
    "FN trap adam":   (2.24e-3, 8.34e-4, 1.46e-3),
    "FN trap lm":     (6.65e-3, 5.87e-3, 5.91e-5),
    "FN trap lm_tr":  (8.33e-3, 6.60e-3, 1.91e-3),
    "NN tanh adam":   (1.64e-2, 9.70e-3, 7.08e-3),
    "NN tanh lm":     (3.26e-5, 4.75e-5, 1.42e-6),
    "NN tanh lm_tr":  (2.82e-7, 3.82e-8, 2.39e-7),
    "NN relu adam":   (1.18e-2, 1.18e-2, 5.32e-3),
    "NN relu lm":     (1.05e-2, 1.65e-2, 2.09e-3),
    "NN relu lm_tr":  (2.77e-3, 4.72e-4, 1.93e-3),
}


def configs(epochs: int) -> list[Config]:
    out = []
    for mf in ("gauss", "trap"):
        for opt in ("adam", "lm", "lm_tr"):
            name = f"FN {mf} {opt}"
            out.append(Config(name, lambda s, mf=mf: FuzzyNetwork([[1, 1, 20]], ante=mf, cons=mf, seed=s),
                              optimizer=opt, epochs=epochs, tags={"model": "FN", "variant": mf},
                              paper=dict(zip(("val_mean", "val_std", "val_best"), PAPER[name]))))
    for act in ("tanh", "relu"):
        for opt in ("adam", "lm", "lm_tr"):
            name = f"NN {act} {opt}"
            out.append(Config(name, lambda s, act=act: MLP(1, 20, 1, act, seed=s),
                              optimizer=opt, epochs=epochs, tags={"model": "NN", "variant": act},
                              paper=dict(zip(("val_mean", "val_std", "val_best"), PAPER[name]))))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 folds, 5 epochs (smoke test)")
    args = ap.parse_args()

    ds = make_sinc(n=20_000)
    cfgs = configs(epochs=5 if args.quick else 100)
    exp = "e1_sinc_quick" if args.quick else "e1_sinc"
    grid = torch.linspace(-1, 1, 1001).unsqueeze(1)
    _, summary = run_experiment(exp, cfgs, ds, k=2 if args.quick else 5, grid=grid)

    fn = [c.name for c in cfgs if c.name.startswith("FN")]
    nn_ = [c.name for c in cfgs if c.name.startswith("NN")]
    plot_curves(exp, fn, "paper", f"{exp}_curves_fn.png", "E1 sinc — FN öğrenme eğrileri (makale protokolü)")
    plot_curves(exp, nn_, "paper", f"{exp}_curves_nn.png", "E1 sinc — NN öğrenme eğrileri (makale protokolü)")
    target = lambda x: np.sinc(10 * x)
    plot_predictions(exp, target, [f"{n} | paper" for n in ("FN gauss lm_tr", "FN trap adam", "FN gauss adam")],
                     f"{exp}_preds_fn.png", "E1 sinc — FN tahminleri (fold 0, makale protokolü)")
    plot_predictions(exp, target, [f"{n} | paper" for n in ("NN tanh lm_tr", "NN relu lm")],
                     f"{exp}_preds_nn.png", "E1 sinc — NN tahminleri (fold 0, makale protokolü)")

    md = ["# E1 — sinc sonuçları\n"]
    for proto, title in (("paper", "Makale protokolü (tüm fold'larda aynı başlatma, seed 42)"),
                         ("varied", "Değişken başlatma (her fold'da farklı seed)")):
        md += [f"## {title}\n", summary_markdown(summary, proto), ""]
    (TABLES / f"{exp}_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
