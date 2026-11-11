"""E1b — initialisation sensitivity on the sinc task.

E1 showed that FN + LM either reaches ~1e-7 or stalls at ~4e-3 depending on
the random initial weights. Here the data split is fixed (fold 0) and only the
initialisation seed changes (20 seeds), so the spread is pure init variance.

    python -m experiments.baseline.e1b_init_sensitivity          # ~10 min CPU
    python -m experiments.baseline.e1b_init_sensitivity --quick
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from src.data import kfold_indices, make_sinc
from src.experiment import FIGURES, TABLES
from src.fuzzy_network import FuzzyNetwork
from src.nn_baseline import MLP
from src.train import train_model

SUCCESS = 1e-5   # a run "converges" if its validation MSE is below this

MODELS = {
    "FN gauss lm_tr": (lambda s: FuzzyNetwork([[1, 1, 20]], "gauss", "gauss", seed=s), "lm_tr"),
    "FN gauss lm":    (lambda s: FuzzyNetwork([[1, 1, 20]], "gauss", "gauss", seed=s), "lm"),
    "NN tanh lm_tr":  (lambda s: MLP(1, 20, 1, "tanh", seed=s), "lm_tr"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    n_seeds, epochs = (3, 5) if args.quick else (20, 100)
    exp = "e1b_init_quick" if args.quick else "e1b_init"

    ds = make_sinc(n=20_000)
    tr, va = (torch.as_tensor(i) for i in kfold_indices(len(ds), 5, 0)[0])
    rows = []
    for name, (mk, opt) in MODELS.items():
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            res = train_model(mk(seed), ds.x[tr], ds.y[tr], ds.x[va], ds.y[va], optimizer=opt,
                              epochs=epochs, batch_size=int(np.ceil(len(tr) / 10)), seed=0)
            rows.append({"config": name, "init_seed": seed, "val_mse": res.val_mse, "time_s": res.time_s})
            print(f"[{exp}] {name:15s} seed {seed:2d}: val {res.val_mse:.3e}", flush=True)

    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{exp}.csv", index=False)

    lines = ["# E1b — Başlatma duyarlılığı (sinc, fold 0 sabit, yalnızca başlatma tohumu değişiyor)\n",
             f"Başarı ölçütü: doğrulama MSE < {SUCCESS:g}\n",
             "| Konfigürasyon | Başarılı / toplam | Medyan val. MSE | En iyi | En kötü |", "|---|---|---|---|---|"]
    for name, g in df.groupby("config", sort=False):
        v = g.val_mse.to_numpy()
        lines.append(f"| {name} | {(v < SUCCESS).sum()}/{len(v)} | {np.median(v):.2e} | {v.min():.2e} | {v.max():.2e} |")
    (TABLES / f"{exp}_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, (name, g) in enumerate(df.groupby("config", sort=False)):
        ax.scatter(np.full(len(g), i) + np.random.default_rng(0).uniform(-0.12, 0.12, len(g)),
                   g.val_mse, alpha=0.75)
    ax.axhline(SUCCESS, color="gray", ls=":", label=f"başarı eşiği {SUCCESS:g}")
    ax.set_xticks(range(df.config.nunique()), df.config.unique())
    ax.set_yscale("log"); ax.set_ylabel("Doğrulama MSE"); ax.legend()
    ax.set_title(f"E1b — {n_seeds} farklı başlatma, aynı veri bölmesi")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig(FIGURES / f"{exp}.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
