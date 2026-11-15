"""P6 — linguistic reading of trained rules (trigonometric data, [3,1,7] trapezoid FN).

Trains one model per initialisation (random vs fcm) on the training part of
fold 0 with LM and turns every rule into an IF-THEN sentence:

  antecedent label from the centre of the trapezoid core on [0, 1]:
      < 1/3 -> Düşük, < 2/3 -> Orta, else -> Yüksek
      "herhangi" if the MF is ~1 on the whole input range (the rule ignores that input)
  consequent label: same thirds on the output range [0, 1]
  "ölü" if the rule never fires on the training data

    python -m experiments.proposed.p6_rules
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.data import kfold_indices, make_trig
from src.experiment import TABLES
from src.fuzzy_network import FuzzyNetwork
from src.initialization import initialize
from src.membership import sort_trap, trapmf
from src.train import train_model

LABELS = ("Düşük", "Orta", "Yüksek")


def label(center: float) -> str:
    return LABELS[int(np.clip(center * 3, 0, 2.999))]


@torch.no_grad()
def describe(net: FuzzyNetwork, x_tr: torch.Tensor, x_va: torch.Tensor, y_va: torch.Tensor) -> list[dict]:
    L = net.layers[0]
    A, C = sort_trap(L.antes), sort_trap(L.cons)
    u = torch.linspace(0, 1, 401)
    w_tr = L.firing_strengths(x_tr)
    w_va = L.firing_strengths(x_va)
    share = w_va / (w_va.sum(dim=1, keepdim=True) + 1e-12)          # normalised firing = significance (Gauss-like)
    y_mean = float(y_va.mean())
    rows = []
    for i in range(L.n_rules):
        parts = []
        for j in range(L.in_dim):
            if trapmf(u, A[i, j]).mean() > 0.99:
                parts.append(f"x{j + 1} herhangi")
            else:
                parts.append(f"x{j + 1} {label(float((A[i, j, 1] + A[i, j, 2]) / 2))}")
        c_center = float((C[i, 0, 1] + C[i, 0, 2]) / 2)
        dead = bool(w_tr[:, i].max() < 1e-6)
        # ground truth for this rule: firing-strength-weighted mean of the true y where the rule fires
        wsum = float(w_va[:, i].sum())
        y_true = float((w_va[:, i] @ y_va[:, 0]) / wsum) if wsum > 1e-9 else float("nan")
        true_lab = label(y_true) if np.isfinite(y_true) else "—"
        rows.append({"rule": f"R{i}", "if": " VE ".join(parts), "then": f"y {label(c_center)}",
                     "cons_center": c_center, "dead": dead,
                     "mean_significance": float(share[:, i].mean()),
                     "y_true_where_fires": y_true, "true_label": true_lab,
                     "consistent": (not dead) and true_lab == label(c_center),
                     # direction: does the rule push y above / below the data mean, and does the data agree?
                     "direction_ok": (not dead) and np.isfinite(y_true)
                                     and np.sign(c_center - y_mean) == np.sign(y_true - y_mean)})
    return rows


def main() -> None:
    ds = make_trig(n=8000, seed=0)
    tr, va = (torch.as_tensor(i) for i in kfold_indices(len(ds), 5, 0)[0])
    out = []
    for init in ("random", "fcm"):
        torch.manual_seed(100)
        net = FuzzyNetwork([[3, 1, 7]], "trap", "trap", seed=100)
        initialize(net, ds.x[tr], ds.y[tr], init, seed=100)
        res = train_model(net, ds.x[tr], ds.y[tr], ds.x[va], ds.y[va], optimizer="lm", epochs=100,
                          batch_size=int(np.ceil(len(tr) / 10)), seed=0)
        for r in describe(net, ds.x[tr], ds.x[va], ds.y[va]):
            out.append({"init": init, "val_mse": res.val_mse, **r})
        print(f"{init}: val {res.val_mse:.3e}")
    df = pd.DataFrame(out)
    df.to_csv(TABLES / "p6_rules.csv", index=False)
    print(df.to_string())


if __name__ == "__main__":
    main()
