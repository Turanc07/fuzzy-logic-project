"""E2 — trigonometric dataset y = sin(2*pi*x2) * cos(2*pi*x3), x1 = pure noise.

Reproduces Table 3 (optimizer comparison on the [3,1,7] trapezoidal FN) and
the interpretability check of Fig. 12 (the noise feature x1 should end up
with membership 1 over [0, 1] in every rule).

    python -m experiments.baseline.e2_trig
    python -m experiments.baseline.e2_trig --quick
"""
from __future__ import annotations

import argparse

import torch

from src.data import make_trig
from src.experiment import TABLES, Config, plot_curves, run_experiment, summary_markdown
from src.fuzzy_network import FuzzyNetwork
from src.membership import trapmf

# Paper, Table 3: validation loss mean, std, best fold.
PAPER = {
    "FN trap adam":  (0.053763, 0.012981, 0.038346),
    "FN trap lm":    (0.010555, 0.014349, 0.003028),
    "FN trap lm_tr": (0.010896, 0.018066, 0.001259),
}


def configs(epochs: int) -> list[Config]:
    return [Config(name, lambda s: FuzzyNetwork([[3, 1, 7]], ante="trap", cons="trap", seed=s),
                   optimizer=name.split()[-1], epochs=epochs, tags={"model": "FN", "variant": "trap"},
                   paper=dict(zip(("val_mean", "val_std", "val_best"), PAPER[name])))
            for name in PAPER]


@torch.no_grad()
def noise_feature_check(net: FuzzyNetwork) -> list[float]:
    """Mean membership of x1 in each rule over [0, 1]; ~1.0 means the rule ignores x1."""
    u = torch.linspace(0, 1, 201)
    A = net.layers[0].antes                                   # (K, 3, 4)
    return [trapmf(u, A[i, 0]).mean().item() for i in range(A.shape[0])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    ds = make_trig(n=8000, seed=0)
    epochs = 5 if args.quick else 100
    exp = "e2_trig_quick" if args.quick else "e2_trig"
    cfgs = configs(epochs)
    _, summary = run_experiment(exp, cfgs, ds, k=2 if args.quick else 5)
    plot_curves(exp, [c.name for c in cfgs], "paper", f"{exp}_curves.png",
                "E2 trigonometrik — FN [3,1,7] öğrenme eğrileri (makale protokolü)")

    # Interpretability: train one LM-TR model on all data and inspect the noise feature x1.
    torch.manual_seed(42)
    from src.train import train_model
    net = FuzzyNetwork([[3, 1, 7]], ante="trap", cons="trap", seed=42)
    res = train_model(net, ds.x, ds.y, optimizer="lm_tr", epochs=epochs, batch_size=len(ds) // 10)
    x1 = noise_feature_check(net)
    rules = net.rules()

    md = ["# E2 — trigonometrik veri sonuçları\n",
          "y, (sin·cos + 1)/2 ile [0, 1]'e normalize edildi (makale y'nin normalize olduğunu söylüyor).\n"]
    for proto, title in (("paper", "Makale protokolü (sabit başlatma, seed 42)"),
                         ("varied", "Değişken başlatma (her fold'da farklı seed)")):
        md += [f"## {title}\n", summary_markdown(summary, proto), ""]
    md += ["## Gürültü özelliği x1 (tüm veriyle eğitilen LM-TR modeli)\n",
           f"Eğitim MSE: {res.train_mse:.3e}\n",
           "| Kural | x1 MF'sinin [0,1] üzerindeki ortalama üyeliği |", "|---|---|"]
    md += [f"| R{i} | {v:.3f} |" for i, v in enumerate(x1)]
    md += ["", "Kurallar:", "```", *rules, "```"]
    (TABLES / f"{exp}_summary.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
