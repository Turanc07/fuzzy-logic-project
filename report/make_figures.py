"""Figures that exist only for the report: architecture diagram and initial MFs.

    python -m report.make_figures
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.data import kfold_indices, make_sinc
from src.experiment import FIGURES
from src.fuzzy_network import FuzzyNetwork
from src.initialization import initialize


def box(ax, x, y, w, h, text, fc):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec="#333333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)


def arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#333333"))


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3.6); ax.axis("off")
    box(ax, 0.1, 1.4, 1.1, 0.8, "Girdi\nx ∈ ℝᴹ", "#eeeeee")
    # layer 1 detail
    ax.add_patch(FancyBboxPatch((1.6, 0.25), 5.2, 3.1, boxstyle="round,pad=0.02", fc="#f7fbff", ec="#4a78b5", lw=1.4, ls="--"))
    ax.text(4.2, 3.12, "Bulanık katman  [M, N, K]", ha="center", fontsize=10, color="#2f5d95", weight="bold")
    box(ax, 1.8, 1.25, 1.35, 1.1, "Bulanıklaştırma\nλᵢⱼ(xⱼ)\n(trap / Gauss)", "#dbe9f6")
    box(ax, 3.5, 2.05, 1.45, 0.7, "R₁: IF … THEN …", "#fde9c9")
    box(ax, 3.5, 1.35, 1.45, 0.55, "⋮", "#fde9c9")
    box(ax, 3.5, 0.5, 1.45, 0.7, "R_K: IF … THEN …", "#fde9c9")
    ax.text(4.22, 2.88, "ateşleme  wᵢ = minⱼ λᵢⱼ", ha="center", fontsize=8)
    box(ax, 5.3, 1.25, 1.35, 1.1, "Durulaştırma\nCoS (trap) /\nağırlıklı ort. (Gauss)", "#d9f0d3")
    arrow(ax, 1.2, 1.8, 1.8, 1.8)
    for yy in (2.4, 1.62, 0.85):
        arrow(ax, 3.15, 1.8, 3.5, yy)
        arrow(ax, 4.95, yy, 5.3, 1.8)
    arrow(ax, 6.65, 1.8, 7.1, 1.8)
    box(ax, 7.1, 1.4, 1.2, 0.8, "Katman 2 …\nKatman L", "#f0f0f0")
    arrow(ax, 8.3, 1.8, 8.7, 1.8)
    box(ax, 8.7, 1.4, 1.15, 0.8, "Çıktı\ny ∈ ℝᴺ", "#eeeeee")
    ax.text(5.0, 0.02, "y = f_L( … f₂( f₁(x) ) )  — katmanlar arasında ek aktivasyon fonksiyonu yok",
            ha="center", fontsize=8.5, style="italic")
    fig.tight_layout(); fig.savefig(FIGURES / "arch.png", dpi=200); plt.close(fig)


def initial_mfs() -> None:
    ds = make_sinc(n=20_000)
    tr, _ = kfold_indices(len(ds), 5, 0)[0]
    tr = torch.as_tensor(tr)
    u = torch.linspace(-1.6, 1.6, 800)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, method, title in zip(axes, ("random", "fcm"), ("Rastgele başlatma (makale)", "FCM tabanlı başlatma (önerilen)")):
        net = FuzzyNetwork([[1, 1, 20]], "gauss", "gauss", seed=0)
        initialize(net, ds.x[tr], ds.y[tr], method, seed=0)
        A = net.layers[0].antes.detach()
        for i in range(A.shape[0]):
            mu, s = A[i, 0, 0], A[i, 0, 1].abs().clamp_min(1e-8)
            ax.plot(u, torch.exp(-0.5 * ((u - mu) / s) ** 2), lw=1)
        ax.axvspan(-1, 1, color="#cccccc", alpha=0.25, label="veri aralığı [-1, 1]")
        ax.set_title(title, fontsize=10); ax.set_xlabel("x"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("üyelik derecesi"); axes[0].legend(fontsize=8, loc="upper right")
    fig.suptitle("sinc görevi, 20 kuralın öncül üyelik fonksiyonları (eğitimden önce, tohum 0)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIGURES / "init_mfs.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    architecture()
    initial_mfs()
    print("ok")
