"""Analysis of P1-P5: tables, paired statistical tests and figures.

Reads results/tables/{p1_convergence.csv, p2_sinc_cv_folds.csv, p3_trig_cv_folds.csv}
and writes results/tables/p_summary.md plus figures in results/figures/.

Statistics (all versus the random baseline, same seeds / folds):
  * Wilcoxon signed-rank test on log10(val MSE), paired, two-sided
  * Fisher's exact test on convergence counts (P1)

    python -m experiments.proposed.p_analysis
    python -m experiments.proposed.p_analysis --quick   # reads the *_quick outputs
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon

from src.experiment import FIGURES, TABLES

from .common import INIT_LABEL, INITS, SUCCESS_SINC

OPT_LABEL = {"adam": "Adam", "lm": "LM", "lm_tr": "LM-TR"}


def e(v: float) -> str:
    return "—" if v is None or not np.isfinite(v) else f"{v:.2e}"


def p_str(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "—"
    return f"{p:.3f}" + (" *" if p < 0.05 else "")


def paired_wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Wilcoxon on log10 MSE; returns (p, fraction of pairs where b < a)."""
    la, lb = np.log10(a), np.log10(b)
    d = lb - la
    if np.allclose(d, 0):
        return 1.0, 0.0
    try:
        p = wilcoxon(la, lb, alternative="two-sided").pvalue
    except ValueError:
        p = np.nan
    return float(p), float((d < 0).mean())


# --------------------------------------------------------------------- P1
def analyse_p1(df: pd.DataFrame) -> list[str]:
    df = df.assign(init=df["config"].str.split(" \\| ").str[1])
    out = ["## P1 — Yakınsama oranı (sinc, Gauss `[1,1,20]`, fold 0, 20 başlatma)\n",
           f"Başarı: val. MSE < {SUCCESS_SINC:g}. Testler random'a karşı, aynı tohumlarla eşleştirilmiş.\n",
           "| Optimizer | Başlatma | Yakınsayan | Medyan val. MSE | En iyi | En kötü | Başlangıç MSE (medyan) | "
           "Fisher p | Wilcoxon p | Daha iyi olduğu çift |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for opt in ("lm_tr", "lm", "adam"):
        base = df[(df.optimizer == opt) & (df.init == "random")].sort_values("init_seed")
        nb = int((base.val_mse < SUCCESS_SINC).sum())
        for init in INITS:
            g = df[(df.optimizer == opt) & (df.init == init)].sort_values("init_seed")
            if g.empty:
                continue
            v = g.val_mse.to_numpy()
            ns = int((v < SUCCESS_SINC).sum())
            if init == "random":
                pf = pw = frac = np.nan
            else:
                pf = fisher_exact([[ns, len(v) - ns], [nb, len(base) - nb]]).pvalue
                pw, frac = paired_wilcoxon(base.val_mse.to_numpy(), v)
            out.append(f"| {OPT_LABEL[opt]} | {INIT_LABEL[init]} | **{ns}/{len(v)}** | {e(np.median(v))} | {e(v.min())} | "
                       f"{e(v.max())} | {e(np.median(g.init_val_mse))} | {p_str(pf)} | {p_str(pw)} | "
                       f"{'—' if init == 'random' else f'{frac:.0%}'} |")
    # figure
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    rng = np.random.default_rng(0)
    for ax, opt in zip(axes, ("lm_tr", "lm", "adam")):
        for i, init in enumerate(INITS):
            g = df[(df.optimizer == opt) & (df.init == init)]
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(g)), g.val_mse, s=18, alpha=0.75)
        ax.axhline(SUCCESS_SINC, color="gray", ls=":")
        ax.set_xticks(range(len(INITS)), INITS); ax.set_yscale("log"); ax.set_title(OPT_LABEL[opt])
        ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Doğrulama MSE")
    fig.suptitle("P1 — sinc, 20 başlatma: başlatma yöntemine göre son hata (nokta = bir başlatma)")
    fig.tight_layout(); fig.savefig(FIGURES / "p1_convergence.png", dpi=150); plt.close(fig)
    return out


# ------------------------------------------------------------------ P2/P3
def cv_table(df: pd.DataFrame, title: str, variants: tuple[str, ...], success: float | None,
             pair_cols: tuple[str, ...], extra_cols: dict[str, str]) -> list[str]:
    df = df.assign(init=df["config"].str.split(" \\| ").str[1])
    hdr = ["Model", "Optimizer", "Başlatma", "Val. MSE ort. ± std", "Medyan", "En iyi"]
    if success is not None:
        hdr.append("Yakınsayan")
    hdr += list(extra_cols.values()) + ["Başlatma süresi [s]", "Eğitim süresi [s]", "Wilcoxon p", "Daha iyi çift"]
    out = [title, "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for var in variants:
        for opt in ("adam", "lm", "lm_tr"):
            base = df[(df.variant == var) & (df.optimizer == opt) & (df.init == "random")].sort_values(list(pair_cols))
            for init in INITS:
                g = df[(df.variant == var) & (df.optimizer == opt) & (df.init == init)].sort_values(list(pair_cols))
                if g.empty:
                    continue
                v = g.val_mse.to_numpy()
                cells = [var, OPT_LABEL[opt], INIT_LABEL[init], f"{e(v.mean())} ± {e(v.std())}", e(np.median(v)), e(v.min())]
                if success is not None:
                    cells.append(f"{int((v < success).sum())}/{len(v)}")
                cells += [f"{g[c].mean():.2f}" for c in extra_cols]
                cells += [f"{g.init_time_s.mean():.3f}", f"{g.time_s.mean():.1f}"]
                if init == "random":
                    cells += ["—", "—"]
                else:
                    pw, frac = paired_wilcoxon(base.val_mse.to_numpy(), v)
                    cells += [p_str(pw), f"{frac:.0%}"]
                out.append("| " + " | ".join(cells) + " |")
    return out


def boxplot(df: pd.DataFrame, variant: str, out: str, title: str) -> None:
    df = df.assign(init=df["config"].str.split(" \\| ").str[1])
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, opt in zip(axes, ("adam", "lm", "lm_tr")):
        data = [df[(df.variant == variant) & (df.optimizer == opt) & (df.init == i)].val_mse for i in INITS]
        ax.boxplot(data, tick_labels=list(INITS))
        ax.set_yscale("log"); ax.set_title(OPT_LABEL[opt]); ax.grid(True, which="both", alpha=0.3)
    axes[0].set_ylabel("Doğrulama MSE")
    fig.suptitle(title); fig.tight_layout(); fig.savefig(FIGURES / out, dpi=150); plt.close(fig)


def preds_figure(suffix: str) -> None:
    path = TABLES / f"p2_sinc_cv{suffix}_preds.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df.x, np.sinc(10 * df.x), color="black", lw=2.5, label="Hedef")
    for init in INITS:
        col = f"FN gauss lm_tr | {init} | varied"
        if col in df:
            ax.plot(df.x, df[col], "--", label=f"LM-TR, {init}")
    ax.set_title("P2 — sinc, Gauss FN + LM-TR, fold 0: başlatmaya göre tahmin"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "p2_sinc_preds.png", dpi=150); plt.close(fig)


def headline(p1: pd.DataFrame | None, p3: pd.DataFrame | None) -> list[str]:
    """Yönerge §6.2 format: metric | baseline | proposed | change."""
    rows = ["## Özet: Baseline ve önerilen yöntem (yönerge §6.2 formatı)\n",
            "| Performans ölçütü | Baseline (random) | Önerilen (fcm) | Değişim |", "|---|---|---|---|"]
    if p1 is not None:
        d = p1.assign(init=p1["config"].str.split(" \\| ").str[1])
        b, f = d[(d.optimizer == "lm_tr") & (d.init == "random")], d[(d.optimizer == "lm_tr") & (d.init == "fcm")]
        sb, sf = (b.val_mse < SUCCESS_SINC).mean(), (f.val_mse < SUCCESS_SINC).mean()
        rows.append(f"| sinc LM-TR yakınsama oranı (P1) | {sb:.0%} | {sf:.0%} | {100 * (sf - sb):+.0f} puan |")
        mb, mf = b.val_mse.median(), f.val_mse.median()
        rows.append(f"| sinc LM-TR medyan val. MSE (P1) | {e(mb)} | {e(mf)} | {mb / mf:.0f}× daha düşük |")
        tb, tf = (b.time_s + b.init_time_s).mean(), (f.time_s + f.init_time_s).mean()
        rows.append(f"| sinc LM-TR toplam süre [s] (P1) | {tb:.1f} | {tf:.1f} | {100 * (tf - tb) / tb:+.0f}% |")
    if p3 is not None:
        d = p3.assign(init=p3["config"].str.split(" \\| ").str[1])
        b, f = d[(d.optimizer == "lm_tr") & (d.init == "random")], d[(d.optimizer == "lm_tr") & (d.init == "fcm")]
        rows.append(f"| trig LM-TR ort. val. MSE (P3) | {e(b.val_mse.mean())} | {e(f.val_mse.mean())} | "
                    f"{b.val_mse.mean() / f.val_mse.mean():.1f}× daha düşük |")
        rows.append(f"| trig LM-TR RMSE (P3) | {np.sqrt(b.val_mse.mean()):.4f} | {np.sqrt(f.val_mse.mean()):.4f} | "
                    f"{100 * (np.sqrt(f.val_mse.mean()) - np.sqrt(b.val_mse.mean())) / np.sqrt(b.val_mse.mean()):+.0f}% |")
        rows.append(f"| trig başlangıçta ölü kural (ort., P3) | {b.dead_rules_init.mean():.2f} | {f.dead_rules_init.mean():.2f} | — |")
        tb, tf = (b.time_s + b.init_time_s).mean(), (f.time_s + f.init_time_s).mean()
        rows.append(f"| trig LM-TR toplam süre [s] (P3) | {tb:.1f} | {tf:.1f} | {100 * (tf - tb) / tb:+.0f}% |")
    return rows + [""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    q = "_quick" if args.quick else ""
    FIGURES.mkdir(parents=True, exist_ok=True)
    md = ["# Checkpoint 5 — Baseline ve önerilen yöntemin karşılaştırması\n",
          "`*` = p < 0.05. \"Daha iyi çift\": önerilen yöntemin random'dan daha düşük hata verdiği eşleştirilmiş çiftlerin oranı.\n"]

    p1 = pd.read_csv(TABLES / f"p1_convergence{q}.csv") if (TABLES / f"p1_convergence{q}.csv").exists() else None
    p2 = pd.read_csv(TABLES / f"p2_sinc_cv{q}_folds.csv") if (TABLES / f"p2_sinc_cv{q}_folds.csv").exists() else None
    p3 = pd.read_csv(TABLES / f"p3_trig_cv{q}_folds.csv") if (TABLES / f"p3_trig_cv{q}_folds.csv").exists() else None

    md += headline(p1, p3)
    if p1 is not None:
        md += analyse_p1(p1) + [""]
    if p2 is not None:
        md += cv_table(p2, "## P2 — sinc, 5-katlı CV (her fold'da farklı başlatma)\n\n"
                           "Not: 5 eşleştirilmiş çiftle Wilcoxon testinin ulaşabileceği en küçük p değeri 0.0625; "
                           "bu tablo betimleyicidir, anlamlılık P1'de test edilir.\n",
                       ("gauss", "trap"), SUCCESS_SINC, ("fold",),
                       {"dead_rules_init": "Ölü kural (başta)"}) + [""]
        boxplot(p2, "gauss", "p2_sinc_gauss_box.png", "P2 — sinc, Gauss FN, 5 fold")
        boxplot(p2, "trap", "p2_sinc_trap_box.png", "P2 — sinc, trapezoid FN, 5 fold")
        preds_figure(q)
    if p3 is not None:
        md += cv_table(p3, "## P3/P4 — Trigonometrik veri, 5-katlı CV × 2 başlatma (10 eşleştirilmiş çift)\n\n"
                           "Yorumlanabilirlik sütunları: başlangıçtaki / eğitim sonundaki ölü kural sayısı, hiçbir kuralın "
                           "ateşlenmediği doğrulama örneği oranı, ortalama kapsama oranı ρ (Denk. 14; 1 = global kural), "
                           "gürültü özelliği x₁'i tamamen yok sayan kural sayısı.\n",
                       ("trap",), None, ("protocol", "fold"),
                       {"dead_rules_init": "Ölü kural (başta)", "dead_rules_final": "Ölü kural (sonda)",
                        "uncovered_val": "Kapsanmayan örnek", "rho_mean": "ρ ort.",
                        "x1_ignored_rules": "x₁'i yok sayan kural"}) + [""]
        boxplot(p3, "trap", "p3_trig_box.png", "P3 — trigonometrik veri, trapezoid FN [3,1,7], 10 çalıştırma")
    (TABLES / f"p_summary{q}.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
