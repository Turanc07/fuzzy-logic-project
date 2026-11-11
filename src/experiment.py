"""Shared machinery for the reproduction / extension experiments.

Written for this project. An experiment is a list of `Config`s evaluated with
k-fold CV under one or more initialisation protocols:

  paper  : every fold starts from the same initial weights (seed 42), as the
           authors' public notebook does (set_seed(42) before each config)
  varied : every fold gets its own initial weights (seed 100 + fold), so the
           spread also contains initialisation variance

Outputs (all CSV, in results/tables):
  <exp>_folds.csv    one row per (config, protocol, fold)
  <exp>_summary.csv  mean / std / best per (config, protocol) + paper values
  <exp>_curves.csv   learning curves (long format)
  <exp>_preds.csv    predictions of the fold-0 models on a plotting grid (optional)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import Dataset, kfold_indices
from .train import train_model

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
PROTOCOL_SEEDS = {"paper": lambda fold: 42, "varied": lambda fold: 100 + fold}


@dataclass
class Config:
    name: str                                   # e.g. "FN gauss lm_tr"
    make_model: Callable[[int], nn.Module]      # seed -> fresh model
    optimizer: str = "adam"
    lr: float = 0.01
    epochs: int = 100
    batches_per_epoch: int = 10                 # batch = train size / 10 (as in the official notebook)
    patience: int | None = None
    paper: dict = field(default_factory=dict)   # {"val_mean":..., "val_std":..., "val_best":...}
    tags: dict = field(default_factory=dict)    # free columns copied into the tables


def run_experiment(
    exp: str,
    configs: list[Config],
    ds: Dataset,
    protocols: tuple[str, ...] = ("paper", "varied"),
    k: int = 5,
    split_seed: int = 0,
    grid: torch.Tensor | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    TABLES.mkdir(parents=True, exist_ok=True)
    folds = kfold_indices(len(ds), k=k, seed=split_seed)
    fold_rows, curve_rows, pred_cols = [], [], {}
    t_start = time.perf_counter()

    for proto in protocols:
        for cfg in configs:
            for f, (tr, va) in enumerate(folds):
                tr_t, va_t = torch.as_tensor(tr), torch.as_tensor(va)
                seed = PROTOCOL_SEEDS[proto](f)
                torch.manual_seed(seed)
                model = cfg.make_model(seed)
                bs = int(np.ceil(len(tr) / cfg.batches_per_epoch))
                res = train_model(model, ds.x[tr_t], ds.y[tr_t], ds.x[va_t], ds.y[va_t],
                                  optimizer=cfg.optimizer, epochs=cfg.epochs, batch_size=bs,
                                  lr=cfg.lr, seed=split_seed * 1000 + f, patience=cfg.patience)
                fold_rows.append({"config": cfg.name, "protocol": proto, "fold": f, "init_seed": seed,
                                  "optimizer": cfg.optimizer, **cfg.tags,
                                  "train_mse": res.train_mse, "val_mse": res.val_mse,
                                  "time_s": res.time_s, "n_params": res.n_params})
                for ep, (a, b) in enumerate(zip(res.history["train_mse"], res.history["val_mse"])):
                    curve_rows.append({"config": cfg.name, "protocol": proto, "fold": f,
                                       "epoch": ep + 1, "train_mse": a, "val_mse": b})
                if grid is not None and f == 0:
                    with torch.no_grad():
                        pred_cols[f"{cfg.name} | {proto}"] = model(grid).reshape(-1).numpy()
                if verbose:
                    print(f"[{exp}] {proto:6s} {cfg.name:22s} fold {f}: val {res.val_mse:.3e} "
                          f"({res.time_s:5.1f}s)  elapsed {time.perf_counter() - t_start:6.0f}s", flush=True)

    folds_df = pd.DataFrame(fold_rows)
    summary = summarize(folds_df, configs)
    folds_df.to_csv(TABLES / f"{exp}_folds.csv", index=False)
    summary.to_csv(TABLES / f"{exp}_summary.csv", index=False)
    pd.DataFrame(curve_rows).to_csv(TABLES / f"{exp}_curves.csv", index=False)
    if grid is not None:
        pd.DataFrame({"x": grid.reshape(-1).numpy(), **pred_cols}).to_csv(TABLES / f"{exp}_preds.csv", index=False)
    return folds_df, summary


def summarize(folds_df: pd.DataFrame, configs: list[Config]) -> pd.DataFrame:
    paper = {c.name: c.paper for c in configs}
    tags = {c.name: c.tags for c in configs}
    rows = []
    for (name, proto), g in folds_df.groupby(["config", "protocol"], sort=False):
        val = g["val_mse"].to_numpy()
        best = val.min()
        rows.append({
            "config": name, "protocol": proto, **tags[name],
            "val_mean": val.mean(), "val_std": val.std(ddof=0), "val_best": best,
            "val_median": float(np.median(val)),
            "train_mean": g["train_mse"].mean(),
            # a fold "did not converge" if it ends >10x worse than the best fold
            "bad_folds": int((val > 10 * best).sum()), "folds": len(val),
            "time_mean_s": g["time_s"].mean(), "n_params": int(g["n_params"].iloc[0]),
            "paper_val_mean": paper[name].get("val_mean", np.nan),
            "paper_val_std": paper[name].get("val_std", np.nan),
            "paper_val_best": paper[name].get("val_best", np.nan),
        })
    return pd.DataFrame(rows)


def fmt(v: float) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.2e}"


def summary_markdown(summary: pd.DataFrame, protocol: str) -> str:
    """Markdown table: ours vs paper for one protocol."""
    s = summary[summary["protocol"] == protocol].sort_values("val_mean")
    lines = ["| Konfigürasyon | Val. MSE ort. ± std (bizim) | En iyi fold (bizim) | Yakınsamayan fold | "
             "Makale ort. ± std | Makale en iyi | Süre [s] | Param. |",
             "|---|---|---|---|---|---|---|---|"]
    for _, r in s.iterrows():
        lines.append(f"| {r['config']} | {fmt(r['val_mean'])} ± {fmt(r['val_std'])} | {fmt(r['val_best'])} | "
                     f"{r['bad_folds']}/{r['folds']} | "
                     f"{fmt(r['paper_val_mean'])} ± {fmt(r['paper_val_std'])} | {fmt(r['paper_val_best'])} | "
                     f"{r['time_mean_s']:.1f} | {r['n_params']} |")
    return "\n".join(lines)


def plot_curves(exp: str, configs: list[str], protocol: str, out: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cur = pd.read_csv(TABLES / f"{exp}_curves.csv")
    folds = pd.read_csv(TABLES / f"{exp}_folds.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in configs:
        g = folds[(folds.config == name) & (folds.protocol == protocol)]
        if g.empty:
            continue
        best_fold = int(g.loc[g.val_mse.idxmin(), "fold"])
        c = cur[(cur.config == name) & (cur.protocol == protocol) & (cur.fold == best_fold)]
        ax.plot(c.epoch, c.val_mse, label=name, linestyle="--" if "trap" in name or "relu" in name else "-")
    ax.set_yscale("log"); ax.set_xlabel("Epoch"); ax.set_ylabel("Doğrulama MSE (en iyi fold)")
    ax.set_title(title); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / out, dpi=150); plt.close(fig)


def plot_predictions(exp: str, target: Callable[[np.ndarray], np.ndarray], columns: list[str],
                     out: str, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(TABLES / f"{exp}_preds.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(df.x, target(df.x.to_numpy()), color="black", linewidth=2.5, label="Hedef")
    for col in columns:
        if col in df:
            ax.plot(df.x, df[col], "--", label=col)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / out, dpi=150); plt.close(fig)
