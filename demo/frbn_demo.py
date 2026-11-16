"""Live demo for the presentation / oral defence.

    python -m demo.frbn_demo compare                       # random vs FCM init, same seed (sinc, LM-TR)
    python -m demo.frbn_demo train   --task trig --init fcm
    python -m demo.frbn_demo rules   --task trig --init fcm
    python -m demo.frbn_demo predict --task trig --init fcm --x 0.2 0.15 0.5
    python -m demo.frbn_demo predict --task sinc --init fcm --x 0.25

Models are cached in demo/models/ (git-ignored) so `rules` and `predict` are instant
after the first `train`. Every command prints what happens inside the fuzzy
system: memberships, firing strengths (min t-norm), each rule's share of the
output and the defuzzified result.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from src.data import kfold_indices, make_sinc, make_trig, trig_target
from src.fuzzy_network import FuzzyNetwork
from src.initialization import dead_rules, initialize, uncovered_fraction
from src.membership import MF, sort_trap
from src.train import train_model

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"
LABELS = ("Düşük", "Orta", "Yüksek")
TASKS = {
    # task: (dataset factory, architecture, MF type, true function)
    "sinc": (lambda: make_sinc(n=20_000), [[1, 1, 20]], "gauss", lambda x: float(np.sinc(10 * x[0]))),
    "trig": (lambda: make_trig(n=8000, seed=0), [[3, 1, 7]], "trap",
             lambda x: float((trig_target(torch.tensor(x[1]), torch.tensor(x[2])) + 1) / 2)),
}


def path(task: str, init: str, opt: str, seed: int) -> Path:
    return MODELS / f"{task}_{init}_{opt}_s{seed}.pt"


def build(task: str, init: str, opt: str, seed: int, verbose: bool = True) -> tuple[FuzzyNetwork, dict]:
    make, arch, mf, _ = TASKS[task]
    ds = make()
    tr, va = (torch.as_tensor(i) for i in kfold_indices(len(ds), 5, 0)[0])
    torch.manual_seed(seed)
    net = FuzzyNetwork(arch, mf, mf, seed=seed)
    t0 = time.perf_counter()
    initialize(net, ds.x[tr], ds.y[tr], init, seed=seed)
    t_init = time.perf_counter() - t0
    with torch.no_grad():
        start = torch.mean((net(ds.x[va]) - ds.y[va]) ** 2).item()
    dead0 = dead_rules(net, ds.x[tr])
    res = train_model(net, ds.x[tr], ds.y[tr], ds.x[va], ds.y[va], optimizer=opt, epochs=100,
                      batch_size=int(np.ceil(len(tr) / 10)), seed=0)
    info = {"val_mse": res.val_mse, "start_mse": start, "dead_init": dead0, "dead_final": dead_rules(net, ds.x[tr]),
            "uncovered": uncovered_fraction(net, ds.x[va]), "time": res.time_s, "init_time": t_init}
    MODELS.mkdir(parents=True, exist_ok=True)
    torch.save({"state": net.state_dict(), "info": info}, path(task, init, opt, seed))
    if verbose:
        print(f"[{task} | init={init} | {opt} | seed={seed}]  başlangıç MSE {start:.2e} → doğrulama MSE {res.val_mse:.2e}  "
              f"(başlatma {t_init:.2f}s + eğitim {res.time_s:.1f}s)")
        print(f"   ölü kural: başta {dead0}, sonda {info['dead_final']}   |   hiçbir kuralın ateşlenmediği doğrulama örneği: "
              f"%{100 * info['uncovered']:.1f}")
    return net, info


def load(task: str, init: str, opt: str, seed: int) -> FuzzyNetwork:
    p = path(task, init, opt, seed)
    if not p.exists():
        print(f"(model önbellekte yok, eğitiliyor: {p.name})")
        return build(task, init, opt, seed)[0]
    _, arch, mf, _ = TASKS[task]
    net = FuzzyNetwork(arch, mf, mf)
    net.load_state_dict(torch.load(p)["state"])
    return net


def label(v: float) -> str:
    return LABELS[int(np.clip(v * 3, 0, 2.999))]


def describe_rules(net: FuzzyNetwork, task: str) -> list[str]:
    L = net.layers[0]
    out = []
    if L.ante_type == "trap":
        A, C = sort_trap(L.antes), sort_trap(L.cons)
        u = torch.linspace(0, 1, 401)
        for i in range(L.n_rules):
            parts = []
            for j in range(L.in_dim):
                if MF["trap"](u, A[i, j]).mean() > 0.99:
                    parts.append(f"x{j + 1} herhangi")
                else:
                    parts.append(f"x{j + 1} {label(float((A[i, j, 1] + A[i, j, 2]) / 2))}")
            c = float((C[i, 0, 1] + C[i, 0, 2]) / 2)
            out.append(f"R{i}: EĞER {' VE '.join(parts)} İSE y {label(c)}   (sonuç merkezi {c:+.2f})")
    else:
        for i in range(L.n_rules):
            mu, s = float(L.antes[i, 0, 0]), abs(float(L.antes[i, 0, 1]))
            out.append(f"R{i:2d}: EĞER x ≈ {mu:+.3f} (σ={s:.3f}) İSE y ≈ {float(L.cons[i, 0, 0]):+.3f}")
    return out


@torch.no_grad()
def explain(net: FuzzyNetwork, task: str, x: list[float]) -> None:
    L = net.layers[0]
    xt = torch.tensor([x], dtype=torch.float32)
    lam = MF[L.ante_type](xt.unsqueeze(1), L.antes)[0]          # (K, M) memberships
    w = lam.min(dim=-1).values                                    # (K,)  firing strengths (min t-norm)
    y = float(net(xt))
    truth = TASKS[task][3](x)
    print(f"\nGirdi x = {x}")
    print("Adım 1 — Bulanıklaştırma ve Adım 2 — kural ateşleme (min t-norm):")
    share = w / (w.sum() + 1e-12)
    order = torch.argsort(w, descending=True)
    rules = describe_rules(net, task)
    for i in order.tolist():
        if w[i] < 1e-4:
            continue
        mems = ", ".join(f"μ{j + 1}={float(lam[i, j]):.2f}" for j in range(L.in_dim))
        print(f"   {rules[i].split('   (')[0]}\n        üyelikler [{mems}] → w = min = {float(w[i]):.3f}   ateşleme payı %{100 * float(share[i]):.0f}")
    silent = int((w < 1e-4).sum())
    print(f"   ({silent} kural bu girdide ateşlenmiyor)")
    if float(w.sum()) < 1e-8:
        print("   !! Hiçbir kural ateşlenmiyor: çıktı 0'a düşer ve bu bölgede gradyan yoktur.")
    print(f"Adım 3 — Durulaştırma ({'Center of Sums' if L.cons_type == 'trap' else 'ağırlıklı ortalama'}):")
    print(f"   model çıktısı y = {y:+.4f}    gerçek değer = {truth:+.4f}    hata = {y - truth:+.4f}")


def cmd_compare(args) -> None:
    import matplotlib.pyplot as plt
    task, opt, seed = "sinc", args.opt, args.seed
    nets = {}
    for init in ("random", args.init if args.init != "random" else "fcm"):
        nets[init], _ = build(task, init, opt, seed)
    grid = torch.linspace(-1, 1, 1001).unsqueeze(1)
    fig, axes = plt.subplots(2, len(nets), figsize=(12, 6.5), sharex=True)
    u = torch.linspace(-1.6, 1.6, 600)
    for k, (init, net) in enumerate(nets.items()):
        with torch.no_grad():
            axes[0, k].plot(grid, np.sinc(10 * grid.numpy()), "k", lw=2.5, label="hedef")
            axes[0, k].plot(grid, net(grid), "--", label="FN")
            A = net.layers[0].antes
            for i in range(A.shape[0]):
                axes[1, k].plot(u, torch.exp(-0.5 * ((u - A[i, 0, 0]) / A[i, 0, 1].abs().clamp_min(1e-6)) ** 2), lw=1)
        info = torch.load(path(task, init, opt, seed))["info"]
        axes[0, k].set_title(f"init={init}: doğrulama MSE {info['val_mse']:.1e}")
        axes[1, k].set_title("eğitim sonrası öncül üyelik fonksiyonları"); axes[1, k].axvspan(-1, 1, color="#ddd", alpha=.4)
        axes[0, k].legend(fontsize=8); axes[0, k].grid(alpha=.3); axes[1, k].grid(alpha=.3)
    fig.suptitle(f"sinc, Gauss FN [1,1,20], {opt}, aynı tohum ({seed}): yalnızca başlatma farklı")
    fig.tight_layout()
    out = MODELS / f"compare_{opt}_s{seed}.png"
    fig.savefig(out, dpi=130)
    print(f"şekil: {out}")
    if not args.no_show:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="FRBN canlı demo")
    ap.add_argument("command", choices=["train", "rules", "predict", "compare"])
    ap.add_argument("--task", choices=list(TASKS), default="trig")
    ap.add_argument("--init", choices=["random", "lhs", "fcm", "fcm_xy"], default="fcm")
    ap.add_argument("--opt", choices=["adam", "lm", "lm_tr"], default="lm_tr")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--x", type=float, nargs="+", help="girdi (sinc: 1 sayı, trig: 3 sayı, [0,1])")
    ap.add_argument("--no-show", action="store_true", help="compare: pencere açmadan sadece PNG kaydet")
    args = ap.parse_args()
    if args.command == "compare":
        args.task = "sinc"
        cmd_compare(args)
    elif args.command == "train":
        net, _ = build(args.task, args.init, args.opt, args.seed)
        print("\n".join(describe_rules(net, args.task)))
    elif args.command == "rules":
        print("\n".join(describe_rules(load(args.task, args.init, args.opt, args.seed), args.task)))
    else:
        need = 1 if args.task == "sinc" else 3
        if not args.x or len(args.x) != need:
            ap.error(f"--x için {need} sayı verin")
        explain(load(args.task, args.init, args.opt, args.seed), args.task, args.x)


if __name__ == "__main__":
    main()
