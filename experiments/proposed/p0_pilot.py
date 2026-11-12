"""P0 — pilot: random vs coverage-aware initialisation (fold 0, few seeds).

Used for the Checkpoint 4 proposal. Output: results/tables/p0_pilot_log.txt
    python -m experiments.proposed.p0_pilot > results/tables/p0_pilot_log.txt   (~10 min)
"""
import numpy as np, torch, warnings
warnings.filterwarnings("ignore")
from src.data import make_sinc, make_trig, kfold_indices
from src.fuzzy_network import FuzzyNetwork
from src.initialization import initialize, dead_rules
from src.train import train_model
def run(ds, layers, mf, opt, method, seed):
    tr, va = (torch.as_tensor(i) for i in kfold_indices(len(ds),5,0)[0])
    net = FuzzyNetwork(layers, mf, mf, seed=seed)
    initialize(net, ds.x[tr], ds.y[tr], method, seed=seed)
    with torch.no_grad(): m0 = torch.mean((net(ds.x[va])-ds.y[va])**2).item()
    d0 = dead_rules(net, ds.x[tr])
    r = train_model(net, ds.x[tr], ds.y[tr], ds.x[va], ds.y[va], optimizer=opt, epochs=100, batch_size=int(np.ceil(len(tr)/10)))
    return m0, d0, r.val_mse
for title, ds, layers, mf, opt, seeds in [("sinc gauss lm_tr", make_sinc(), [[1,1,20]], "gauss", "lm_tr", range(8)),
                                          ("trig trap lm_tr", make_trig(), [[3,1,7]], "trap", "lm_tr", range(4))]:
    for method in ["random","lhs","fcm","fcm_xy"]:
        res = [run(ds, layers, mf, opt, method, s) for s in seeds]
        v = np.array([r[2] for r in res])
        print(f"{title:17s} {method:7s} start {np.median([r[0] for r in res]):.1e}  dead0 {np.mean([r[1] for r in res]):.1f}  "
              f"final median {np.median(v):.2e}  best {v.min():.1e}  <1e-5: {(v<1e-5).sum()}/{len(v)}  all: {' '.join(f'{x:.0e}' for x in v)}", flush=True)
