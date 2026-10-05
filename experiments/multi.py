"""Несколько точек изменения: последовательный режим с перекалибровкой порога."""
import json, time, os
import numpy as np, pandas as pd
import mlacs as m
from experiments.common import *
t0 = time.time()
def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)
summary = json.load(open("results/summary.json")) if os.path.exists("results/summary.json") else {}
N_MULTI = int(os.environ.get("N_MULTI", 200))
# ---------- 7. Несколько точек изменения ----------
REG = [(0.5,), (0.3, 0.4), (0.5,), (0.75, -0.5)]
CPS = [3010, 7010, 11010]
NTOT = 15000

def multi_series(n_runs, seed):
    rng = np.random.default_rng(seed)
    bounds = [0] + CPS + [NTOT]
    segs = [m.Segment(phi=REG[i], length=bounds[i + 1] - bounds[i]) for i in range(4)]
    return m.simulate_piecewise_ar(segs, n_runs=n_runs, rng=rng)

def sequential(y, det, seed=0):
    ev = m.detect_sequential(y, detector=det, n_ref=N_REF, alpha=ALPHA, horizon=1000,
                             n_sim=200, seed=seed)
    ev = [e for e in ev if e["alarm"] >= 0]
    return [e["alarm"] for e in ev], [e["change_est"] for e in ev]

Y = multi_series(N_MULTI, seed=999)
det = m.MLACS(lags=3, block=BLOCK, k=K)
mrows, all_alarms = [], []
for i in range(len(Y)):
    al, es = sequential(Y[i], det, seed=i)
    all_alarms.append(dict(alarms=al, ests=es))
    used = set()
    for c in CPS:
        hit = [a for a in al if c < a <= c + 1500 and a not in used]
        if hit:
            used.add(hit[0])
        mrows.append(dict(run=i, cp=c, found=bool(hit), delay=(hit[0] - c) if hit else np.nan))
    mrows.append(dict(run=i, cp=-1, found=False, delay=np.nan, false=len(al) - len(used)))
mr = pd.DataFrame(mrows)
mr.to_csv("results/multi.csv", index=False)
# для рисунка берём первую реализацию, где все три изменения найдены без ложных тревог
clean = [i for i in range(len(Y)) if mr[(mr.run == i) & (mr.cp > 0)].found.all()
         and mr[(mr.run == i) & (mr.cp == -1)]["false"].iloc[0] == 0]
di = clean[0]
np.save("results/multi_demo_series.npy", Y[di])
summary["multi_demo"] = dict(run=di, **all_alarms[di])
summary["multi_clean_share"] = 100 * len(clean) / len(Y)
summary["multi"] = {str(c): dict(rate=float(100 * mr[mr.cp == c].found.mean()),
                                 median_delay=float(mr[mr.cp == c].delay.median()))
                    for c in CPS}
summary["multi_false_per_run"] = float(mr[mr.cp == -1]["false"].mean())
log("multi", summary["multi"], summary["multi_false_per_run"])

json.dump(summary, open("results/summary.json", "w"), ensure_ascii=False, indent=1, default=float)
log("done")
