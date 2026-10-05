"""Устойчивость к тяжёлым хвостам шума (t-распределение, 5 степеней свободы).
Пороги — те же, что откалиброваны на гауссовском шуме (модель шума неверна)."""
import numpy as np, pandas as pd
import mlacs as m
from experiments.common import *

rng = np.random.default_rng(1234)
b, a = SCEN["B"]
Xn = make_null(b, N_CAL, seed=101)             # тот же нулевой набор, что в run_all для B
segs = [m.Segment(length=CHANGE, **b), m.Segment(length=N - CHANGE, **a)]
Xt = m.simulate_piecewise_ar(segs, n_runs=N_TEST, rng=rng, noise="t5")
rows = []
for d in [m.ARResidualDetector(block=BLOCK, k=K), m.ACF1Detector(block=BLOCK, k=K),
          m.SpectralDetector(block=BLOCK, k=K), m.MLACS(lags=3, block=BLOCK, k=K)]:
    h = m.calibrate_threshold(d, Xn, N_REF, ALPHA)
    r = evaluate(d, Xt, h)
    rows.append(dict(det=d.name, far=r["far"], det_rate=r["det_rate"],
                     mean_delay=r["mean_delay"], median_delay=r["median_delay"]))
df = pd.DataFrame(rows)
df.to_csv("results/robust_t5.csv", index=False)
print(df.round(1))
