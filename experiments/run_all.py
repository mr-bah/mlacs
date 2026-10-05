"""Все эксперименты работы. Запуск: python -m experiments.run_all
Результаты: results/*.csv, results/summary.json; рисунки строит experiments/plots.py
"""
import json, time, os
import numpy as np, pandas as pd
import mlacs as m
from experiments.common import *

os.makedirs("results", exist_ok=True)
summary = {}
t0 = time.time()

def log(*a):
    print(f"[{time.time()-t0:6.1f}s]", *a, flush=True)

# ---------- 0. Теоретические характеристики сценариев ----------
rows = []
for s, (b, a) in SCEN.items():
    for tag, seg in (("до", b), ("после", a)):
        acf = m.ar_acf(seg["phi"], 5)
        rows.append(dict(scen=s, part=tag, phi=str(seg["phi"]),
                         mean=seg.get("mean", 0.0), var=seg.get("variance", 1.0),
                         sigma=m.Segment(length=1, **seg).sigma(),
                         **{f"rho{k+1}": acf[k] for k in range(5)}))
pd.DataFrame(rows).to_csv("results/scenarios.csv", index=False)
log("scenarios ok")

# ---------- 1. Калибровка порогов + основные сравнения ----------
H = {}          # (scen_null, detector name, block) -> h
main_rows = []
store_delays = {}
null_models = {"A": SCEN["A"][0], "B": SCEN["B"][0]}  # V, M имеют тот же «до», что и B
nulls = {k: make_null(v, N_CAL, seed=100 + i) for i, (k, v) in enumerate(null_models.items())}
dets = detectors()
for key, Xn in nulls.items():
    for d in dets:
        H[(key, d.name, BLOCK)] = m.calibrate_threshold(d, Xn, N_REF, ALPHA)
log("calibration ok")

# проверка калибровки на независимых нулевых рядах
chk = []
for key, v in null_models.items():
    Xc = make_null(v, N_TEST, seed=900 + len(key))
    for d in dets:
        h = H[(key, d.name, BLOCK)]
        S, _, _ = d.statistic_path(Xc, N_REF)
        chk.append(dict(null=key, det=d.name, h=h, far=100 * (S.max(axis=1) > h).mean()))
pd.DataFrame(chk).to_csv("results/calibration.csv", index=False)
log("calibration check ok")

for s, (b, a) in SCEN.items():
    X = make(b, a, N_TEST, seed=200 + ord(s))
    key = "A" if s == "A" else "B"
    for d in dets:
        r = evaluate(d, X, H[(key, d.name, BLOCK)])
        store_delays[(s, d.name)] = r.pop("delays").tolist()
        r.pop("loc")
        main_rows.append(dict(scen=s, det=d.name, **r))
    log("scenario", s, "ok")
main = pd.DataFrame(main_rows)
main.to_csv("results/main.csv", index=False)
json.dump({f"{k[0]}|{k[1]}": v for k, v in store_delays.items()},
          open("results/delays.json", "w"))

# ---------- 2. Сила сдвига ----------
sw = []
sel = [d for d in dets if d.name in ("Остатки AR(p)", "ACF-1 (лаг 1)", "Спектр (4 полосы)",
                                     "MLACS (L=2)", "MLACS (L=3)", "Дисперсия")]
for s in ("A", "B"):
    b, a = SCEN[s]
    for delta in np.round(np.arange(0, 1.01, 0.1), 2):
        X = make(b, interp(b, a, delta), 500, seed=int(3000 + 100 * delta) + ord(s))
        for d in sel:
            r = evaluate(d, X, H[(s, d.name, BLOCK)])
            sw.append(dict(scen=s, delta=delta, det=d.name, det_rate=r["det_rate"],
                           far=r["far"], median_delay=r["median_delay"]))
    log("sweep", s, "ok")
pd.DataFrame(sw).to_csv("results/sweep.csv", index=False)

# ---------- 3. Число лагов ----------
lr = []
for s in ("A", "B"):
    b, a = SCEN[s]
    Xn = nulls[s]
    X = make(b, a, N_TEST, seed=400 + ord(s))
    for L in range(1, 11):
        d = m.MLACS(lags=L, block=BLOCK, k=K)
        h = m.calibrate_threshold(d, Xn, N_REF, ALPHA)
        r = evaluate(d, X, h)
        dl = np.array(r["delays"])
        lr.append(dict(scen=s, L=L, h=h, far=r["far"], det=r["det_rate"],
                       mean_delay=r["mean_delay"], median_delay=r["median_delay"],
                       se_delay=float(dl.std(ddof=1) / np.sqrt(len(dl))) if len(dl) > 1 else np.nan))
    log("lags", s, "ok")
pd.DataFrame(lr).to_csv("results/lags.csv", index=False)

# ---------- 4. Длина блока ----------
br = []
b0, a0 = SCEN["B"]
Xb = make(b0, a0, N_TEST, seed=555)
for blk in (25, 50, 100):
    Xn = nulls["B"]
    for d in (m.ACF1Detector(block=blk, k=K), m.MLACS(lags=3, block=blk, k=K)):
        h = m.calibrate_threshold(d, Xn, N_REF, ALPHA)
        r = evaluate(d, Xb, h)
        br.append(dict(block=blk, det=d.name, h=h, far=r["far"], det_rate=r["det_rate"],
                       mean_delay=r["mean_delay"], median_delay=r["median_delay"]))
log("block ok")
pd.DataFrame(br).to_csv("results/block.csv", index=False)

# ---------- 5. Локализация: онлайн MLACS vs офлайн перебор ----------
Xl = make(b0, a0, 500, seed=777)
d3 = m.MLACS(lags=3, block=BLOCK, k=K)
r = evaluate(d3, Xl, H[("B", d3.name, BLOCK)])
loc_rows = [dict(method="MLACS (L=3), онлайн", err=e) for e in r["loc"]]
for model, label in (("mean", "Перебор: среднее"), ("meanvar", "Перебор: среднее+дисперсия"),
                     ("ar", "Перебор: AR(2)")):
    for i in range(len(Xl)):
        tau = m.offline_single_split(Xl[i], model=model, order=2, min_size=200, jump=5)
        loc_rows.append(dict(method=label, err=abs(tau - CHANGE)))
loc = pd.DataFrame(loc_rows)
loc.to_csv("results/localization.csv", index=False)
log("localization ok")

# перекрёстная проверка офлайн-перебора библиотекой ruptures
import ruptures as rpt
cc = []
for i in range(60):
    y = Xl[i]
    own_ar = m.offline_single_split(y, model="ar", order=2, min_size=200, jump=5)
    own_mean = m.offline_single_split(y, model="mean", min_size=200, jump=5)
    rp_ar = rpt.Binseg(model="ar", params={"order": 2}, min_size=200, jump=5).fit(y).predict(n_bkps=1)[0]
    rp_mean = rpt.Binseg(model="l2", min_size=200, jump=5).fit(y).predict(n_bkps=1)[0]
    cc.append(dict(own_ar=own_ar, rpt_ar=rp_ar, own_mean=own_mean, rpt_mean=rp_mean))
cc = pd.DataFrame(cc)
cc.to_csv("results/ruptures_check.csv", index=False)
log("ruptures ok")

# ---------- 6. Практическая калибровка (параметрический бутстрэп) ----------
Xbs = make(b0, a0, 300, seed=888)
d3 = m.MLACS(lags=3, block=BLOCK, k=K)
hs = np.array([m.calibrate_threshold_bootstrap(d3, Xbs[i, :N_REF], CHANGE - 10 - N_REF,
                                               ALPHA, n_sim=200, seed=i) for i in range(len(Xbs))])
S, zl, _ = d3.statistic_path(Xbs, N_REF)
at = np.array([N_REF + (np.argmax(S[i] > hs[i]) + 1) * BLOCK if (S[i] > hs[i]).any() else -1
               for i in range(len(Xbs))])
fa = (at > 0) & (at <= CHANGE)
ok = at > CHANGE
boot = dict(h_mean=float(hs.mean()), h_sd=float(hs.std()), h_exact=H[("B", d3.name, BLOCK)],
            far=float(100 * fa.mean()), det=float(100 * ok.mean()),
            median_delay=float(np.median(at[ok] - CHANGE)))
summary["bootstrap"] = boot
log("bootstrap", boot)

json.dump(summary, open("results/summary.json", "w"), ensure_ascii=False, indent=1, default=float)
log("done; далее: python -m experiments.multi")
