"""Рисунок Б.1: графики приложения app.py на примере ряда с четырьмя режимами.
Запуск: python -m experiments.app_figure"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlacs as m

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True, "grid.alpha": 0.3})
EX = [((0.5,), 3010), ((0.3, 0.4), 4000), ((0.5,), 4000), ((0.75, -0.5), 3990)]
y = m.simulate_piecewise_ar([m.Segment(phi=p, length=L) for p, L in EX], 1, rng=np.random.default_rng(3))[0]
true_cps = list(np.cumsum([L for _, L in EX])[:-1])
b, L, n_ref, w = 50, 3, 2000, 200
ev = m.detect_sequential(y, lags=L, block=b, k=0.5, n_ref=n_ref, alpha=0.05, horizon=1000, n_sim=200)
alarms = [e for e in ev if e["alarm"] >= 0]

fig, axs = plt.subplots(3, 1, figsize=(10, 6.5), sharex=True)
axs[0].plot(y, lw=0.3, color="#444"); axs[0].set_ylabel("x_t")
t = np.arange(w, len(y) + 1)
yy = np.lib.stride_tricks.sliding_window_view(y, w); yy = yy - yy.mean(axis=1, keepdims=True)
den = (yy ** 2).sum(axis=1)
for lag, c in zip(range(1, 4), ["#2c3e50", "#c0392b", "#27ae60"]):
    axs[1].plot(t, (yy[:, lag:] * yy[:, :-lag]).sum(axis=1) / den, lw=0.7, color=c, label=f"ACF, лаг {lag}")
axs[1].legend(loc="lower left", fontsize=8); axs[1].set_ylabel(f"окно {w}")
for e in ev:
    tt = e["segment_start"] + n_ref + (np.arange(len(e["S"])) + 1) * b
    stop = np.searchsorted(tt, e["alarm"], side="right") if e["alarm"] >= 0 else len(tt)
    axs[2].plot(tt[:stop], e["S"][:stop] / e["h"], color="#c0392b", lw=1)
    axs[2].axvspan(e["segment_start"], e["segment_start"] + n_ref, color="#bbb", alpha=0.25)
axs[2].axhline(1, color="k", lw=1); axs[2].set_ylabel("S / h"); axs[2].set_xlabel("время t")
for ax in axs:
    for c in true_cps:
        ax.axvline(c, color="k", ls="--", lw=1)
    for e in alarms:
        ax.axvline(e["alarm"], color="#e67e22", lw=1.5)
axs[0].set_title("штрих – истинные изменения; оранжевые – тревоги; серое – опорные участки")
fig.savefig("figures/figB1_app.png", dpi=200, bbox_inches="tight")
print("тревоги:", [(e["alarm"], e["change_est"]) for e in alarms])
