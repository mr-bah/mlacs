"""Построение всех рисунков работы. Запуск: python -m experiments.plots"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import mlacs as m
from experiments.common import *

os.makedirs("figures/extra", exist_ok=True)
HORIZ = {"orientation": "horizontal"} if tuple(int(v) for v in matplotlib.__version__.split(".")[:2]) >= (3, 10) else {"vert": False}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 100, "savefig.dpi": 200,
                     "savefig.bbox": "tight"})
C = {"MLACS (L=3)": "#c0392b", "MLACS (L=2)": "#e67e22", "MLACS (L=5)": "#8e44ad",
     "ACF-1 (лаг 1)": "#2c3e50", "Остатки AR(p)": "#27ae60", "Спектр (4 полосы)": "#2980b9",
     "Дисперсия": "#7f8c8d", "Среднее": "#bdc3c7"}


def rolling(x, w, f):
    return np.array([f(x[i - w:i]) for i in range(w, len(x) + 1)])


def r_k(v, k):
    v = v - v.mean()
    return np.sum(v[k:] * v[:-k]) / np.sum(v * v)


# ---- Рис. 2.1: схема алгоритма ----
fig, ax = plt.subplots(figsize=(10, 2.8)); ax.axis("off")
boxes = ["Ряд $x_t$\n(поток\nданных)", "Блоки\nдлины $b$", "Вектор ACF\nблока\n$r_{j,1},\\ldots,r_{j,L}$",
         "Расстояние\nМахаланобиса\n$D_j^2$", "CUSUM\n$S_j$", "Тревога,\nесли\n$S_j>h$"]
n = len(boxes); wbox = 0.125; gap = (1 - n * wbox) / (n + 1)
xs = [gap + i * (wbox + gap) for i in range(n)]
for i, (x0, t) in enumerate(zip(xs, boxes)):
    ax.add_patch(FancyBboxPatch((x0, 0.28), wbox, 0.5, boxstyle="round,pad=0.01",
                                fc="#fdecea" if i in (2, 3, 4) else "#f2f2f2", ec="#555"))
    ax.text(x0 + wbox / 2, 0.53, t, ha="center", va="center", fontsize=9)
    if i < n - 1:
        ax.annotate("", xy=(xs[i + 1] - 0.004, 0.53), xytext=(x0 + wbox + 0.006, 0.53),
                    arrowprops=dict(arrowstyle="->", color="#333"))
ax.text(0.5, 0.06, "по опорному участку: оценка $\\mu_0,\\ \\Sigma_0$ и порога $h$ (параметрический бутстрэп)",
        ha="center", fontsize=9, style="italic")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
fig.savefig("figures/fig2_1_scheme.png"); plt.close(fig)

# ---- Рис. 3.1: пример скрытого сдвига (сценарий Б) ----
b, a = SCEN["B"]
x = make(b, a, 1, seed=31)[0]
w = 250
t = np.arange(w, len(x) + 1)
fig, axs = plt.subplots(3, 1, figsize=(9, 5.6), sharex=True)
axs[0].plot(x, lw=0.4, color="#444"); axs[0].set_ylabel("$x_t$")
axs[0].set_title("Ряд: AR(1), φ=0.5  →  AR(2), φ=(0.3; 0.4); момент изменения t = 3010")
axs[1].plot(t, rolling(x, w, np.mean), label="скользящее среднее")
axs[1].plot(t, rolling(x, w, np.var), label="скользящая дисперсия")
axs[1].legend(loc="upper left"); axs[1].set_ylabel("окно 250")
axs[2].plot(t, rolling(x, w, lambda v: r_k(v, 1)), label="ACF, лаг 1", color="#2c3e50")
axs[2].plot(t, rolling(x, w, lambda v: r_k(v, 2)), label="ACF, лаг 2", color="#c0392b")
axs[2].legend(loc="upper left"); axs[2].set_ylabel("окно 250"); axs[2].set_xlabel("время t")
for ax_ in axs:
    ax_.axvline(CHANGE, color="k", ls="--", lw=1)
fig.savefig("figures/fig3_1_hidden_shift.png"); plt.close(fig)

# ---- Рис. 3.2: теоретическая ACF ----
fig, axs = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
lags = np.arange(1, 9)
for ax_, s, ttl in ((axs[0], "A", "Сценарий А"), (axs[1], "B", "Сценарий Б")):
    bb, aa = SCEN[s]
    ax_.bar(lags - 0.2, m.ar_acf(bb["phi"], 8), 0.4, label="до изменения", color="#95a5a6")
    ax_.bar(lags + 0.2, m.ar_acf(aa["phi"], 8), 0.4, label="после изменения", color="#c0392b")
    ax_.set_title(ttl); ax_.set_xlabel("лаг k"); ax_.set_xticks(lags)
axs[0].set_ylabel("$\\rho_k$"); axs[0].legend()
fig.savefig("figures/extra/theor_acf.png"); plt.close(fig)

# ---- Рис. 3.3: пути CUSUM на примере ----
Xn = make_null(b, N_CAL, seed=101)
fig, ax = plt.subplots(figsize=(9, 3.4))
for d in (m.ACF1Detector(block=BLOCK, k=K), m.MLACS(lags=3, block=BLOCK, k=K)):
    h = m.calibrate_threshold(d, Xn, N_REF, ALPHA)
    S, _, _ = d.statistic_path(x[None, :], N_REF)
    tt = N_REF + (np.arange(S.shape[1]) + 1) * BLOCK
    ax.plot(tt, S[0] / h, marker="o", ms=2.5, lw=1.2, color=C[d.name], label=d.name)
ax.axhline(1, color="k", lw=1, label="порог (S/h = 1)")
ax.axvline(CHANGE, color="k", ls="--", lw=1)
ax.set_xlabel("время t"); ax.set_ylabel("$S_j / h$"); ax.legend()
fig.savefig("figures/extra/cusum_paths.png"); plt.close(fig)

# ---- Рис. 3.4: доля обнаружений от силы сдвига ----
sw = pd.read_csv("results/sweep.csv")
fig, axs = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
for ax_, s, ttl in ((axs[0], "A", "Сценарий А"), (axs[1], "B", "Сценарий Б")):
    for name, g in sw[sw.scen == s].groupby("det", sort=False):
        ax_.plot(g.delta, g.det_rate, marker="o", ms=3.5, color=C[name], label=name)
    ax_.set_title(ttl); ax_.set_xlabel("сила сдвига δ")
axs[0].set_ylabel("доля обнаружений, %"); axs[1].legend(fontsize=8, loc="lower right")
fig.savefig("figures/fig3_2_sweep.png"); plt.close(fig)

# ---- Рис. 3.5: задержки в сценарии Б (boxplot) ----
dl = json.load(open("results/delays.json"))
names = ["Остатки AR(p)", "ACF-1 (лаг 1)", "Спектр (4 полосы)", "MLACS (L=2)", "MLACS (L=3)", "MLACS (L=5)"]
fig, ax = plt.subplots(figsize=(9, 3.6))
data = [dl[f"B|{n}"] for n in names]
bp = ax.boxplot(data, **HORIZ, showfliers=False, patch_artist=True, widths=0.55)
for patch, n in zip(bp["boxes"], names):
    patch.set_facecolor(C[n]); patch.set_alpha(0.55)
ax.set_yticks(range(1, len(names) + 1)); ax.set_yticklabels(names)
ax.set_xlabel("задержка обнаружения, наблюдений (среди обнаруженных)")
fig.savefig("figures/extra/delays_B.png"); plt.close(fig)

# ---- Рис. 3.6: число лагов ----
lr = pd.read_csv("results/lags.csv")
fig, ax = plt.subplots(figsize=(8, 3.6))
for s, col, lab in (("A", "#2c3e50", "сценарий А"), ("B", "#c0392b", "сценарий Б")):
    g = lr[lr.scen == s]
    ax.errorbar(g.L, g.mean_delay, yerr=1.96 * g.se_delay, marker="o", ms=4, capsize=3,
                color=col, label=lab)
ax.set_xlabel("число лагов L"); ax.set_ylabel("средняя задержка, наблюдений")
ax.set_xticks(range(1, 11)); ax.set_ylim(0, 450)
ax.annotate("L=1: %.0f" % lr[(lr.scen == "B") & (lr.L == 1)].mean_delay.iloc[0],
            xy=(1, 440), xytext=(1.6, 400), arrowprops=dict(arrowstyle="->"), fontsize=9)
ax.legend()
fig.savefig("figures/fig3_3_lags.png"); plt.close(fig)

# ---- Рис. 3.7: локализация ----
loc = pd.read_csv("results/localization.csv")
order = ["MLACS (L=3), онлайн", "Перебор: AR(2)", "Перебор: среднее", "Перебор: среднее+дисперсия"]
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.boxplot([loc[loc.method == o].err + 1 for o in order], **HORIZ, showfliers=False, widths=0.55)
ax.set_yticks(range(1, 5)); ax.set_yticklabels(order); ax.set_xscale("log")
ax.set_xlabel("ошибка локализации |τ̂ − τ| + 1, наблюдений (лог. шкала)")
fig.savefig("figures/extra/localization.png"); plt.close(fig)

# ---- Рис. 3.8: несколько точек изменения ----
if os.path.exists("results/summary.json") and "multi_demo" in json.load(open("results/summary.json")):
    sm = json.load(open("results/summary.json"))
    y = np.load("results/multi_demo_series.npy")
    CPS = [3010, 7010, 11010]
    t = np.arange(w, len(y) + 1)
    fig, axs = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    axs[0].plot(y, lw=0.3, color="#444"); axs[0].set_ylabel("$x_t$")
    labs = ["AR(1) 0.5", "AR(2) (0.3; 0.4)", "AR(1) 0.5", "AR(2) (0.75; −0.5)"]
    bnd = [0] + CPS + [len(y)]
    for i in range(4):
        axs[0].text((bnd[i] + bnd[i + 1]) / 2, axs[0].get_ylim()[1] * 0.85, labs[i],
                    ha="center", fontsize=8.5)
    axs[1].plot(t, rolling(y, w, lambda v: r_k(v, 1)), color="#2c3e50", lw=0.8, label="ACF, лаг 1")
    axs[1].plot(t, rolling(y, w, lambda v: r_k(v, 2)), color="#c0392b", lw=0.8, label="ACF, лаг 2")
    axs[1].legend(loc="lower left", fontsize=8); axs[1].set_xlabel("время t")
    for ax_ in axs:
        for c in CPS:
            ax_.axvline(c, color="k", ls="--", lw=1)
        for a_ in sm["multi_demo"]["alarms"]:
            ax_.axvline(a_, color="#e67e22", lw=1.6)
    axs[0].set_title("Штрих — истинные моменты изменения; оранжевые — тревоги MLACS (L=3)")
    fig.savefig("figures/fig3_4_multi.png"); plt.close(fig)
print("plots ok")
