"""Интерактивное приложение MLACS.  Запуск:  streamlit run app.py"""
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import mlacs as m

st.set_page_config(page_title="MLACS — скрытые автокорреляционные сдвиги", layout="wide")
st.title("MLACS: поиск скрытых автокорреляционных сдвигов")
st.caption("Онлайн-детектор: вектор ACF по нескольким лагам в блоках + CUSUM по расстоянию "
           "Махаланобиса. Порог подбирается параметрическим бутстрэпом по опорному участку.")

EXAMPLES = {
    "4 режима с одинаковыми средним, дисперсией и ρ₁": [((0.5,), 3010), ((0.3, 0.4), 4000),
                                                      ((0.5,), 4000), ((0.75, -0.5), 3990)],
    "Сценарий Б: AR(1) 0.5 → AR(2) (0.3; 0.4)": [((0.5,), 3010), ((0.3, 0.4), 1990)],
    "Сценарий А: AR(1) 0.2 → AR(1) 0.6": [((0.2,), 3010), ((0.6,), 1990)],
    "Без изменений: AR(1) 0.5": [((0.5,), 5000)],
}

with st.sidebar:
    st.header("Данные")
    src = st.radio("Источник", ["Синтетический пример", "Свой CSV-файл"])
    true_cps = []
    if src == "Синтетический пример":
        ex = st.selectbox("Пример", list(EXAMPLES))
        seed = st.number_input("Зерно генератора", 0, 10_000, 3)
        segs = [m.Segment(phi=ph, length=L) for ph, L in EXAMPLES[ex]]
        y = m.simulate_piecewise_ar(segs, 1, rng=np.random.default_rng(int(seed)))[0]
        true_cps = list(np.cumsum([L for _, L in EXAMPLES[ex]])[:-1])
    else:
        f = st.file_uploader("CSV со столбцом значений", type=["csv", "txt"])
        if f is None:
            st.info("Загрузите файл или выберите синтетический пример.")
            st.stop()
        df = pd.read_csv(f)
        num = df.select_dtypes("number").columns.tolist()
        col = st.selectbox("Столбец", num)
        y = df[col].dropna().to_numpy(float)
    st.header("Параметры")
    L = st.slider("Число лагов L", 1, 10, 3)
    b = st.select_slider("Длина блока b", [25, 50, 100, 200], 50)
    n_ref = st.number_input("Опорный участок, наблюдений", 5 * b, 20_000, min(2000, (len(y) // 3) // b * b), step=b)
    alpha = st.select_slider("Вероятность ложной тревоги α (на горизонте)", [0.01, 0.05, 0.1], 0.05)
    horizon = st.number_input("Горизонт для α, наблюдений", 200, 10_000, 1000, step=100)
    k = st.slider("Параметр CUSUM k", 0.0, 2.0, 0.5, 0.1)

n_ref = int(n_ref) // b * b
if len(y) < n_ref + 2 * b:
    st.error("Ряд слишком короткий для выбранного опорного участка.")
    st.stop()

with st.spinner("Калибровка порога и мониторинг..."):
    ev = m.detect_sequential(y, lags=L, block=b, k=k, n_ref=n_ref, alpha=alpha,
                             horizon=int(horizon), n_sim=200)
alarms = [e for e in ev if e["alarm"] >= 0]

c1, c2, c3 = st.columns(3)
c1.metric("Длина ряда", len(y)); c2.metric("Тревог", len(alarms))
c3.metric("Порог h (первый участок)", f"{ev[0]['h']:.2f}")

w = max(4 * b, 200)
fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
axs[0].plot(y, lw=0.4, color="#444"); axs[0].set_ylabel("x_t")
t = np.arange(w, len(y) + 1)
yy = np.lib.stride_tricks.sliding_window_view(y, w)
yy = yy - yy.mean(axis=1, keepdims=True)
den = (yy ** 2).sum(axis=1)
for lag, col_ in zip(range(1, min(L, 3) + 1), ["#2c3e50", "#c0392b", "#27ae60"]):
    r = (yy[:, lag:] * yy[:, :-lag]).sum(axis=1) / den
    axs[1].plot(t, r, lw=0.8, color=col_, label=f"ACF, лаг {lag}")
axs[1].legend(loc="lower left"); axs[1].set_ylabel(f"окно {w}")
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
axs[0].set_title("штрих — истинные изменения (для примера); оранжевые — тревоги; серое — опорные участки")
st.pyplot(fig)

if alarms:
    res = pd.DataFrame([dict(тревога=e["alarm"], оценка_момента_изменения=e["change_est"],
                             порог=round(e["h"], 3)) for e in alarms])
    st.dataframe(res, width="stretch")
    st.download_button("Скачать результаты (CSV)", res.to_csv(index=False).encode("utf-8"),
                       "mlacs_alarms.csv", "text/csv")
else:
    st.success("Тревог нет: автокорреляционная структура на мониторинге не менялась "
               "(при заданной вероятности ложной тревоги).")
