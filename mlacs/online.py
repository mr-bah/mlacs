"""Последовательный (потоковый) режим для нескольких точек изменения."""
import numpy as np
from .detectors import MLACS
from .calibration import calibrate_threshold_bootstrap


def detect_sequential(y, lags: int = 3, block: int = 50, k: float = 0.5,
                      n_ref: int = 2000, alpha: float = 0.05, horizon: int = 1000,
                      n_sim: int = 200, seed: int = 0, detector=None):
    """Мониторинг ряда y с перезапуском после каждой тревоги.

    1) первые n_ref наблюдений — опорный участок; порог h — параметрическим
       бутстрэпом так, чтобы P(ложная тревога на horizon наблюдений) = alpha;
    2) при тревоге фиксируются момент тревоги и оценка момента изменения;
    3) новый опорный участок начинается с момента тревоги, шаг 1 повторяется.
    Возвращает список словарей: alarm, change_est, h, segment_start, S (путь).
    """
    y = np.asarray(y, dtype=float)
    det = detector or MLACS(lags=lags, block=block, k=k)
    events, start, it = [], 0, 0
    while start + n_ref + det.block <= len(y):
        seg = y[start:]
        h = calibrate_threshold_bootstrap(det, seg[:n_ref], horizon, alpha,
                                          n_sim=n_sim, seed=seed * 10 + it)
        r = det.run(seg[None, :], n_ref, h)
        ev = dict(segment_start=start, h=h, S=r["S"][0],
                  alarm=int(r["alarm_time"][0]), change_est=int(r["change_est"][0]))
        if ev["alarm"] < 0:
            events.append(ev)          # последний участок без тревоги (для графика)
            break
        ev["alarm"] += start
        ev["change_est"] += start
        events.append(ev)
        start = ev["alarm"]
        it += 1
    return events
