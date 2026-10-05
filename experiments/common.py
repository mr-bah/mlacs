"""Общие настройки экспериментов."""
import numpy as np
import mlacs as m

N_REF = 2000        # опорный участок (40 блоков)
CHANGE = 3010       # момент изменения (намеренно не на границе блока)
N = 5000            # длина ряда
BLOCK = 50
K = 0.5
ALPHA = 0.05        # целевая вероятность ложной тревоги на 1000 наблюдений до изменения
N_CAL = 2000        # рядов для калибровки порога
N_TEST = 1000       # рядов для проверки

SCEN = {
    # имя: (участок до, участок после)
    "A": (dict(phi=(0.2,)), dict(phi=(0.6,))),
    "B": (dict(phi=(0.5,)), dict(phi=(0.3, 0.4))),
    "V": (dict(phi=(0.5,)), dict(phi=(0.5,), variance=2.0)),
    "M": (dict(phi=(0.5,)), dict(phi=(0.5,), mean=1.0)),
}


def interp(before, after, delta):
    """Промежуточный сценарий: коэффициенты AR — линейная интерполяция."""
    p = max(len(before["phi"]), len(after["phi"]))
    b = np.zeros(p); b[:len(before["phi"])] = before["phi"]
    a = np.zeros(p); a[:len(after["phi"])] = after["phi"]
    phi = tuple((1 - delta) * b + delta * a)
    d = dict(after); d["phi"] = phi
    return d


def make(before, after, n_runs, seed, n=N, change=CHANGE):
    rng = np.random.default_rng(seed)
    segs = [m.Segment(length=change, **before), m.Segment(length=n - change, **after)]
    return m.simulate_piecewise_ar(segs, n_runs=n_runs, rng=rng)


def make_null(before, n_runs, seed, n=CHANGE - 10):
    rng = np.random.default_rng(seed)
    return m.simulate_piecewise_ar([m.Segment(length=n, **before)], n_runs=n_runs, rng=rng)


def detectors(block=BLOCK):
    return [
        m.MeanDetector(block=block, k=K),
        m.VarianceDetector(block=block, k=K),
        m.ARResidualDetector(block=block, k=K),
        m.ACF1Detector(block=block, k=K),
        m.SpectralDetector(n_bands=4, block=block, k=K),
        m.MLACS(lags=2, block=block, k=K),
        m.MLACS(lags=3, block=block, k=K),
        m.MLACS(lags=5, block=block, k=K),
    ]


def evaluate(det, X, h, change=CHANGE, n_ref=N_REF):
    r = det.run(X, n_ref, h)
    at = r["alarm_time"]
    fa = (at > 0) & (at <= change)
    det_ok = at > change
    delay = (at - change)[det_ok]
    loc = np.abs(r["change_est"] - change)[det_ok]
    return dict(far=100 * fa.mean(), det_rate=100 * det_ok.mean(),
                mean_delay=float(delay.mean()) if len(delay) else np.nan,
                median_delay=float(np.median(delay)) if len(delay) else np.nan,
                loc_median=float(np.median(loc)) if len(loc) else np.nan,
                delays=delay, loc=loc)
