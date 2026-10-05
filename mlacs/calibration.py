"""Калибровка порога h на заданную вероятность ложной тревоги."""
import numpy as np
from .generators import simulate_piecewise_ar, Segment
from .detectors import fit_ar_aic


def calibrate_threshold(detector, null_series: np.ndarray, n_ref: int,
                        alpha: float = 0.05) -> float:
    """h = (1-alpha)-квантиль максимума статистики на рядах без изменения."""
    S, _, _ = detector.statistic_path(null_series, n_ref)
    return float(np.quantile(S.max(axis=1), 1 - alpha))


def calibrate_threshold_bootstrap(detector, reference: np.ndarray,
                                  monitor_len: int, alpha: float = 0.05,
                                  n_sim: int = 300, p_max: int = 5,
                                  seed: int = 0) -> float:
    """Практическая калибровка без знания истинной модели: по опорному участку
    оценивается AR(p) (p по AIC), из неё моделируются n_sim рядов без
    изменения (параметрический бутстрэп), по ним — квантиль максимума CUSUM."""
    reference = np.asarray(reference, dtype=float)
    n_ref = len(reference)
    c, phi, s2 = fit_ar_aic(reference, p_max)
    mean = c / (1 - np.sum(phi))
    seg = Segment(phi=tuple(phi), length=n_ref + monitor_len, mean=mean,
                  noise_sigma=float(np.sqrt(s2)))
    rng = np.random.default_rng(seed)
    sims = simulate_piecewise_ar([seg], n_runs=n_sim, rng=rng)
    return calibrate_threshold(detector, sims, n_ref, alpha)
