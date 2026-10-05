"""Генерация кусочно-стационарных AR-процессов с сохранением дисперсии."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def ar_psi(phi, n_terms: int = 4000) -> np.ndarray:
    """Коэффициенты psi представления AR(p) в виде MA(inf): x_t = sum psi_j e_{t-j}."""
    phi = np.asarray(phi, dtype=float)
    p = len(phi)
    psi = np.zeros(n_terms)
    psi[0] = 1.0
    for j in range(1, n_terms):
        s = 0.0
        for i in range(1, min(p, j) + 1):
            s += phi[i - 1] * psi[j - i]
        psi[j] = s
    return psi


def ar_variance(phi, sigma: float = 1.0) -> float:
    """Дисперсия стационарного AR(p) при дисперсии шума sigma^2."""
    psi = ar_psi(phi)
    return float(sigma ** 2 * np.sum(psi ** 2))


def ar_acf(phi, max_lag: int = 10) -> np.ndarray:
    """Теоретическая ACF rho_1..rho_max_lag стационарного AR(p)."""
    psi = ar_psi(phi)
    g0 = np.sum(psi ** 2)
    return np.array([np.sum(psi[:-k] * psi[k:]) / g0 for k in range(1, max_lag + 1)])


def check_stationary(phi) -> bool:
    phi = np.asarray(phi, dtype=float)
    if len(phi) == 0:
        return True
    roots = np.roots(np.r_[1.0, -phi][::-1])  # корни 1 - phi1 z - ... = 0
    return bool(np.all(np.abs(roots) > 1.0))


@dataclass
class Segment:
    """Участок ряда: AR-коэффициенты, длина, среднее и целевая дисперсия ряда."""
    phi: tuple
    length: int
    mean: float = 0.0
    variance: float = 1.0          # дисперсия самого ряда (не шума!)
    noise_sigma: float | None = field(default=None)  # задать явно, если нужно

    def sigma(self) -> float:
        if self.noise_sigma is not None:
            return float(self.noise_sigma)
        # подбираем sigma шума так, чтобы Var(x) = variance
        return float(np.sqrt(self.variance / ar_variance(self.phi, 1.0)))


def simulate_piecewise_ar(segments, n_runs: int = 1, burn_in: int = 500,
                          rng: np.random.Generator | None = None,
                          noise: str = "normal") -> np.ndarray:
    """Моделирует n_runs независимых реализаций кусочно-стационарного AR-ряда.

    На границе участков меняются коэффициенты и дисперсия шума; состояние
    процесса (прошлые значения) переносится, поэтому переход непрерывен.
    Возвращает массив формы (n_runs, суммарная длина).
    """
    rng = np.random.default_rng() if rng is None else rng
    p = max(len(s.phi) for s in segments)
    total = sum(s.length for s in segments)
    n = burn_in + total
    if noise == "normal":
        e = rng.standard_normal((n_runs, n))
    elif noise == "t5":
        e = rng.standard_t(5, (n_runs, n)) / np.sqrt(5 / 3)
    else:
        raise ValueError(noise)
    # таблица параметров по времени
    phis = np.zeros((n, p))
    sig = np.zeros(n)
    mu = np.zeros(n)
    t = 0
    for k, s in enumerate(segments):
        L = s.length + (burn_in if k == 0 else 0)
        ph = np.zeros(p)
        ph[:len(s.phi)] = s.phi
        phis[t:t + L] = ph
        sig[t:t + L] = s.sigma()
        mu[t:t + L] = s.mean
        t += L
    y = np.zeros((n_runs, n))
    for t in range(n):
        acc = sig[t] * e[:, t]
        for i in range(p):
            if t - 1 - i >= 0 and phis[t, i] != 0.0:
                acc = acc + phis[t, i] * y[:, t - 1 - i]
        y[:, t] = acc
    y = y + mu[None, :]
    return y[:, burn_in:]
