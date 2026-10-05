"""Онлайн-детекторы на блочных признаках + последовательный CUSUM.

Все детекторы устроены одинаково:
  1) ряд режется на неперекрывающиеся блоки длины b;
  2) в каждом блоке считается вектор признаков f_j (d чисел);
  3) по опорному участку (первые n_ref наблюдений) оцениваются среднее mu0
     и ковариация Sigma0 признаков;
  4) для каждого нового блока считается нормированное отклонение z_j,
     которое накапливается статистикой CUSUM; тревога — при S_j > h.
Детекторы различаются только признаками — это делает сравнение честным.
"""
from __future__ import annotations
import numpy as np
from .features import block_acf, block_spectrum, block_mean, block_logvar


class BlockCusum:
    """Базовый класс: CUSUM по блочным признакам."""
    name = "base"

    def __init__(self, block: int = 50, k: float = 0.5, shrink: float = 0.2):
        self.block = int(block)
        self.k = float(k)
        self.shrink = float(shrink)

    # --- признаки (переопределяется в наследниках) ---
    def features(self, x: np.ndarray, n_ref: int) -> np.ndarray:
        raise NotImplementedError

    @property
    def two_sided_scalar(self) -> bool:
        return True

    # --- основная процедура ---
    def statistic_path(self, x: np.ndarray, n_ref: int):
        """Возвращает (S, zero_last, side):
        S         — (R, M) путь статистики CUSUM по M блокам мониторинга;
        zero_last — (R, M) индекс последнего блока с S=0 (для локализации);
        side      — (R, M) для скалярного двустороннего варианта: +1/-1.
        """
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if n_ref % self.block:
            raise ValueError("n_ref должно быть кратно длине блока")
        F = self.features(x, n_ref)                  # (R, nb, d)
        m = n_ref // self.block
        ref, mon = F[:, :m], F[:, m:]
        R, M, d = mon.shape
        mu = ref.mean(axis=1)                        # (R, d)
        S = np.zeros((R, M))
        zl = np.full((R, M), -1, dtype=int)
        side = np.zeros((R, M), dtype=int)
        if d == 1 and self.two_sided_scalar:
            sd = ref[..., 0].std(axis=1, ddof=1)
            z = (mon[..., 0] - mu) / sd[:, None]      # (R, M)
            sp = np.zeros(R); sm = np.zeros(R)
            lp = np.full(R, -1); lm = np.full(R, -1)
            for j in range(M):
                sp = np.maximum(0.0, sp + z[:, j] - self.k)
                sm = np.maximum(0.0, sm - z[:, j] - self.k)
                lp = np.where(sp == 0, j, lp)
                lm = np.where(sm == 0, j, lm)
                up = sp >= sm
                S[:, j] = np.where(up, sp, sm)
                zl[:, j] = np.where(up, lp, lm)
                side[:, j] = np.where(up, 1, -1)
            return S, zl, side
        # многомерный случай: квадрат расстояния Махаланобиса
        C = np.einsum("rmi,rmj->rij", ref - mu[:, None], ref - mu[:, None]) / (m - 1)
        diag = np.einsum("rii->ri", C)
        C = (1 - self.shrink) * C + self.shrink * np.einsum("ri,ij->rij", diag, np.eye(d))
        Ci = np.linalg.inv(C)
        dev = mon - mu[:, None]
        D2 = np.einsum("rmi,rij,rmj->rm", dev, Ci, dev)
        z = (D2 - d) / np.sqrt(2 * d)
        s = np.zeros(R); last = np.full(R, -1)
        for j in range(M):
            s = np.maximum(0.0, s + z[:, j] - self.k)
            last = np.where(s == 0, j, last)
            S[:, j] = s
            zl[:, j] = last
        side[:] = 1
        return S, zl, side

    def run(self, x, n_ref: int, h: float):
        """Первая тревога для каждой реализации.

        Возвращает словарь: alarm_block (индекс блока мониторинга или -1),
        alarm_time (абсолютное время конца блока тревоги или -1),
        change_est (оценка момента изменения или -1), S (путь статистики).
        """
        S, zl, _ = self.statistic_path(x, n_ref)
        over = S > h
        has = over.any(axis=1)
        j = np.where(has, over.argmax(axis=1), -1)
        b = self.block
        alarm_time = np.where(has, n_ref + (j + 1) * b, -1)
        z_at = np.where(has, zl[np.arange(len(j)), np.maximum(j, 0)], -1)
        change_est = np.where(has, n_ref + (z_at + 1) * b, -1)
        return dict(alarm_block=j, alarm_time=alarm_time,
                    change_est=change_est, S=S)


class MLACS(BlockCusum):
    """Предлагаемый детектор: вектор ACF на лагах 1..L + CUSUM по расстоянию
    Махаланобиса (MLACS — Multi-Lag AutoCorrelation Shift detector)."""
    def __init__(self, lags: int = 3, **kw):
        super().__init__(**kw)
        self.lags = int(lags)
        self.name = f"MLACS (L={self.lags})"

    @property
    def two_sided_scalar(self):
        return False            # при L=1 — тоже хи-квадрат вариант (для ряда по L)

    def features(self, x, n_ref):
        return block_acf(x, self.block, self.lags)


class ACF1Detector(BlockCusum):
    """Однолаговый детектор: двусторонний CUSUM по r_1 блока (вариант «А»)."""
    name = "ACF-1 (лаг 1)"

    def features(self, x, n_ref):
        return block_acf(x, self.block, 1)


class SpectralDetector(BlockCusum):
    """Спектральный вариант: доли мощности в частотных полосах + тот же CUSUM."""
    def __init__(self, n_bands: int = 4, **kw):
        super().__init__(**kw)
        self.n_bands = n_bands
        self.name = f"Спектр ({n_bands} полосы)"

    def features(self, x, n_ref):
        return block_spectrum(x, self.block, self.n_bands)


class MeanDetector(BlockCusum):
    name = "Среднее"

    def features(self, x, n_ref):
        return block_mean(x, self.block)


class VarianceDetector(BlockCusum):
    name = "Дисперсия"

    def features(self, x, n_ref):
        return block_logvar(x, self.block)


def fit_ar_aic(y: np.ndarray, p_max: int = 5):
    """OLS-оценка AR(p) с константой, порядок по AIC. Возвращает (c, phi, sigma2)."""
    y = np.asarray(y, dtype=float)
    n = len(y) - p_max
    best = None
    for p in range(1, p_max + 1):
        X = np.column_stack([np.ones(n)] + [y[p_max - i:len(y) - i] for i in range(1, p + 1)])
        t = y[p_max:]
        beta, *_ = np.linalg.lstsq(X, t, rcond=None)
        rss = float(np.sum((t - X @ beta) ** 2))
        aic = n * np.log(rss / n) + 2 * (p + 1)
        if best is None or aic < best[0]:
            best = (aic, beta[0], beta[1:], rss / (n - p - 1))
    return best[1], best[2], best[3]


class ARResidualDetector(BlockCusum):
    """Классический подход: AR(p) по опорному участку (p по AIC), затем
    двусторонний CUSUM по log среднего квадрата остатков прогноза в блоке."""
    name = "Остатки AR(p)"

    def __init__(self, p_max: int = 5, **kw):
        super().__init__(**kw)
        self.p_max = p_max

    def features(self, x, n_ref):
        R, n = x.shape
        b = self.block
        nb = n // b
        out = np.empty((R, nb, 1))
        for r in range(R):
            c, phi, _ = fit_ar_aic(x[r, :n_ref], self.p_max)
            p = len(phi)
            pred = np.full(n, c)
            for i in range(1, p + 1):
                pred[i:] += phi[i - 1] * x[r, :-i]
            e = x[r] - pred
            e[:p] = e[p]                          # первые p точек без прошлого
            out[r, :, 0] = np.log(np.mean(e[:nb * b].reshape(nb, b) ** 2, axis=1))
        return out
