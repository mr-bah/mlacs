"""Блочные признаки: ACF по нескольким лагам, спектральные доли, среднее, дисперсия."""
import numpy as np


def _blocks(x: np.ndarray, b: int) -> np.ndarray:
    """(..., n) -> (..., n//b, b), хвост отбрасывается."""
    x = np.asarray(x, dtype=float)
    nb = x.shape[-1] // b
    return x[..., :nb * b].reshape(*x.shape[:-1], nb, b)


def block_acf(x: np.ndarray, b: int, max_lag: int) -> np.ndarray:
    """Выборочная ACF r_1..r_L в каждом неперекрывающемся блоке длины b.

    r_k = sum_{t=k+1}^{b} (x_t - m)(x_{t-k} - m) / sum_{t=1}^{b} (x_t - m)^2,
    m — среднее блока. Возвращает (..., n_blocks, L).
    """
    B = _blocks(x, b)
    B = B - B.mean(axis=-1, keepdims=True)
    den = np.sum(B * B, axis=-1)
    out = np.empty(B.shape[:-1] + (max_lag,))
    for k in range(1, max_lag + 1):
        out[..., k - 1] = np.sum(B[..., k:] * B[..., :-k], axis=-1) / den
    return out


def block_spectrum(x: np.ndarray, b: int, n_bands: int = 4) -> np.ndarray:
    """Доли мощности периодограммы блока в n_bands равных частотных полосах.

    Нормировка на полную мощность делает признак нечувствительным к масштабу
    (как и ACF). Последняя полоса отбрасывается (сумма долей = 1).
    Возвращает (..., n_blocks, n_bands - 1).
    """
    B = _blocks(x, b)
    B = B - B.mean(axis=-1, keepdims=True)
    P = np.abs(np.fft.rfft(B, axis=-1)) ** 2
    P = P[..., 1:]                       # без нулевой частоты
    edges = np.linspace(0, P.shape[-1], n_bands + 1).astype(int)
    bands = np.stack([P[..., edges[i]:edges[i + 1]].sum(axis=-1)
                      for i in range(n_bands)], axis=-1)
    frac = bands / bands.sum(axis=-1, keepdims=True)
    return frac[..., :-1]


def block_mean(x: np.ndarray, b: int) -> np.ndarray:
    return _blocks(x, b).mean(axis=-1)[..., None]


def block_logvar(x: np.ndarray, b: int) -> np.ndarray:
    return np.log(_blocks(x, b).var(axis=-1, ddof=1))[..., None]
