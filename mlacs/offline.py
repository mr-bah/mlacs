"""Офлайн-поиск одной точки разладки перебором (по схеме Бая–Перрона).

Для каждого кандидата tau ряд делится на две части, в каждой оценивается
регрессия (только константа или AR(p) с константой), выбирается tau с
минимальной суммарной стоимостью. Суммы XᵀX, Xᵀy, yᵀy берутся накопленными
суммами, поэтому весь перебор линеен по длине ряда.
"""
import numpy as np


def offline_single_split(y: np.ndarray, model: str = "ar", order: int = 2,
                         min_size: int = 200, jump: int = 5) -> int:
    """model: 'mean' — сдвиг среднего (RSS), 'meanvar' — среднего и дисперсии
    (гауссово правдоподобие), 'ar' — сдвиг AR(order)-коэффициентов (RSS)."""
    y = np.asarray(y, dtype=float)
    p = order if model == "ar" else 0
    t = y[p:]
    cols = [np.ones(len(t))] + [y[p - i:len(y) - i] for i in range(1, p + 1)]
    X = np.column_stack(cols)
    n, d = X.shape
    XX = np.concatenate([np.zeros((1, d, d)), np.cumsum(X[:, :, None] * X[:, None, :], axis=0)])
    Xy = np.concatenate([np.zeros((1, d)), np.cumsum(X * t[:, None], axis=0)])
    yy = np.concatenate([[0.0], np.cumsum(t * t)])
    cand = np.arange(min_size, n - min_size + 1, jump)

    def rss(a, b):
        A = XX[b] - XX[a]
        v = Xy[b] - Xy[a]
        beta = np.linalg.solve(A + 1e-10 * np.eye(d), v[..., None])[..., 0]
        return (yy[b] - yy[a]) - np.einsum("...i,...i->...", beta, v)

    r1 = rss(np.zeros_like(cand), cand)
    r2 = rss(cand, np.full_like(cand, n))
    if model == "meanvar":
        n1, n2 = cand, n - cand
        cost = n1 * np.log(r1 / n1) + n2 * np.log(r2 / n2)
    else:
        cost = r1 + r2
    return int(cand[np.argmin(cost)] + p)
