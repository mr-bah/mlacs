"""Командная строка:  python -m mlacs data.csv --column value [--lags 3 ...]

Читает одну колонку CSV (или первый числовой столбец), выполняет
последовательный мониторинг и печатает найденные тревоги.
"""
import argparse, sys
import numpy as np
import pandas as pd
from .online import detect_sequential


def main(argv=None):
    p = argparse.ArgumentParser(prog="mlacs", description="Поиск скрытых автокорреляционных сдвигов (MLACS)")
    p.add_argument("csv"); p.add_argument("--column", default=None)
    p.add_argument("--lags", type=int, default=3); p.add_argument("--block", type=int, default=50)
    p.add_argument("--k", type=float, default=0.5); p.add_argument("--n-ref", type=int, default=2000)
    p.add_argument("--alpha", type=float, default=0.05); p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--out", default=None)
    a = p.parse_args(argv)
    df = pd.read_csv(a.csv)
    col = a.column or df.select_dtypes("number").columns[0]
    y = df[col].dropna().to_numpy(float)
    n_ref = (a.n_ref // a.block) * a.block
    ev = detect_sequential(y, lags=a.lags, block=a.block, k=a.k, n_ref=n_ref,
                           alpha=a.alpha, horizon=a.horizon, seed=a.seed)
    rows = [dict(alarm=e["alarm"], change_est=e["change_est"], threshold=round(e["h"], 3))
            for e in ev if e["alarm"] >= 0]
    print(f"Ряд '{col}', n = {len(y)}; найдено тревог: {len(rows)}")
    for r in rows:
        print(f"  тревога в t = {r['alarm']:>7}   оценка момента изменения = {r['change_est']:>7}")
    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
