"""Усиление 7, шаг A: событийное исследование позиционирования.
Гипотеза (записана до расчёта): после экстремального Z исход зависит от перегрева толпы.
  Z > +lam и высокий фандинг / рост OI -> хуже (разворот); Z > +lam и низкий фандинг -> лучше (продолжение).
  Z < -lam и низкий (отрицательный) фандинг -> отскок.
Признаки (только прошлое): F = средний фандинг за 3 дня до close t; dOI = изменение OI за 5 дней.
Оба переводятся в ранг 0..1 по расширяющемуся окну внутри актива. high = ранг > 2/3, low = ранг < 1/3.
Исход: 5-дневная лог-доходность минус обычная для этого режима (база из прошлого). Train 2020-2022, test 2023-2024."""
import json
import numpy as np
import pandas as pd
from lib import *

H, W = 5, 20


def exp_rank(s, minp=180):
    return s.expanding(minp).apply(lambda x: (x[:-1] < x[-1]).sum() / max(1, (~np.isnan(x[:-1])).sum()), raw=True)   # NaN в начале ряда не считаем


def features(a):
    p = load(a); lp = np.log(p)
    f = pd.read_csv(f"data/{a}_funding.csv", index_col=0, parse_dates=True)["last_funding_rate"]
    f.index = f.index.floor("s") + pd.Timedelta(seconds=1)          # расчёт в 00:00 дня t+1 относим к дню t+1, не t
    F = f.resample("1D").mean().rolling(3).mean().reindex(p.index)
    oi = pd.read_csv(f"data/{a}_oi.csv", index_col=0, parse_dates=True)["oi"].reindex(p.index)
    z = (p - p.rolling(W).mean()) / p.rolling(W).std()
    df = pd.DataFrame({"asset": a, "z": z, "bull": regime_pos(p), "fwd": lp.shift(-H) - lp, "F": F, "F_rank": exp_rank(F),
                       "dOI_rank": exp_rank(oi.pct_change(5))})
    df["fwd_ex"] = df["fwd"] - df.groupby("bull")["fwd"].transform(lambda s: s.expanding().mean().shift(H))
    return df


def tstat(by_date):
    e = by_date.values - by_date.mean(); n = len(e)
    v = e @ e / n + 2 * sum((1 - j / (H + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, min(H, n - 1) + 1))
    return float(by_date.mean() / np.sqrt(v / n))


def compare(d, col):
    """Разница исхода: high минус low по признаку col, одна точка на дату."""
    hi, lo = d[d[col] > 2 / 3], d[d[col] < 1 / 3]
    if len(hi) < 15 or len(lo) < 15:
        return None
    g = pd.concat([hi.groupby(level=0)["fwd_ex"].mean().rename("v").to_frame().assign(g=1),
                   lo.groupby(level=0)["fwd_ex"].mean().rename("v").to_frame().assign(g=0)]).sort_index()
    x = g["g"].values.astype(float); y = g["v"].values
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]; u = X * (y - X @ b)[:, None]
    S = u.T @ u + sum((1 - j / (H + 1)) * (u[j:].T @ u[:-j] + u[:-j].T @ u[j:]) for j in range(1, H + 1))
    V = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
    return {"N_high": len(hi), "N_low": len(lo), "mean_ex_high": round(float(hi.fwd_ex.mean()), 4), "mean_ex_low": round(float(lo.fwd_ex.mean()), 4),
            "up_high": round(float((hi.fwd > 0).mean()), 3), "up_low": round(float((lo.fwd > 0).mean()), 3),
            "diff_high_minus_low": round(float(b[1]), 4), "t": round(float(b[1] / np.sqrt(V[1, 1])), 2)}


if __name__ == "__main__":
    D = pd.concat([features(a) for a in ASSETS]).dropna(subset=["z", "bull", "fwd_ex", "F_rank"])
    D = D[D.index >= "2020-07-01"]
    out = {}
    sets = {"Z>+1.9": D.z > 1.9, "Z<-1.9": D.z < -1.9, "Z>+2.4": D.z > 2.4, "Z<-2.4": D.z < -2.4,
            "bull_all_days": D.bull == 1, "bear_all_days": D.bull == 0}
    for period, pm in (("train_2020_2022", D.index < "2023-01-01"), ("test_2023_2024", D.index >= "2023-01-01"), ("all", D.index >= "2000")):
        for sname, sm in sets.items():
            for col in ("F_rank", "dOI_rank"):
                d = D[pm & sm].dropna(subset=[col]); res = compare(d, col)
                if res:
                    out[f"{period}|{sname}|{col}"] = res; print(f"{period:16s} {sname:14s} {col:9s}", res)
    json.dump(out, open("results/phase4/exp7_positioning_study.json", "w"), indent=1)
