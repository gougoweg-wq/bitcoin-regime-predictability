"""Усиление 3, шаг A: событийное исследование. Что происходит через 5 дней после |Z|>lam
в каждой клетке (режим x знак Z)? Сравнение с обычной 5-дневной доходностью в том же режиме.
Train = до 2022, test = 2022-2024. Активы объединены; t-статистика по дневным средним (кластер по дате)."""
import json
import numpy as np
import pandas as pd
from lib import *

H, W = 5, 20
rows = []
for a in ASSETS:
    p = load(a); lp = np.log(p)
    z = (p - p.rolling(W).mean()) / p.rolling(W).std()
    df = pd.DataFrame({"asset": a, "z": z, "bull": regime_pos(p), "fwd": lp.shift(-H) - lp}).dropna()
    df["fwd_ex"] = df["fwd"] - df.groupby("bull")["fwd"].transform(lambda s: s.expanding().mean().shift(H))  # база только из прошлого
    rows.append(df)
D = pd.concat(rows).dropna()


def cell(d):
    by_date = d.groupby(level=0)["fwd_ex"].mean()          # одна точка на дату
    e = by_date.values - by_date.mean(); n = len(e); L = H
    v = e @ e / n + 2 * sum((1 - j / (L + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, min(L, n - 1) + 1))
    return {"N": len(d), "dates": n, "mean_fwd": round(float(d["fwd"].mean()), 4), "up_rate": round(float((d["fwd"] > 0).mean()), 3),
            "mean_excess": round(float(by_date.mean()), 4), "t": round(float(by_date.mean() / np.sqrt(v / n)), 2) if n > 10 else None}


out = {}
for lam in (1.9, 2.4):
    for period, d in (("train_to_2021", D[D.index < "2022-01-01"]), ("test_2022_2024", D[D.index >= "2022-01-01"]), ("all", D)):
        for reg in (1.0, 0.0):
            for side, m in (("Z>+lam", d.z > lam), ("Z<-lam", d.z < -lam)):
                key = f"lam{lam}|{period}|{'bull' if reg else 'bear'}|{side}"
                sub = d[m & (d.bull == reg)]
                if len(sub) >= 8:
                    out[key] = cell(sub); print(key, out[key])
json.dump(out, open("results/phase4/exp3_zsign_study.json", "w"), indent=1)
