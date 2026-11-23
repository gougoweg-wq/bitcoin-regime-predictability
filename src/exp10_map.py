"""Усиление 10, шаг A: карта предсказуемости. В каких типах рынка направление BTC на завтра отличается от монетки?
Состояние на закрытии дня t (только прошлое):
  тренд:  Bull, если P_t > SMA50, иначе Bear
  волатильность: 30-дневная реализованная, ранг в скользящем окне 365 дней; low < 1/3, high > 2/3
Исход: лог-доходность дня t+1. Данные Yahoo BTC-USD 2014-11..2024-12 (длиннее истории нет), сверка на Binance.
Train = до 2021 включительно, test = 2022-2024."""
import json
import numpy as np
import pandas as pd
from scipy.stats import norm, binomtest


def state(path, start="2014-09-17", end="2024-12-31"):
    p = pd.read_csv(path, index_col=0, parse_dates=True).loc[start:end, "close"]; r = np.log(p).diff()
    sma = p.rolling(50).mean(); rv = r.rolling(30).std()
    d = pd.DataFrame({"p": p, "r_next": r.shift(-1), "bull": (p > sma).where(sma.notna()),
                      "T": np.log(p / sma) / (rv * np.sqrt(50)),                       # тренд в сигмах
                      "V": rv.rolling(365, min_periods=180).rank(pct=True)}).dropna()   # ранг волатильности 0..1
    d["vol"] = pd.cut(d["V"], [0, 1 / 3, 2 / 3, 1.0001], labels=["low", "mid", "high"], right=False)
    return d


def nw_t(x, L=5):
    x = np.asarray(x); n = len(x); e = x - x.mean()
    v = e @ e / n + 2 * sum((1 - j / (L + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, L + 1))
    return float(x.mean() / np.sqrt(v / n))


def table(d):
    out = {}
    for bull in (1.0, 0.0):
        for vol in ("low", "mid", "high"):
            c = d[(d.bull == bull) & (d.vol == vol)]
            if len(c) < 30:
                continue
            up = int((c.r_next > 0).sum())
            out[f"{'bull' if bull else 'bear'}|{vol}"] = {
                "days": len(c), "up_rate": round(up / len(c), 3), "p_vs_50": round(float(binomtest(up, len(c), 0.5).pvalue), 4),
                "mean_next_pct": round(float(c.r_next.mean() * 100), 3), "t_nw": round(nw_t(c.r_next.values), 2),
                "sharpe_ann": round(float(c.r_next.mean() / c.r_next.std() * np.sqrt(365)), 2)}
    return out


if __name__ == "__main__":
    out = {}
    for name, path, start in (("yahoo", "data/btc_yahoo_1d.csv", "2014-09-17"), ("binance", "data/btc_binance_1d.csv", "2017-08-17")):
        if name == "yahoo":      # в репозитории Yahoo лежит с 2016-10; для длинной истории докачиваем 2014+
            import yfinance as yf, os
            if not os.path.exists("data/btc_yahoo_1d_long.csv"):
                y = yf.download("BTC-USD", start="2014-09-01", end="2026-11-23", interval="1d", auto_adjust=False, progress=False)
                y.columns = [c[0].lower() for c in y.columns]; y.index.name = "date"; y[["open", "high", "low", "close", "volume"]].to_csv("data/btc_yahoo_1d_long.csv")
            path = "data/btc_yahoo_1d_long.csv"
        d = state(path, start)
        out[name] = {"all": table(d), "train_to_2021": table(d[:"2021-12-31"]), "test_2022_2024": table(d["2022-01-01":]),
                     "base_up_rate": round(float((d.r_next > 0).mean()), 3), "n": len(d), "from": str(d.index[0].date())}
        print(f"\n===== {name}: {out[name]['from']}..2024-12-31, дней {len(d)}, базовая частота роста {out[name]['base_up_rate']}")
        for per in ("all", "train_to_2021", "test_2022_2024"):
            print(f"--- {per}")
            for k, v in out[name][per].items():
                print(f"  {k:10s}", v)
    json.dump(out, open("results/phase4/exp10_map.json", "w"), indent=1)
