"""Фаза 1: воспроизведение цифр работы «Regime-Based Statistical Modelling of Bitcoin».

Трактовка, которая даёт цифры работы (найдена перебором вариантов):
  - Yahoo BTC-USD, дневной close, 2017-01-01..2024-12-31 (2922 точки)
  - режим: P_t > SMA50, где SMA50 ВКЛЮЧАЕТ P_t (rolling(50).mean())
  - статистики режима: доходность ТОГО ЖЕ дня t, что и метка режима
  - Z_t = (P_t - mean20) / std20 (окно включает P_t, ddof=1)
  - прогноз: sign(Z_t) = знак лог-доходности за следующие 5 дней
Запуск: .venv/bin/python src/baseline.py
"""
import json
from math import exp
import numpy as np
import pandas as pd
from scipy.stats import binom

START, END, SMA_W, Z_W, H = "2017-01-01", "2024-12-31", 50, 20, 5


def binom_p(k, n, p0=0.5):
    """Точный односторонний p-value P(X >= k), X ~ Bin(n, p0)."""
    return float(binom.sf(k - 1, n, p0))


def run(path):
    p = pd.read_csv(path, index_col=0, parse_dates=True).loc[START:END, "close"]
    lp = np.log(p)
    r = lp.diff()
    out = {"n_points": len(p)}

    # --- режимы (разд. 3.3, 5.1)
    sma = p.rolling(SMA_W).mean()
    bull = (p > sma)[sma.notna()]
    rr, nx = r[bull.index], r.shift(-1)[bull.index]
    T = pd.crosstab(bull.shift(1)[1:], bull[1:], normalize="index")
    out["regime"] = {
        "n_valid": len(bull), "n_bull": int(bull.sum()), "n_bear": int((~bull).sum()),
        "same_day": {"mu_bull": rr[bull].mean(), "sd_bull": rr[bull].std(),
                     "mu_bear": rr[~bull].mean(), "sd_bear": rr[~bull].std()},
        "next_day": {"mu_bull": nx[bull].mean(), "sd_bull": nx[bull].std(),
                     "mu_bear": nx[~bull].mean(), "sd_bear": nx[~bull].std()},
        "T_bull_bull": T.loc[True, True], "T_bear_bear": T.loc[False, False],
    }

    # --- Master formula при r_t = 0 (разд. 5.2), параметры same-day
    s = out["regime"]["same_day"]
    pi = out["regime"]["n_bull"] / out["regime"]["n_valid"]
    lb = exp(-0.5 * (s["mu_bull"] / s["sd_bull"]) ** 2) / s["sd_bull"] * pi
    le = exp(-0.5 * (s["mu_bear"] / s["sd_bear"]) ** 2) / s["sd_bear"] * (1 - pi)
    out["posterior_bull_r0"] = lb / (lb + le)

    # --- Z-фильтр (разд. 6)
    z = (p - p.rolling(Z_W).mean()) / p.rolling(Z_W).std()
    fwd = lp.shift(-H) - lp
    out["base_rate_up_5d"] = float((fwd.dropna() > 0).mean())
    out["zfilter"] = {}
    for lam in (0.5, 1.9, 2.4):
        m = (z.abs() > lam) & fwd.notna()
        hit = np.sign(z[m]) == np.sign(fwd[m])
        k, n = int(hit.sum()), int(m.sum())
        up = int((fwd[m] > 0).sum())
        clusters = int((m & ~m.shift(1, fill_value=False)).sum())  # серии подряд идущих дней
        out["zfilter"][str(lam)] = {
            "N": n, "hits": k, "acc": k / n, "p_vs_50": binom_p(k, n),
            "always_long_acc": up / n, "share_Z_positive": float((z[m] > 0).mean()),
            "clusters": clusters,
            "nonoverlap_events": int(len(_nonoverlap(m[m].index, H))),
        }
    return out


def _nonoverlap(idx, h):
    """Жадно оставляет события, отстоящие друг от друга минимум на h дней."""
    keep, last = [], None
    for t in idx:
        if last is None or (t - last).days >= h:
            keep.append(t); last = t
    return keep


if __name__ == "__main__":
    res = {"yahoo": run("data/btc_yahoo_1d.csv"), "binance": run("data/btc_binance_1d.csv")}
    json.dump(res, open("results/baseline.json", "w"), indent=2, default=float)
    print(json.dumps(res, indent=2, default=float))
