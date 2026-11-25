"""Сохраняет распределение результатов процедуры «поиск + подтверждение» на рядах без предсказуемости (для рис. 5).
Та же симуляция, что в exp17 (вариант iid), но с сохранением t-статистик каждого прогона."""
import numpy as np
import pandas as pd

HO1, DEV_END = "2026-10-31", "2024-12-31"
rng = np.random.default_rng(42)


def nw_t(x, L=5):
    n = len(x)
    if n < 20: return 0.0
    e = x - x.mean(); v = e @ e / n + 2 * sum((1 - j / (L + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, L + 1))
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else 0.0


p_full = pd.read_csv("data/btc_yahoo_1d_long.csv", index_col=0, parse_dates=True)["close"]["2014-09-17":HO1]
r_full = np.log(p_full).diff().dropna().values; n_all = len(r_full); idx = p_full.index[1:]; n_dev = int((idx <= DEV_END).sum())
rows = []
for _ in range(2000):
    rs = rng.permutation(r_full); p = pd.Series(np.exp(np.cumsum(rs)), index=idx); r = pd.Series(rs, index=idx)
    sma = p.rolling(50).mean(); V = r.rolling(30).std().rolling(365, min_periods=180).rank(pct=True)
    ok = (sma.notna() & V.notna()).to_numpy(copy=True); ok[-1] = False
    code = (np.where(p > sma, 0, 3) + np.clip((V * 3).fillna(0).astype(int), 0, 2)).astype(int); nxt = r.shift(-1).values; pos = np.arange(n_all)
    dv, hv = ok & (pos < n_dev), ok & (pos >= n_dev)
    td = np.array([nw_t(nxt[dv][code[dv] == c]) for c in range(6)]); th = np.array([nw_t(nxt[hv][code[hv] == c]) for c in range(6)])
    b = int(np.abs(td).argmax()); rows.append((abs(td[b]), th[b] * np.sign(td[b])))
pd.DataFrame(rows, columns=["dev_max_abs_t", "holdout_t_same_sign"]).to_csv("results/phase4/exp17b_null_sims.csv", index=False)
a = np.array(rows); print("found:", (a[:, 0] > 2.64).mean(), "confirmed:", ((a[:, 0] > 2.64) & (a[:, 1] > 1.645)).mean(), "as strong:", ((a[:, 0] >= 3.48) & (a[:, 1] >= 2.72)).mean())
