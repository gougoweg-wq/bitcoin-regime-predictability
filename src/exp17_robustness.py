"""Усиление 17: проверки на случайность (сделаны ПОСЛЕ hold-out, новых торговых гипотез не вводят).
  A) Перестановочный max-t тест: метки типов рынка циклически сдвигаются относительно доходностей (автокорреляция обоих
     рядов сохраняется, связь между ними рвётся). Нулевое распределение максимума |t| по шести клеткам -> p-value с учётом
     того, что лучшая клетка выбрана из шести.
  B) Симуляция всей процедуры на рядах без предсказуемости: доходности BTC перемешиваются (iid и блоками по 5 дней),
     из них строится цена, по ней заново считаются типы рынка. Доля симуляций, где процедура «находит» клетку в разработке
     (|t| > 2.64) и «подтверждает» её на отложенной части (t того же знака, |t| > 1.645).
  C) Мощность: сколько дней нужно, чтобы различить наблюдаемые эффекты.
  D) Хвостовой риск: VaR и Expected Shortfall, число степеней свободы t-распределения по режимам."""
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student
sys.path.insert(0, "src")
from exp10_map import state

DEV_END, HO0, HO1 = "2024-12-31", "2025-01-01", "2026-10-31"
rng = np.random.default_rng(42); OUT = {}


def nw_t(x, L=5):
    n = len(x)
    if n < 20: return 0.0
    e = x - x.mean(); v = e @ e / n + 2 * sum((1 - j / (L + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, L + 1))
    return float(x.mean() / np.sqrt(v / n)) if v > 0 else 0.0


def cell_t(codes, r):
    return np.array([nw_t(r[codes == c]) for c in range(6)])


def codes_of(d):
    return (np.where(d["bull"].astype(bool), 0, 3) + d["vol"].cat.codes.values).astype(int)   # 0..2 Bull low/mid/high, 3..5 Bear


d = state("data/btc_yahoo_1d_long.csv", "2014-09-17", HO1)
dev, ho = d[:DEV_END], d[HO0:]

# ---------- A
A = {}
for name, part in (("development", dev), ("holdout", ho)):
    c, r = codes_of(part), part["r_next"].values; obs = cell_t(c, r); n = len(r)
    null = np.array([np.abs(cell_t(np.roll(c, int(s)), r)).max() for s in rng.integers(30, n - 30, 5000)])
    A[name] = {"t_by_cell": [round(float(x), 2) for x in obs], "bull_mid_t": round(float(obs[1]), 2), "max_abs_t": round(float(np.abs(obs).max()), 2),
               "familywise_p_of_bull_mid": float((null >= abs(obs[1])).mean()), "null_95pct_of_max_abs_t": round(float(np.percentile(null, 95)), 2),
               "single_cell_p_bull_mid": float((np.array([cell_t(np.roll(c, int(s)), r)[1] for s in rng.integers(30, n - 30, 5000)]) >= obs[1]).mean())}
OUT["A_permutation"] = A; print("A", json.dumps(A))

# ---------- B
p_full = pd.read_csv("data/btc_yahoo_1d_long.csv", index_col=0, parse_dates=True)["close"]["2014-09-17":HO1]
r_full = np.log(p_full).diff().dropna().values; n_all = len(r_full); idx = p_full.index[1:]
n_dev_rows = int((idx <= DEV_END).sum())


def simulate(block):
    if block == 1:
        rs = rng.permutation(r_full)
    else:
        nb = int(np.ceil(n_all / block)); rs = r_full[(rng.integers(0, n_all, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n_all] % n_all]
    p = pd.Series(np.exp(np.cumsum(rs)), index=idx); r = pd.Series(rs, index=idx)
    sma = p.rolling(50).mean(); rv = r.rolling(30).std(); V = rv.rolling(365, min_periods=180).rank(pct=True)
    ok = sma.notna() & V.notna(); ok.iloc[-1] = False
    code = (np.where(p > sma, 0, 3) + np.clip((V * 3).fillna(0).astype(int), 0, 2)).astype(int)
    nxt = r.shift(-1).values; m = ok.values; pos = np.arange(n_all)
    dv, hv = m & (pos < n_dev_rows), m & (pos >= n_dev_rows)
    return cell_t(code[dv], nxt[dv]), cell_t(code[hv], nxt[hv])


B = {}
for block, label in ((1, "iid_shuffle"), (5, "block_5_days")):
    sims = [simulate(block) for _ in range(2000)]
    td = np.array([s[0] for s in sims]); th = np.array([s[1] for s in sims])
    found = np.abs(td).max(axis=1) > 2.64; best = np.abs(td).argmax(axis=1); ar = np.arange(len(td))
    conf = found & (np.sign(th[ar, best]) == np.sign(td[ar, best])) & (np.abs(th[ar, best]) > 1.645)
    as_strong = (np.abs(td[ar, best]) >= 3.48) & (np.sign(th[ar, best]) == np.sign(td[ar, best])) & (np.abs(th[ar, best]) >= 2.72)
    B[label] = {"simulations": len(td), "share_finding_a_cell_in_development": float(found.mean()), "share_found_and_confirmed_on_holdout": float(conf.mean()),
                "share_as_strong_as_observed": float(as_strong.mean()), "null_95pct_max_abs_t_development": round(float(np.percentile(np.abs(td).max(axis=1), 95)), 2)}
OUT["B_simulation"] = B; print("B", json.dumps(B))

# ---------- C
def n_for_freq(p1, p0=0.5, a=0.05, pw=0.8):
    return int(np.ceil(((norm.ppf(1 - a) * np.sqrt(p0 * (1 - p0)) + norm.ppf(pw) * np.sqrt(p1 * (1 - p1))) / (p1 - p0)) ** 2))
def n_for_mean(mu, sd, a=0.05, pw=0.8):
    return int(np.ceil(((norm.ppf(1 - a) + norm.ppf(pw)) * sd / mu) ** 2))
c_dev = dev[codes_of(dev) == 1]["r_next"]
OUT["C_power"] = {"days_to_detect_56.3pct_vs_50pct": n_for_freq(0.563), "days_to_detect_55pct_vs_50pct": n_for_freq(0.55), "days_to_detect_52pct_vs_50pct": n_for_freq(0.52),
                  "bull_mid_daily_mean": float(c_dev.mean()), "bull_mid_daily_sd": float(c_dev.std()), "days_to_detect_bull_mid_mean": n_for_mean(c_dev.mean(), c_dev.std()),
                  "power_of_holdout_freq_test_n95": float(1 - norm.cdf((norm.ppf(0.95) * 0.5 - (0.563 - 0.5) * np.sqrt(95)) / np.sqrt(0.563 * 0.437))),
                  "bull_mid_days_per_year": round(float(len(c_dev) / (len(dev) / 365)), 0)}
print("C", json.dumps(OUT["C_power"]))

# ---------- D
q = pd.read_csv("data/btc_binance_1d.csv", index_col=0, parse_dates=True)["close"][:HO1]; b = (q > q.rolling(50).mean()).astype(float); held = b.shift(1)
reg = (held * q.pct_change() - held.diff().abs().fillna(0) * 10 / 1e4).dropna()["2017-10-06":]; bh = q.pct_change()["2017-10-06":]
def tail(x):
    v95, v99 = np.percentile(x, 5), np.percentile(x, 1)
    return {"VaR95_pct": round(float(-v95 * 100), 2), "ES95_pct": round(float(-x[x <= v95].mean() * 100), 2), "VaR99_pct": round(float(-v99 * 100), 2),
            "ES99_pct": round(float(-x[x <= v99].mean() * 100), 2), "worst_day_pct": round(float(x.min() * 100), 2), "ann_vol_pct": round(float(x.std() * np.sqrt(365) * 100), 1)}
D = {per: {"regime_rule": tail(reg[a:z].values), "buy_and_hold": tail(bh[a:z].values)} for per, a, z in (("development_2017_2024", None, DEV_END), ("holdout_2025_2026", HO0, HO1))}
lr = np.log(q).diff(); nx = lr.shift(-1)["2017-10-06":DEV_END].dropna(); bb = b[nx.index]
for nm, x in (("bull", nx[bb == 1].values), ("bear", nx[bb == 0].values), ("all", nx.values)):
    nu, loc, sc = student.fit(x); D[f"student_t_fit_{nm}"] = {"nu": round(float(nu), 2), "scale_pct": round(float(sc * 100), 2), "n": len(x)}
OUT["D_tail_risk"] = D; print("D", json.dumps(D))
json.dump(OUT, open("results/phase4/exp17_robustness.json", "w"), indent=1)
