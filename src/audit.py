"""Фаза 2: аудит на прочность. Только 2017-01-01..2024-12-31; 2025+ = hold-out, не трогаем.

Ядро работы переведено в две минимальные стратегии (в работе PnL нет):
  REGIME: long, если P_t > SMA50_t, иначе вне рынка; сигнал на close t, держим день t+1.
  ZMOM  : при |Z_t| > lam позиция sign(Z_t) на h дней (перекрывающиеся сигналы усредняются).
Запуск: .venv/bin/python src/audit.py
"""
import json
from itertools import combinations
import numpy as np
import pandas as pd
from scipy.stats import norm, binom, kurtosis, skew, jarque_bera

START, END, ANN = "2017-01-01", "2024-12-31", 365
rng = np.random.default_rng(42)
R = {}


def load(name):
    return pd.read_csv(f"data/{name}_1d.csv", index_col=0, parse_dates=True).loc[START:END]


def regime_pos(p, w=50):
    return (p > p.rolling(w).mean()).astype(float).where(p.rolling(w).mean().notna())


def z_pos(p, w=20, lam=2.4, h=5):
    z = (p - p.rolling(w).mean()) / p.rolling(w).std()
    sig = np.sign(z).where(z.abs() > lam, 0.0)
    return sig.rolling(h).mean().where(z.notna())


def strat(p, pos, cost_bps=0.0, lag=0):
    """Дневные доходности: позиция, решённая на close t(+lag), зарабатывает доходность t+1(+lag)."""
    held = pos.shift(1 + lag)
    gross = held * p.pct_change()
    return (gross - held.diff().abs().fillna(0) * cost_bps / 1e4).dropna()


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(ANN) if x.std() > 0 else 0.0


def metrics(x, pos=None):
    eq = (1 + x).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    cagr = eq.iloc[-1] ** (ANN / len(x)) - 1
    act = x[x != 0]
    m = {"sharpe": sharpe(x), "sortino": x.mean() / x[x < 0].std() * np.sqrt(ANN), "cagr": cagr,
         "max_dd": dd, "calmar": cagr / abs(dd), "hit_rate": (act > 0).mean(),
         "profit_factor": act[act > 0].sum() / -act[act < 0].sum()}
    if pos is not None:
        m["turnover_per_year"] = pos.diff().abs().sum() / len(pos) * ANN
        m["time_in_market"] = (pos != 0).mean()
    return m


def nw_tstat(x, L=10):
    """t-статистика среднего с поправкой Newey–West."""
    x = np.asarray(x); n = len(x); e = x - x.mean()
    v = e @ e / n + 2 * sum((1 - j / (L + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, L + 1))
    return x.mean() / np.sqrt(v / n)


def block_boot(x, fn, block=20, reps=2000):
    """Круговой блочный bootstrap: 95% ДИ статистики fn."""
    x = np.asarray(x); n = len(x); nb = int(np.ceil(n / block))
    stats = []
    for _ in range(reps):
        idx = (rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        stats.append(fn(x[idx]))
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def np_sharpe(x):
    return x.mean() / x.std() * np.sqrt(ANN)


def psr(x, sr_star_ann=0.0):
    """Probabilistic Sharpe Ratio (Bailey & Lopez de Prado), SR в пересчёте на день."""
    sr, s0 = x.mean() / x.std(), sr_star_ann / np.sqrt(ANN)
    den = np.sqrt(1 - skew(x) * sr + (kurtosis(x, fisher=False) - 1) / 4 * sr**2)
    return float(norm.cdf((sr - s0) * np.sqrt(len(x) - 1) / den))


def expected_max_sr(sr_trials_ann):
    """Ожидаемый максимум Sharpe среди N бесполезных стратегий (для DSR)."""
    n, v, g = len(sr_trials_ann), np.var(sr_trials_ann, ddof=1), 0.5772156649
    return float(np.sqrt(v) * ((1 - g) * norm.ppf(1 - 1 / n) + g * norm.ppf(1 - 1 / (n * np.e))))


def min_trl_years(x, sr_star_ann=0.0, alpha=0.05):
    sr, s0 = x.mean() / x.std(), sr_star_ann / np.sqrt(ANN)
    if sr <= s0:
        return float("inf")
    num = 1 - skew(x) * sr + (kurtosis(x, fisher=False) - 1) / 4 * sr**2
    return float((1 + num * (norm.ppf(1 - alpha) / (sr - s0)) ** 2) / ANN)


def pbo(mat, S=10):
    """Probability of Backtest Overfitting через CSCV. mat: T x N дневных доходностей."""
    blocks = np.array_split(np.arange(len(mat)), S)
    logits = []
    for comb in combinations(range(S), S // 2):
        is_idx = np.concatenate([blocks[i] for i in comb])
        oos_idx = np.concatenate([blocks[i] for i in range(S) if i not in comb])
        sr_is = mat[is_idx].mean(0) / (mat[is_idx].std(0) + 1e-12)
        sr_oos = mat[oos_idx].mean(0) / (mat[oos_idx].std(0) + 1e-12)
        best = np.argmax(sr_is)
        w = (sr_oos < sr_oos[best]).mean() * len(sr_oos) / (len(sr_oos) + 1) + 0.5 / (len(sr_oos) + 1)
        logits.append(np.log(w / (1 - w)))
    return float((np.array(logits) <= 0).mean())


def ols_hac(y, x, L=10):
    """y = a + b*x, t-статистика b с Newey–West."""
    X = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(X, y, rcond=None)[0]
    u = X * (y - X @ b)[:, None]
    S = u.T @ u + sum((1 - j / (L + 1)) * (u[j:].T @ u[:-j] + u[:-j].T @ u[j:]) for j in range(1, L + 1))
    V = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
    return float(b[1]), float(b[1] / np.sqrt(V[1, 1]))


d = load("btc_yahoo"); p = d["close"]; lp = np.log(p); r = lp.diff()
bh = p.pct_change().dropna()
pos_r, pos_z = regime_pos(p), z_pos(p)
ret_r, ret_z = strat(p, pos_r), strat(p, pos_z)

# ---------- 1. Look-ahead / leakage
bull = pos_r.dropna() == 1
R["1_leakage"] = {
    "regime_mu_same_day_bull_bear": [r[bull.index][bull].mean(), r[bull.index][~bull].mean()],
    "regime_mu_next_day_bull_bear": [r.shift(-1)[bull.index][bull].mean(), r.shift(-1)[bull.index][~bull].mean()],
    "regime_sharpe_fill_same_close": sharpe(ret_r),
    "regime_sharpe_fill_next_close": sharpe(strat(p, pos_r, lag=1)),
    "zmom_sharpe_fill_same_close": sharpe(ret_z),
    "zmom_sharpe_fill_next_close": sharpe(strat(p, pos_z, lag=1)),
}
b = load("btc_binance")["close"]; j = p.index.intersection(b.index)
diff = (p[j] / b[j] - 1).abs()
R["1_leakage"]["yahoo_vs_binance_close_absdiff_bps"] = {"median": diff.median() * 1e4, "p99": diff.quantile(.99) * 1e4, "max": diff.max() * 1e4}

# ---------- 2. Валидация: то, что в работе объявлено, но не сделано
def event_acc(p_, w, lam, h, lo=None, hi=None):
    lp_ = np.log(p_); z = (p_ - p_.rolling(w).mean()) / p_.rolling(w).std(); f = lp_.shift(-h) - lp_
    m = (z.abs() > lam) & f.notna()
    if lo: m &= z.index >= lo
    if hi: m &= z.index + pd.Timedelta(days=h) < hi   # embargo: исход события целиком до границы
    k = int((np.sign(z[m]) == np.sign(f[m])).sum()); n = int(m.sum())
    return k, n, int((f[m] > 0).sum())

k, n, up = event_acc(p, 20, 2.4, 5, hi=pd.Timestamp("2022-01-01"))
k2, n2, up2 = event_acc(p, 20, 2.4, 5, lo=pd.Timestamp("2022-01-01"))
R["2_validation"] = {"paper_split_fixed_params": {
    "train_2017_2021": {"N": n, "acc": k / n, "always_long": up / n},
    "test_2022_2024": {"N": n2, "acc": k2 / n2, "always_long": up2 / n2, "p_vs_50": float(binom.sf(k2 - 1, n2, .5))}}}
GRID = [(w, lam, h) for w in (10, 20, 30, 50) for lam in np.round(np.arange(0.5, 3.01, 0.25), 2) for h in (1, 2, 3, 5, 7, 10)]
wf_k = wf_n = wf_up = 0; picks = {}
for year in range(2020, 2025):
    lo, hi = pd.Timestamp(f"{year}-01-01"), pd.Timestamp(f"{year + 1}-01-01")
    best = max((c for c in GRID if event_acc(p, *c, hi=lo)[1] >= 50), key=lambda c: event_acc(p, *c, hi=lo)[0] / event_acc(p, *c, hi=lo)[1])
    kk, nn, uu = event_acc(p[:hi - pd.Timedelta(days=1)], *best, lo=lo)
    wf_k += kk; wf_n += nn; wf_up += uu
    tk, tn, _ = event_acc(p, *best, hi=lo)
    picks[year] = {"w_lam_h": best, "train_acc": tk / tn, "test_acc": kk / nn if nn else None, "test_N": nn}
R["2_validation"]["walk_forward_param_selection"] = {"picks": picks, "oos_acc": wf_k / wf_n, "oos_N": wf_n,
                                                      "oos_always_long": wf_up / wf_n, "p_vs_50": float(binom.sf(wf_k - 1, wf_n, .5))}

# ---------- 3. Переобучение: DSR, PBO, MinTRL
zmat = pd.DataFrame({c: strat(p, z_pos(p, *c)) for c in GRID}).dropna()
z_sr = zmat.apply(sharpe)
rmat = pd.DataFrame({w: strat(p, regime_pos(p, w)) for w in range(10, 201, 10)}).dropna()
r_sr = rmat.apply(sharpe)
xz, xr = ret_z.values, ret_r.values
R["3_overfitting"] = {
    "zmom": {"n_configs": len(GRID), "sharpe_paper_config": sharpe(ret_z), "sharpe_best_in_grid": z_sr.max(),
             "best_config": [float(v) for v in z_sr.idxmax()], "share_configs_sharpe_gt0": (z_sr > 0).mean(),
             "expected_max_sharpe_if_no_edge": expected_max_sr(z_sr.values),
             "PSR_vs_0": psr(xz), "DSR": psr(xz, expected_max_sr(z_sr.values)),
             "DSR_of_best": psr(zmat[z_sr.idxmax()].values, expected_max_sr(z_sr.values)),
             "PBO": pbo(zmat.values), "MinTRL_years_vs_0": min_trl_years(xz)},
    "regime": {"n_configs": rmat.shape[1], "sharpe_sma50": sharpe(ret_r), "sharpe_by_window": {int(w): v for w, v in r_sr.items()},
               "expected_max_sharpe_if_no_edge": expected_max_sr(r_sr.values),
               "PSR_vs_0": psr(xr), "DSR": psr(xr, expected_max_sr(r_sr.values)),
               "PSR_vs_buyhold_sharpe": psr(xr, sharpe(bh)), "PBO": pbo(rmat.values),
               "MinTRL_years_vs_0": min_trl_years(xr), "MinTRL_years_vs_buyhold": min_trl_years(xr, sharpe(bh))},
}

# ---------- 4. Издержки
costs = (0, 5, 10, 20, 30, 50, 100)
vol_usd = (load("btc_binance")["volume"] * load("btc_binance")["close"]).loc["2023":].median()
sig_d = r.std()
R["4_costs"] = {
    "regime_sharpe_by_cost_bps": {c: sharpe(strat(p, pos_r, c)) for c in costs},
    "regime_cagr_by_cost_bps": {c: metrics(strat(p, pos_r, c))["cagr"] for c in costs},
    "zmom_sharpe_by_cost_bps": {c: sharpe(strat(p, pos_z, c)) for c in costs},
    "regime_trades_per_year": pos_r.diff().abs().sum() / len(pos_r.dropna()) * ANN,
    "zmom_turnover_per_year": pos_z.diff().abs().sum() / len(pos_z.dropna()) * ANN,
    "regime_breakeven_bps_vs_zero": ret_r.sum() / pos_r.diff().abs().sum() * 1e4,
    "regime_breakeven_bps_vs_buyhold_sharpe": next((c for c in range(0, 301) if sharpe(strat(p, pos_r, c)) < sharpe(bh.loc[ret_r.index[0]:])), None),
    "zmom_breakeven_bps_vs_zero": ret_z.sum() / pos_z.diff().abs().sum() * 1e4,
    "binance_btcusdt_median_daily_volume_usd_2023_24": vol_usd,
    "sqrt_law_order_usd_for_5bps_impact": vol_usd * (5e-4 / sig_d) ** 2,
    "sqrt_law_order_usd_for_20bps_impact": vol_usd * (20e-4 / sig_d) ** 2,
}

# ---------- 5. Устойчивость
bh_al = bh.loc[ret_r.index[0]:]
by_year = {}
for y in range(2017, 2025):
    s = str(y); nxt = r.shift(-1)[bull.index].loc[s]; bl = bull.loc[s]
    kk, nn, uu = event_acc(p.loc[:s], 20, 2.4, 5, lo=pd.Timestamp(f"{y}-01-01"))
    by_year[y] = {"buyhold_ret": (1 + bh_al.loc[s]).prod() - 1, "regime_ret": (1 + ret_r.loc[s]).prod() - 1,
                  "buyhold_sharpe": sharpe(bh_al.loc[s]), "regime_sharpe": sharpe(ret_r.loc[s]),
                  "next_day_mu_bull_minus_bear": nxt[bl].mean() - nxt[~bl].mean(),
                  "zmom_ret": (1 + ret_z.loc[s]).prod() - 1, "z_events_N": nn, "z_acc": kk / nn if nn else None}
rv = r.rolling(30).std().shift(1)
terc = pd.qcut(rv.loc[ret_r.index], 3, labels=["low_vol", "mid_vol", "high_vol"])
assets = {}
for name in ("btc_yahoo", "btc_binance", "eth_binance", "sol_binance"):
    q = load(name)["close"]; rr_, zz_ = strat(q, regime_pos(q)), strat(q, z_pos(q)); bq = q.pct_change().loc[rr_.index[0]:]
    kk, nn, uu = event_acc(q, 20, 2.4, 5)
    assets[name] = {"from": str(q.index[0].date()), "buyhold_sharpe": sharpe(bq), "regime_sharpe": sharpe(rr_),
                    "buyhold_maxdd": metrics(bq)["max_dd"], "regime_maxdd": metrics(rr_)["max_dd"],
                    "zmom_sharpe": sharpe(zz_), "z_N": nn, "z_acc": kk / nn, "z_always_long": uu / nn}
R["5_robustness"] = {
    "by_year": by_year,
    "by_vol_tercile": {k_: {"regime_sharpe": sharpe(ret_r[terc == k_]), "buyhold_sharpe": sharpe(bh_al[terc == k_])} for k_ in terc.cat.categories},
    "by_asset": assets,
    "regime_sma_pm20pct_sharpe": {w: sharpe(strat(p, regime_pos(p, w))) for w in (40, 45, 50, 55, 60)},
    "z_acc_pm20pct": {f"w{w}_lam{lam}_h{h}": round(event_acc(p, w, lam, h)[0] / event_acc(p, w, lam, h)[1], 4)
                      for w in (16, 20, 24) for lam in (1.9, 2.4, 2.9) for h in (4, 5, 6)},
}

# ---------- 6. Статистика
z = (p - p.rolling(20).mean()) / p.rolling(20).std(); f5 = lp.shift(-5) - lp; base = (f5.dropna() > 0).mean()
def up_after(mask):
    m = mask & f5.notna(); idx, last = [], None
    for t in m[m].index:                       # неперекрывающиеся события
        if last is None or (t - last).days >= 5: idx.append(t); last = t
    u, n_ = int((f5[idx] > 0).sum()), len(idx)
    return {"N_overlapping": int(m.sum()), "up_rate_overlapping": float((f5[m] > 0).mean()), "N_independent": n_,
            "up_rate_independent": u / n_, "p_two_sided_vs_base_rate": float(min(1, 2 * min(binom.cdf(u, n_, base), binom.sf(u - 1, n_, base))))}
nxt = r.shift(-1)[bull.index].dropna(); bl = bull[nxt.index].astype(float)
slope, t_slope = ols_hac(nxt.values, bl.values)
pv = sorted([("lam0.5", 0.0205), ("lam1.9", 0.0242), ("lam2.4", 0.0591)], key=lambda t: t[1])
ex = (ret_r - bh_al).dropna()
R["6_statistics"] = {
    "returns_excess_kurtosis": float(kurtosis(r.dropna())), "returns_skew": float(skew(r.dropna())),
    "jarque_bera_p": float(jarque_bera(r.dropna()).pvalue),
    "worst_day": float(r.min()), "worst_day_in_sigmas": float((r.min() - r.mean()) / r.std()),
    "base_rate_up_5d": float(base),
    "up_rate_after_Z_gt_2.4": up_after(z > 2.4), "up_rate_after_Z_lt_-2.4": up_after(z < -2.4),
    "regime_next_day_bull_minus_bear": {"diff": slope, "t_newey_west": t_slope, "p_two_sided": float(2 * norm.sf(abs(t_slope)))},
    "holm_paper_lambda_tests": [{"test": n_, "p": p_, "holm_threshold": 0.05 / (3 - i),   # step-down: после первого провала всё незначимо
                                 "significant": all(q < 0.05 / (3 - j) for j, (_, q) in enumerate(pv[:i + 1]))} for i, (n_, p_) in enumerate(pv)],
    "regime": {**metrics(ret_r, pos_r.dropna()), "sharpe_ci95_block_boot": block_boot(ret_r.values, np_sharpe), "t_newey_west_mean": float(nw_tstat(ret_r.values))},
    "zmom": {**metrics(ret_z, pos_z.dropna()), "sharpe_ci95_block_boot": block_boot(ret_z.values, np_sharpe), "t_newey_west_mean": float(nw_tstat(ret_z.values))},
    "buyhold": {**metrics(bh_al), "sharpe_ci95_block_boot": block_boot(bh_al.values, np_sharpe)},
    "regime_minus_buyhold": {"mean_daily": float(ex.mean()), "t_newey_west": float(nw_tstat(ex.values)),
                             "sharpe_diff_ci95_block_boot": block_boot(np.column_stack([ret_r.values, bh_al.values]),
                                                                       lambda a: np_sharpe(a[:, 0]) - np_sharpe(a[:, 1]))},
}

def clean(o):
    if isinstance(o, dict): return {str(k_): clean(v) for k_, v in o.items()}
    if isinstance(o, (list, tuple)): return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)): return round(float(o), 5)
    if isinstance(o, np.integer): return int(o)
    return o

json.dump(clean(R), open("results/audit.json", "w"), indent=1, ensure_ascii=False)
print(json.dumps(clean(R), indent=1, ensure_ascii=False))
