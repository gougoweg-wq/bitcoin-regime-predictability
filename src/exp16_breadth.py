"""Усиление 16: ширина. Бессрочные фьючерсы Binance, на каждый день 30 самых ликвидных (по медиане оборота за 30 дней,
история не меньше 60 дней; снятые с торгов монеты включены). Период разработки 2020-07..2024-12, 2025+ не читается.
Доходность лонга во фьючерсе за день = изменение цены - фандинг; шорта = наоборот. Издержки 10 б.п. с оборота.
Сигналы считаются на закрытии дня t, позиции пересобираются раз в 7 дней и держатся со дня t+1.
  CARRY : шорт 20% монет с самым высоким фандингом за 7 дней, лонг 20% с самым низким (нейтрально к рынку).
  MOM   : лонг 20% с лучшей доходностью за 30 дней, шорт 20% с худшей (нейтрально к рынку).
  REGIME: каждая монета в лонге, если P > SMA50, иначе вне рынка; веса 1/30 (ядро работы, но в ширину).
Гипотеза (записана до расчёта): премия CARRY выше в бычьем режиме BTC, когда спрос на плечо и разброс фандинга больше."""
import json
import numpy as np
import pandas as pd

START, END, ANN, COST, TOP, Q, REB = "2020-07-01", "2024-12-31", 365, 10 / 1e4, 30, 0.2, 7
K = pd.read_parquet("data/perp_1d.parquet"); F = pd.read_parquet("data/perp_funding.parquet")
K = K[K.date <= END]; F = F[F.date <= END]
P = K.pivot(index="date", columns="sym", values="close").sort_index(); V = K.pivot(index="date", columns="sym", values="qvol").reindex(P.index)
Fd = F.pivot(index="date", columns="sym", values="funding").reindex(index=P.index, columns=P.columns)
ret = P.pct_change(fill_method=None); tr = ret - Fd.fillna(0)                 # полная доходность лонга
ret = ret.clip(-0.9, 5); tr = tr.clip(-0.9, 5)

hist = P.notna().cumsum()
liq = V.rolling(30, min_periods=20).median().where(hist >= 60)
uni = liq.rank(axis=1, ascending=False) <= TOP                                # вселенная на закрытии дня t

carry_sig = Fd.rolling(7, min_periods=5).mean().where(uni)
mom_sig = (P / P.shift(30) - 1).where(uni)
reg_sig = (P > P.rolling(50).mean()).astype(float).where(uni & (hist >= 60))


def ls_weights(sig, long_low):
    r = sig.rank(axis=1, pct=True); n = sig.notna().sum(axis=1)
    lo, hi = (r <= Q).astype(float), (r > 1 - Q).astype(float)
    lo = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0) * 0.5; hi = hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    w = (lo - hi) if long_low else (hi - lo)
    return w.where(n >= 15, 0).fillna(0)


def hold(w):
    """Пересборка раз в REB дней: веса фиксируются в дни пересборки и держатся до следующей."""
    mask = np.zeros(len(w), dtype=bool); mask[::REB] = True
    return w.where(pd.Series(mask, index=w.index), np.nan).ffill().fillna(0)


def pnl(w):
    held = hold(w).shift(1).fillna(0)
    gross = (held * tr.fillna(0)).sum(axis=1); fund = (-held * Fd.fillna(0)).sum(axis=1)
    cost = held.diff().abs().sum(axis=1) * COST
    return (gross - cost)[START:], fund[START:], held[START:]


def stats(x, fund=None, held=None):
    eq = (1 + x).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    rng = np.random.default_rng(42); a = x.values; n = len(a); blk = 20; nb = int(np.ceil(n / blk))
    bs = [(lambda s: s.mean() / s.std() * np.sqrt(ANN))(a[(rng.integers(0, n, nb)[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n]) for _ in range(2000)]
    o = {"sharpe": round(float(x.mean() / x.std() * np.sqrt(ANN)), 2), "sharpe_ci95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
         "annual_return": round(float(eq.iloc[-1] ** (ANN / n) - 1), 3), "annual_vol": round(float(x.std() * np.sqrt(ANN)), 3), "max_dd": round(dd, 3),
         "by_year_return": {int(y): round(float((1 + x[x.index.year == y]).prod() - 1), 3) for y in sorted(set(x.index.year))}}
    if fund is not None:
        o["funding_part_annual"] = round(float(fund.mean() * ANN), 3); o["turnover_per_year"] = round(float(held.diff().abs().sum(axis=1).mean() * ANN), 1)
    return o


def nw_t(x, L=10):
    x = np.asarray(x); n = len(x); e = x - x.mean()
    v = e @ e / n + 2 * sum((1 - j / (L + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, L + 1))
    return float(x.mean() / np.sqrt(v / n))


if __name__ == "__main__":
    W = {"CARRY": ls_weights(carry_sig, long_low=True), "MOM": ls_weights(mom_sig, long_low=False), "REGIME": (reg_sig / TOP).fillna(0),
         "EW_BUYHOLD": (uni.astype(float) / TOP)}
    R, out = {}, {"universe": {"symbols_total": int(P.shape[1]), "ever_in_top30": int(uni[START:].any().sum()),
                               "avg_names_per_day": round(float(uni[START:].sum(axis=1).mean()), 1)}, "strategies": {}}
    for k, w in W.items():
        x, fund, held = pnl(w); R[k] = x; out["strategies"][k] = stats(x, fund, held)
    R["BTC_BUYHOLD"] = tr["BTC"][START:]; out["strategies"]["BTC_BUYHOLD"] = stats(R["BTC_BUYHOLD"])
    combo = (R["CARRY"] / R["CARRY"].std() + R["MOM"] / R["MOM"].std()) / 2 * 0.10 / np.sqrt(ANN)       # равный риск, масштаб ~10% годовой волатильности
    R["COMBO_CARRY_MOM"] = combo; out["strategies"]["COMBO_CARRY_MOM"] = stats(combo)
    out["correlations"] = pd.DataFrame(R).corr().round(2).to_dict()

    btc = P["BTC"]; bull = (btc > btc.rolling(50).mean()).shift(1).reindex(R["CARRY"].index)                 # режим BTC, известный накануне
    cond = {}
    for k in ("CARRY", "MOM"):
        x = R[k]; b, e = x[bull == True], x[bull == False]
        y = x.values; g = (bull == True).astype(float).values; X = np.column_stack([np.ones(len(g)), g]); beta = np.linalg.lstsq(X, y, rcond=None)[0]; u = X * (y - X @ beta)[:, None]
        S = u.T @ u + sum((1 - j / 11) * (u[j:].T @ u[:-j] + u[:-j].T @ u[j:]) for j in range(1, 11)); Vb = np.linalg.inv(X.T @ X) @ S @ np.linalg.inv(X.T @ X)
        cond[k] = {"bull": {"days": len(b), "annual_return": round(float(b.mean() * ANN), 3), "sharpe": round(float(b.mean() / b.std() * np.sqrt(ANN)), 2), "t_nw": round(nw_t(b.values), 2)},
                   "bear": {"days": len(e), "annual_return": round(float(e.mean() * ANN), 3), "sharpe": round(float(e.mean() / e.std() * np.sqrt(ANN)), 2), "t_nw": round(nw_t(e.values), 2)},
                   "diff_bull_minus_bear_annual": round(float(beta[1] * ANN), 3), "diff_t_nw": round(float(beta[1] / np.sqrt(Vb[1, 1])), 2),
                   "by_year_[bull, bear]_annual": {int(yr): [round(float(b[b.index.year == yr].mean() * ANN), 2) if (b.index.year == yr).sum() > 20 else None,
                                                             round(float(e[e.index.year == yr].mean() * ANN), 2) if (e.index.year == yr).sum() > 20 else None] for yr in sorted(set(x.index.year))}}
    out["regime_conditional"] = cond
    disp = carry_sig.std(axis=1)[START:] * ANN
    out["funding_dispersion_annualized"] = {"bull": round(float(disp[bull == True].mean()), 3), "bear": round(float(disp[bull == False].mean()), 3)}
    json.dump(out, open("results/phase4/exp16_breadth.json", "w"), indent=1)
    pd.DataFrame(R).to_csv("results/phase4/exp16_daily_returns.csv")
    print("вселенная:", out["universe"])
    for k, v in out["strategies"].items(): print(f"{k:16s}", v)
    print("корреляции:", {k: out["correlations"][k] for k in ("CARRY", "MOM")})
    for k, v in cond.items(): print("режим |", k, v)
    print("разброс фандинга:", out["funding_dispersion_annualized"])
