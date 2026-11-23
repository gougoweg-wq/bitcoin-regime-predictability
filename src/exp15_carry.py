"""Усиление 15: нейтральное к цене керри на фандинге BTC (как зарабатывают без прогноза направления).
Позиция: +1 BTC на споте, -1 BTC в бессрочном фьючерсе. Цена сокращается, остаётся фандинг: раз в 8 часов
шорт получает ставку f_t, если она положительна, и платит, если отрицательна.
Данные: Binance BTCUSDT funding 2020-01..2024-12 (2025+ не читается). Допущения:
  - капитал = 1.5 x номинал (1.0 спот + 0.5 маржа фьючерса), доходность считается на весь капитал;
  - вход или выход стоит 15 б.п. от номинала (10 спот + 5 фьючерс), полный круг 30 б.п.;
  - колебания базиса (разницы цен фьючерса и спота) не моделируются: в среднем они нулевые, но добавляют шум.
Варианты: A) держать всегда; B) держать, пока средний фандинг за 3 дня > 0 (иначе вне позиции);
          C) как B, но только при бычьем режиме (P > SMA50) — связка с ядром работы."""
import json
import numpy as np
import pandas as pd

END, CAP, COST, PER_YEAR = "2024-12-31 23:59", 1.5, 15 / 1e4, 3 * 365
f = pd.read_csv("data/btc_funding.csv", index_col=0, parse_dates=True)["last_funding_rate"][:END]
f.index = f.index.floor("s")
p = pd.read_csv("data/btc_binance_1d.csv", index_col=0, parse_dates=True)["close"][:END]
bull = (p > p.rolling(50).mean()).astype(float)
bull.index = bull.index + pd.Timedelta(days=1)                       # режим по закрытию дня D известен с 00:00 дня D+1
bull8 = bull.reindex(f.index, method="ffill")

trail = f.rolling(9).mean().shift(1)                                 # средний фандинг за 3 дня, известен до расчёта f_t
pos = {"A_always": pd.Series(1.0, index=f.index), "B_trailing_positive": (trail > 0).astype(float),
       "C_trailing_positive_and_bull": ((trail > 0) & (bull8 == 1)).astype(float)}


def stats(w):
    r = (w * f - w.diff().abs().fillna(w.iloc[0]) * COST) / CAP      # доходность на капитал за 8 часов
    eq = (1 + r).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    rng = np.random.default_rng(42); x = r.values; n = len(x); blk = 90; nb = int(np.ceil(n / blk))
    bs = [(lambda s: s.mean() / s.std() * np.sqrt(PER_YEAR))(x[(rng.integers(0, n, nb)[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n]) for _ in range(2000)]
    return {"annual_return": round(float(eq.iloc[-1] ** (PER_YEAR / n) - 1), 4), "annual_vol": round(float(r.std() * np.sqrt(PER_YEAR)), 4),
            "sharpe_funding_only": round(float(r.mean() / r.std() * np.sqrt(PER_YEAR)), 2),
            "sharpe_ci95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
            "max_dd": round(dd, 4), "time_in_position": round(float(w.mean()), 3), "switches_per_year": round(float(w.diff().abs().sum() / n * PER_YEAR), 1),
            "by_year_return": {int(y): round(float((1 + r[r.index.year == y]).prod() - 1), 4) for y in sorted(set(r.index.year))}}


out = {"funding": {"prints": len(f), "from": str(f.index[0]), "mean_per_8h": float(f.mean()), "annualized_mean": float(f.mean() * PER_YEAR),
                   "share_positive": float((f > 0).mean()), "autocorr_lag1": float(f.autocorr(1)), "autocorr_lag9": float(f.autocorr(9)),
                   "by_regime_annualized": {"bull": float(f[bull8 == 1].mean() * PER_YEAR), "bear": float(f[bull8 == 0].mean() * PER_YEAR)}}}
m = trail.notna()
pred = trail[m] > 0; act = f[m] > 0
out["forecast_sign_of_next_funding"] = {"rule": "знак среднего за 3 дня", "accuracy": round(float((pred == act).mean()), 3),
                                        "always_positive_accuracy": round(float(act.mean()), 3),
                                        "accuracy_by_year": {int(y): round(float((pred == act)[act.index.year == y].mean()), 3) for y in sorted(set(act.index.year))}}
med = f.rolling(1095, min_periods=270).median().shift(1); mm = med.notna() & trail.notna()
pred2 = trail[mm] > med[mm]; act2 = f[mm] > med[mm]
out["forecast_funding_above_1y_median"] = {"accuracy": round(float((pred2 == act2).mean()), 3), "share_above_actual": round(float(act2.mean()), 3),
                                           "accuracy_by_year": {int(y): round(float((pred2 == act2)[act2.index.year == y].mean()), 3) for y in sorted(set(act2.index.year))}}
out["strategies"] = {k: stats(w) for k, w in pos.items()}
json.dump(out, open("results/phase4/exp15_carry.json", "w"), indent=1)
for k, v in out.items():
    if k == "strategies":
        for s, vv in v.items(): print(s, vv)
    else: print(k, v)
