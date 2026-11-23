"""Усиление 15b: керри BTC с реальными ценами фьючерса. Позиция +1 спот, -1 бессрочный фьючерс, держим всегда.
Дневная доходность на номинал = доходность спота - доходность фьючерса + фандинг; на капитал делим на 1.5.
Маржа шорта (0.5 номинала) выравнивается раз в сутки; проверяем, хватало ли её внутри дня (по дневным максимумам спота)."""
import json
import numpy as np
import pandas as pd

END, CAP, ANN = "2024-12-31", 1.5, 365
spot = pd.read_csv("data/btc_binance_1d.csv", index_col=0, parse_dates=True)[:END]
K = pd.read_parquet("data/perp_1d.parquet"); F = pd.read_parquet("data/perp_funding.parquet")
perp = K[K.sym == "BTC"].set_index("date")["close"][:END]; fund = F[F.sym == "BTC"].set_index("date")["funding"][:END]
d = pd.DataFrame({"spot": spot["close"], "high": spot["high"], "perp": perp, "fund": fund}).dropna()
basis = d.perp / d.spot - 1
r = (d.spot.pct_change() - d.perp.pct_change() + d.fund).dropna() / CAP
eq = (1 + r).cumprod(); rng = np.random.default_rng(42); x = r.values; n = len(x); nb = int(np.ceil(n / 20))
bs = [(lambda s: s.mean() / s.std() * np.sqrt(ANN))(x[(rng.integers(0, n, nb)[:, None] + np.arange(20)[None, :]).ravel()[:n] % n]) for _ in range(2000)]
intraday_up = (d.high / d.spot.shift(1) - 1).dropna()
out = {"from": str(d.index[0].date()), "days": n, "annual_return": round(float(eq.iloc[-1] ** (ANN / n) - 1), 4), "annual_vol": round(float(r.std() * np.sqrt(ANN)), 4),
       "sharpe": round(float(r.mean() / r.std() * np.sqrt(ANN)), 2), "sharpe_ci95": [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)],
       "max_dd": round(float((eq / eq.cummax() - 1).min()), 4), "worst_day": round(float(r.min()), 4), "worst_day_date": str(r.idxmin().date()),
       "by_year_return": {int(y): round(float((1 + r[r.index.year == y]).prod() - 1), 4) for y in sorted(set(r.index.year))},
       "basis_bps": {"mean": round(float(basis.mean() * 1e4), 1), "std": round(float(basis.std() * 1e4), 1), "min": round(float(basis.min() * 1e4), 1), "max": round(float(basis.max() * 1e4), 1)},
       "max_intraday_rise_vs_prev_close": round(float(intraday_up.max()), 3), "max_intraday_rise_date": str(intraday_up.idxmax().date()),
       "margin_buffer_before_liquidation": "около +49% за сутки при марже 0.5 номинала"}
json.dump(out, open("results/phase4/exp15b_carry_basis.json", "w"), indent=1)
for k, v in out.items(): print(k, v)
