"""Проверка: насколько предсказуем не знак, а РАЗМАХ движения BTC.
Прогноз на закрытии дня t: «в следующие h дней волатильность будет выше своей медианы за прошлый год», если
текущая EWMA-волатильность (halflife 10 дней) выше своей медианы за прошлый год. Параметров для подгонки нет.
Оценка 2018-01-01..2024-12-31, Yahoo BTC-USD. Блочный bootstrap для ДИ точности."""
import json
import numpy as np
import pandas as pd

p = pd.read_csv("data/btc_yahoo_1d_long.csv", index_col=0, parse_dates=True).loc["2014-09-17":"2024-12-31", "close"]
r = np.log(p).diff()
ew = np.sqrt((r**2).ewm(halflife=10, min_periods=30).mean())
sig = ew > ew.rolling(365, min_periods=180).median()                         # прогноз: «будет бурно»
rng = np.random.default_rng(42); out = {}
for h in (1, 5, 10, 20):
    fut = r.abs().shift(-1) if h == 1 else r.rolling(h).std().shift(-h)      # будущий размах
    past = r.abs() if h == 1 else r.rolling(h).std()
    y = fut > past.rolling(365, min_periods=180).median()                     # порог известен на день t
    d = pd.DataFrame({"sig": sig, "y": y, "ok": fut.notna() & ew.notna()}).loc["2018-01-01":]
    d = d[d.ok]; hit = (d.sig == d.y).astype(float).values; n = len(hit); blk = max(2 * h, 10); nb = int(np.ceil(n / blk))
    bs = [hit[(rng.integers(0, n, nb)[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n].mean() for _ in range(2000)]
    out[f"h{h}"] = {"days": n, "accuracy": round(float(hit.mean()), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
                    "share_high_actual": round(float(d.y.mean()), 3),
                    "by_year": {int(yr): round(float(hit[d.index.year == yr].mean()), 2) for yr in sorted(set(d.index.year))}}
    print(f"h={h:2d}", out[f"h{h}"])
json.dump(out, open("results/phase4/exp12_vol_predict.json", "w"), indent=1)
