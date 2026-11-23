"""Скачивает дневные свечи BTC: Yahoo BTC-USD (как в работе) и Binance BTCUSDT (сверка)."""
import json, urllib.request, time
import pandas as pd
import yfinance as yf

y = yf.download("BTC-USD", start="2016-10-01", end="2026-11-23", interval="1d", auto_adjust=False, progress=False)
y.columns = [c[0].lower() for c in y.columns]
y.index.name = "date"
y[["open", "high", "low", "close", "volume"]].to_csv("data/btc_yahoo_1d.csv")
print("yahoo", y.index[0].date(), y.index[-1].date(), len(y))

rows, start = [], int(pd.Timestamp("2017-08-01", tz="UTC").timestamp() * 1000)
while True:
    url = f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000&startTime={start}"
    part = json.load(urllib.request.urlopen(url, timeout=30))
    if not part:
        break
    rows += part
    start = part[-1][0] + 86_400_000
    time.sleep(0.2)
b = pd.DataFrame(rows).iloc[:, :6]
b.columns = ["date", "open", "high", "low", "close", "volume"]
b["date"] = pd.to_datetime(b["date"], unit="ms")
b.set_index("date").astype(float).to_csv("data/btc_binance_1d.csv")
print("binance", b["date"].iloc[0].date(), b["date"].iloc[-1].date(), len(b))
