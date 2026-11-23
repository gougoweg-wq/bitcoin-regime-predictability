"""Часовые свечи BTCUSDT с Binance (спот). Итог: data/btc_binance_1h.csv."""
import json, time, urllib.request
import pandas as pd

rows, start = [], int(pd.Timestamp("2017-08-01", tz="UTC").timestamp() * 1000)
while True:
    url = f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000&startTime={start}"
    part = json.load(urllib.request.urlopen(url, timeout=30))
    if not part:
        break
    rows += part
    start = part[-1][0] + 3_600_000
    time.sleep(0.1)
b = pd.DataFrame(rows).iloc[:, [0, 1, 2, 3, 4, 5, 7, 9]]
b.columns = ["time", "open", "high", "low", "close", "volume", "qvol", "taker_buy_vol"]
b["time"] = pd.to_datetime(b["time"], unit="ms")
b = b.set_index("time").astype(float)
b = b[b.index < pd.Timestamp.now("UTC").tz_localize(None).floor("h")]      # без незакрытой свечи
b.to_csv("data/btc_binance_1h.csv")
print(b.index[0], b.index[-1], len(b))
