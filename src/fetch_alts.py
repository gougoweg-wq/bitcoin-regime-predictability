"""Дневные свечи ETHUSDT и SOLUSDT с Binance (для проверки устойчивости по активам)."""
import json, time, urllib.request
import pandas as pd

for sym in ("ETHUSDT", "SOLUSDT"):
    rows, start = [], int(pd.Timestamp("2017-08-01", tz="UTC").timestamp() * 1000)
    while True:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={sym}&interval=1d&limit=1000&startTime={start}"
        part = json.load(urllib.request.urlopen(url, timeout=30))
        if not part:
            break
        rows += part
        start = part[-1][0] + 86_400_000
        time.sleep(0.2)
    b = pd.DataFrame(rows).iloc[:, :6]
    b.columns = ["date", "open", "high", "low", "close", "volume"]
    b["date"] = pd.to_datetime(b["date"], unit="ms")
    b.set_index("date").astype(float).to_csv(f"data/{sym[:3].lower()}_binance_1d.csv")
    print(sym, b["date"].iloc[0].date(), b["date"].iloc[-1].date(), len(b))
