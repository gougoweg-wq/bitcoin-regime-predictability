"""Loading price data from disk or from the public Binance market-data endpoint."""
import json
import time
import urllib.request

import pandas as pd

BINANCE = "https://data-api.binance.vision/api/v3/klines"


def load_close(path, start=None, end=None):
    """Daily closing prices from a CSV with a date index and a `close` column."""
    close = pd.read_csv(path, index_col=0, parse_dates=True)["close"]
    return close.loc[start:end]


def fetch_binance_daily(symbol="BTCUSDT", start="2017-08-01"):
    """Completed daily candles from Binance spot. No API key is required."""
    rows, cursor = [], int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    while True:
        url = f"{BINANCE}?symbol={symbol}&interval=1d&limit=1000&startTime={cursor}"
        part = json.load(urllib.request.urlopen(url, timeout=30))
        if not part:
            break
        rows += part
        cursor = part[-1][0] + 86_400_000
        time.sleep(0.1)
    frame = pd.DataFrame(rows).iloc[:, :6]
    frame.columns = ["date", "open", "high", "low", "close", "volume"]
    frame["date"] = pd.to_datetime(frame["date"], unit="ms")
    frame = frame.set_index("date").astype(float)
    today = pd.Timestamp.now("UTC").tz_localize(None).floor("D")
    return frame[frame.index < today]            # drop the candle that is still forming
