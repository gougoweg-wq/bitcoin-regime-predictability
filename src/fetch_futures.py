"""Фандинг (помесячно) и open interest (ежедневные metrics) Binance USDT-M с data.binance.vision.
Сырые zip лежат в data/raw_futures/, сводные CSV — в data/."""
import io, os, sys, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"
RAW = "data/raw_futures"; os.makedirs(RAW, exist_ok=True)
SYMS = {"btc": "BTCUSDT", "eth": "ETHUSDT", "sol": "SOLUSDT"}
END = "2026-10-31"


def get(url):
    fn = os.path.join(RAW, url.rsplit("/", 1)[1])
    if not os.path.exists(fn):
        try:
            data = urllib.request.urlopen(url, timeout=30).read()
        except Exception:
            return None
        open(fn, "wb").write(data)
    try:
        z = zipfile.ZipFile(fn)
        return pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
    except Exception:
        return None


for a, s in SYMS.items():
    months = pd.period_range("2020-01", END[:7], freq="M")
    with ThreadPoolExecutor(8) as ex:
        parts = list(ex.map(lambda m: get(f"{BASE}/monthly/fundingRate/{s}/{s}-fundingRate-{m}.zip"), months))
    f = pd.concat([x for x in parts if x is not None])
    f["time"] = pd.to_datetime(f["calc_time"], unit="ms")
    f = f.set_index("time").sort_index()[["last_funding_rate"]]
    f.to_csv(f"data/{a}_funding.csv"); print(a, "funding", f.index[0], f.index[-1], len(f))

    days = pd.date_range("2020-09-01", END, freq="D")
    def oi(d):
        x = get(f"{BASE}/daily/metrics/{s}/{s}-metrics-{d.date()}.zip")
        if x is None or not len(x):
            return None
        last = x.iloc[-1]   # последний 5-минутный снимок дня (23:55 UTC) — известен на закрытии дневной свечи
        return {"date": d, "oi": last["sum_open_interest"], "oi_usd": last["sum_open_interest_value"],
                "ls_ratio": last.get("count_long_short_ratio"), "top_ls_ratio": last.get("sum_toptrader_long_short_ratio")}
    with ThreadPoolExecutor(16) as ex:
        rows = [x for x in ex.map(oi, days) if x]
    o = pd.DataFrame(rows).set_index("date"); o.to_csv(f"data/{a}_oi.csv"); print(a, "oi", o.index[0].date(), o.index[-1].date(), len(o))
