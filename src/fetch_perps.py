"""Все бессрочные USDT-фьючерсы Binance (включая снятые с торгов): дневные свечи и фандинг из data.binance.vision.
Итог: data/perp_1d.parquet (date, sym, close, qvol) и data/perp_funding.parquet (date, sym, funding — сумма за сутки)."""
import io, os, re, time, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
RAW = "data/raw_perp"; os.makedirs(RAW, exist_ok=True)
SKIP = {"USDC", "BUSD", "TUSD", "FDUSD", "USDP", "DAI", "EUR", "BTCDOM", "DEFI", "BLUEBIRD", "FOOTBALL"}


def fetch(url, tries=4):
    for i in range(tries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


def listing(prefix, key):
    out, marker = [], ""
    while True:
        x = fetch(f"{S3}?delimiter=/&prefix={prefix}&marker={marker}").decode()
        out += re.findall(r"<Prefix>([^<]+)</Prefix>" if key == "Prefix" else r"<Key>([^<]+\.zip)</Key>", x)
        if "<IsTruncated>true</IsTruncated>" not in x:
            return out
        marker = re.findall(r"<NextMarker>([^<]+)</NextMarker>", x)[0]


def grab(sym, kind):
    fn = f"{RAW}/{sym}_{kind}.csv"
    if os.path.exists(fn):
        return
    prefix = f"data/futures/um/monthly/klines/{sym}/1d/" if kind == "k" else f"data/futures/um/monthly/fundingRate/{sym}/"
    rows = []
    for key in listing(prefix, "Key"):
        try:
            z = zipfile.ZipFile(io.BytesIO(fetch(f"https://data.binance.vision/{key}")))
            d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None)
            d = d[pd.to_numeric(d[0], errors="coerce").notna()]
            rows.append(d[[0, 4, 7]] if kind == "k" else d[[0, 2]])
        except Exception as e:
            print("fail", key, type(e).__name__, flush=True); return
    if rows:
        pd.concat(rows).to_csv(fn, index=False, header=False)


if __name__ == "__main__":
    syms = [p.rstrip("/").rsplit("/", 1)[1] for p in listing("data/futures/um/monthly/klines/", "Prefix")]
    syms = [s for s in syms if s.endswith("USDT") and s[:-4] not in SKIP and "_" not in s]
    print("символов:", len(syms), flush=True)
    def safe(a):
        try:
            grab(*a)
        except Exception as e:
            print("fail", a, type(e).__name__, flush=True)
    with ThreadPoolExecutor(16) as ex:
        list(ex.map(safe, [(s, k) for s in syms for k in ("k", "f")]))
    K, F = [], []
    for s in syms:
        if os.path.exists(f"{RAW}/{s}_k.csv"):
            d = pd.read_csv(f"{RAW}/{s}_k.csv", header=None, names=["t", "close", "qvol"]).astype(float)
            d["date"] = pd.to_datetime(d["t"], unit="ms").dt.floor("D"); d["sym"] = s[:-4]; K.append(d.drop(columns="t").drop_duplicates("date"))
        if os.path.exists(f"{RAW}/{s}_f.csv"):
            d = pd.read_csv(f"{RAW}/{s}_f.csv", header=None, names=["t", "funding"]).astype(float)
            d["date"] = (pd.to_datetime(d["t"], unit="ms") - pd.Timedelta(seconds=1)).dt.floor("D")   # расчёт в 00:00 относится к прошедшим суткам
            F.append(d.groupby("date", as_index=False)["funding"].sum().assign(sym=s[:-4]))
    K = pd.concat(K).sort_values(["sym", "date"]); F = pd.concat(F).sort_values(["sym", "date"])
    K.to_parquet("data/perp_1d.parquet"); F.to_parquet("data/perp_funding.parquet")
    print("свечи:", len(K), "строк,", K.sym.nunique(), "монет,", K.date.min().date(), "..", K.date.max().date())
    print("фандинг:", len(F), "строк,", F.sym.nunique(), "монет")
