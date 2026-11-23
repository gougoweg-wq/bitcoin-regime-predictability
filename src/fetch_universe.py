"""Дневные свечи всех спотовых USDT-пар Binance из публичного архива data.binance.vision,
включая делистнутые (чтобы не было survivorship bias). Итог: data/universe_1d.parquet."""
import io, os, re, zipfile, urllib.request
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

S3 = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
RAW = "data/raw_universe"; os.makedirs(RAW, exist_ok=True)
STABLE = {"USDC", "BUSD", "TUSD", "USDP", "PAX", "DAI", "FDUSD", "USDS", "USDSB", "SUSD", "UST", "USTC", "EUR", "GBP", "AUD", "EURI", "AEUR",
          "BKRW", "IDRT", "BIDR", "TRY", "BRL", "RUB", "UAH", "NGN", "ZAR", "PAXG", "WBTC", "WBETH", "BETH", "USD1", "XUSD", "BFUSD", "VAI"}


def listing(prefix, key="Prefix"):
    out, marker = [], ""
    while True:
        x = urllib.request.urlopen(f"{S3}?delimiter=/&prefix={prefix}&marker={marker}", timeout=60).read().decode()
        out += re.findall(r"<Prefix>([^<]+)</Prefix>" if key == "Prefix" else r"<Key>([^<]+\.zip)</Key>", x)
        if "<IsTruncated>true</IsTruncated>" not in x:
            return out
        marker = re.findall(r"<NextMarker>([^<]+)</NextMarker>", x)[0]


def symbol_ok(s):
    if not s.endswith("USDT"):
        return False
    b = s[:-4]
    return b not in STABLE and not re.search(r"(UP|DOWN|BULL|BEAR)$", b)


def fetch(sym):
    fn = f"{RAW}/{sym}.csv"
    if os.path.exists(fn):
        return
    rows = []
    for key in listing(f"data/spot/monthly/klines/{sym}/1d/", key="Key"):
        try:
            z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(f"https://data.binance.vision/{key}", timeout=60).read()))
            d = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])), header=None, usecols=[0, 1, 2, 3, 4, 7])
            rows.append(d[pd.to_numeric(d[0], errors="coerce").notna()])
        except Exception as e:
            print("skip", key, e)
    if rows:
        pd.concat(rows).to_csv(fn, index=False, header=False)


if __name__ == "__main__":
    syms = [p.rstrip("/").rsplit("/", 1)[1] for p in listing("data/spot/monthly/klines/")]
    syms = [s for s in syms if symbol_ok(s)]
    print("символов:", len(syms))
    with ThreadPoolExecutor(24) as ex:
        list(ex.map(fetch, syms))
    frames = []
    for s in syms:
        fn = f"{RAW}/{s}.csv"
        if not os.path.exists(fn):
            continue
        d = pd.read_csv(fn, header=None, names=["t", "open", "high", "low", "close", "qvol"]).astype(float)
        t = d["t"].where(d["t"] < 1e14, d["t"] / 1000)             # с 2025 года архив в микросекундах
        d["date"] = pd.to_datetime(t, unit="ms").dt.floor("D"); d["sym"] = s[:-4]
        frames.append(d.drop(columns="t").drop_duplicates("date"))
    U = pd.concat(frames).sort_values(["sym", "date"])
    U.to_parquet("data/universe_1d.parquet")
    print("строк:", len(U), "| монет:", U.sym.nunique(), "|", U.date.min().date(), "..", U.date.max().date())
