"""Усиление 14: нейросети в 2 и 3 скрытых слоя, обучение с самого раннего доступного года.
  Дневные: Yahoo BTC-USD с 2014-09, цель — знак доходности следующего дня. Тест по годам 2017..2024.
  4-часовые: Binance с 2017-08, цель — знак доходности за следующие 24 часа. Тест по годам 2020..2024.
Каждый тестовый год прогнозируется сетями, обученными на ВСЕЙ истории до него (с зазором). Ансамбль из 5 сидов.
Архитектуры заданы заранее: 2 слоя (64, 32) и 3 слоя (64, 32, 16). Данные 2025+ не читаются."""
import json, sys, warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
ARCH = {"2_layers_64_32": (64, 32), "3_layers_64_32_16": (64, 32, 16)}
END = "2024-12-31"


def net(hidden, seed):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-2, early_stopping=True, validation_fraction=0.15,
                                       n_iter_no_change=15, max_iter=300, random_state=seed))


def walk(X, y, hidden, years, gap_days):
    proba = pd.Series(np.nan, index=X.index); ok = y.notna() & X.notna().mean(axis=1).gt(0.7)
    for yr in years:
        t0, t1 = pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr + 1}-01-01")
        tr = ok & (X.index < t0 - pd.Timedelta(days=gap_days)); te = ok & (X.index >= t0) & (X.index < t1)
        proba[te] = np.mean([net(hidden, s).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] for s in range(5)], axis=0)
    return proba


def report(proba, y, years, blk):
    m = proba.notna() & y.notna(); p, t = proba[m], y[m]; rng = np.random.default_rng(42)
    hit = ((p > 0.5) == (t == 1)).astype(float); conf = (p - 0.5).abs() >= (p - 0.5).abs().quantile(0.8)
    def ci(x, b):
        x = np.asarray(x); n = len(x); nb = int(np.ceil(n / b))
        bs = [x[(rng.integers(0, n, nb)[:, None] + np.arange(b)[None, :]).ravel()[:n] % n].mean() for _ in range(2000)]
        return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    return {"forecasts": int(m.sum()), "always_up_accuracy": round(float(max(t.mean(), 1 - t.mean())), 3),
            "accuracy": round(float(hit.mean()), 3), "ci95": ci(hit.values, blk), "auc": round(float(roc_auc_score(t, p)), 3),
            "top20pct_confident_accuracy": round(float(hit[conf].mean()), 3), "top20pct_ci95": ci(hit[conf].values, max(5, blk // 3)),
            "share_predicted_up": round(float((p > 0.5).mean()), 3),
            "accuracy_by_year": {int(yr): round(float(hit[hit.index.year == yr].mean()), 3) for yr in years}}


def daily_features():
    d = pd.read_csv("data/btc_yahoo_1d_long.csv", index_col=0, parse_dates=True)[:END]
    p, lp = d["close"], np.log(d["close"]); r = lp.diff(); vol = np.sqrt((r**2).ewm(halflife=10, min_periods=30).mean())
    X = pd.DataFrame(index=d.index)
    for w in (10, 20, 50, 100, 200):
        X[f"dist_sma{w}"] = np.log(p / p.rolling(w).mean()) / (vol * np.sqrt(w))
    for k in (1, 3, 7, 30):
        X[f"mom{k}"] = (lp - lp.shift(k)) / (vol * np.sqrt(k))
    X["z20"] = (p - p.rolling(20).mean()) / p.rolling(20).std()
    X["accel"] = (lp - 2 * lp.shift(1) + lp.shift(2)) / vol
    rv = {k: r.rolling(k).std() for k in (7, 30)}
    X["rv7_30"] = rv[7] / rv[30]; X["rv30_rank"] = rv[30].rolling(365, min_periods=180).rank(pct=True)
    X["range"] = np.log(d["high"] / d["low"]) / vol
    for k in (30, 90, 365):
        X[f"dd{k}"] = np.log(p / d["high"].rolling(k, min_periods=30).max())
    X["vol_ratio"] = d["volume"] / d["volume"].rolling(30).mean(); X["dow"] = d.index.dayofweek
    z = (r / vol.shift(1)).clip(-8, 8)
    for k in range(30):
        X[f"r{k}"] = z.shift(k)                                   # сырые доходности последних 30 дней в сигмах
    X = X.replace([np.inf, -np.inf], np.nan)
    fwd = r.shift(-1); return X, (fwd > 0).astype(float).where(fwd.notna())


if __name__ == "__main__":
    res = {"daily_from_2014": {}, "4h_from_2017": {}}
    X, y = daily_features(); yrs = range(2017, 2025)
    print(f"дневные: {X.index[0].date()}..{X.index[-1].date()}, примеров {len(X)}, признаков {X.shape[1]}")
    for name, h in ARCH.items():
        res["daily_from_2014"][name] = report(walk(X, y, h, yrs, 2), y, yrs, 20); print("  дневные", name, json.dumps(res["daily_from_2014"][name]))
        json.dump(res, open("results/phase4/exp14_nn_long.json", "w"), indent=1)
    sys.path.insert(0, "src")
    from meta import load_bars, build, D, TEST_YEARS
    b = load_bars(); X4, _, _ = build(b); lp = np.log(b["close"]); f = lp.shift(-D) - lp; y4 = (f > 0).astype(float).where(f.notna())
    print(f"4-часовые: {X4.index[0]}..{X4.index[-1]}, примеров {len(X4)}, признаков {X4.shape[1]}")
    for name, h in ARCH.items():
        res["4h_from_2017"][name] = report(walk(X4, y4, h, TEST_YEARS, 2), y4, TEST_YEARS, 60); print("  4-часовые", name, json.dumps(res["4h_from_2017"][name]))
        json.dump(res, open("results/phase4/exp14_nn_long.json", "w"), indent=1)
