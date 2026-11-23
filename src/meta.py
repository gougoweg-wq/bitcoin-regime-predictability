"""Усиление 8 (только BTC): ансамбль режимов + мета-разметка + метамодель.

Бар решения: 4 часа (признаки считаются из часовых свечей). Период разработки до 2024-12-31, 2025+ не читается.
  Первичная модель: доля окон {10,20,50,100,200 дней}, где P > SMA. Long, если доля >= 0.5.
  Мета-разметка: по барам, где первичная модель в long; тройной барьер +-2 сигмы за 5 дней, таймаут -> знак доходности.
  Метамодель: среднее LightGBM (5 сидов) и логистической регрессии; гиперпараметры заданы заранее, не подбираются.
  Размер позиции: clip((p - 0.5) / 0.1, 0, 1) с шагом 0.25.
  Walk-forward: тест по годам 2020..2024, обучение на всём прошлом, зазор 5 дней (метки не заглядывают в тест).
Запуск: .venv/bin/python src/meta.py
"""
import json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, log_loss

warnings.filterwarnings("ignore")
END, BAR_H, D = "2024-12-31 23:59", 4, 6                 # D = баров в сутках
ANN, COST = 365 * D, 10 / 1e4
WINDOWS, HOLD, BARRIER = (10, 20, 50, 100, 200), 5 * D, 2.0
TEST_YEARS = range(2020, 2025)


def load_bars():
    h = pd.read_csv("data/btc_binance_1h.csv", index_col=0, parse_dates=True)[:END]
    b = h.resample("4h").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum", "qvol": "sum", "taker_buy_vol": "sum"}).dropna()
    b.index = b.index + pd.Timedelta(hours=BAR_H)        # метка времени = момент закрытия бара
    return b


def build(b):
    p, lp = b["close"], np.log(b["close"]); r = lp.diff()
    vol = np.sqrt((r**2).ewm(halflife=5 * D, min_periods=10 * D).mean())            # сигма на бар
    X = pd.DataFrame(index=b.index)
    above = []
    for w in WINDOWS:
        sma = p.rolling(w * D).mean()
        X[f"dist_sma{w}"] = np.log(p / sma) / (vol * np.sqrt(w * D)); above.append((p > sma).where(sma.notna()))
    score = pd.concat(above, axis=1).astype(float).mean(axis=1, skipna=False)
    X["trend_score"] = score
    primary = (score >= 0.5).astype(float).where(score.notna())
    X["bars_since_flip"] = primary.groupby((primary != primary.shift()).cumsum()).cumcount()
    for d in (1, 3, 7, 30):
        X[f"mom{d}d"] = (lp - lp.shift(d * D)) / (vol * np.sqrt(d * D))
    X["z20"] = (p - p.rolling(20 * D).mean()) / p.rolling(20 * D).std()               # Z из работы
    X["accel"] = (lp - 2 * lp.shift(D) + lp.shift(2 * D)) / (vol * np.sqrt(D))        # «ускорение» из работы, в сигмах
    rv = {d: r.rolling(d * D).std() for d in (1, 7, 30)}
    X["rv1_30"], X["rv7_30"] = rv[1] / rv[30], rv[7] / rv[30]
    X["rv30_rank"] = rv[30].rolling(365 * D, min_periods=90 * D).rank(pct=True)
    X["range1d"] = np.log(b["high"].rolling(D).max() / b["low"].rolling(D).min()) / (vol * np.sqrt(D))
    for d in (30, 90, 365):
        X[f"dd{d}d"] = np.log(p / b["high"].rolling(d * D, min_periods=30 * D).max())
    X["vol_ratio"] = b["qvol"].rolling(D).sum() / b["qvol"].rolling(30 * D).sum() * 30
    tb = b["taker_buy_vol"] / b["volume"]
    X["taker_buy_1d"], X["taker_buy_7d"] = tb.rolling(D).mean() - 0.5, tb.rolling(7 * D).mean() - 0.5
    X["dow"], X["hour"] = b.index.dayofweek, b.index.hour
    f = pd.read_csv("data/btc_funding.csv", index_col=0, parse_dates=True)["last_funding_rate"]
    f.index = f.index.floor("s")
    f = f.reindex(b.index.union(f.index)).sort_index()
    X["fund_3d"] = f.rolling("3D").mean().reindex(b.index); X["fund_30d"] = f.rolling("30D").mean().reindex(b.index)
    oi = pd.read_csv("data/btc_oi.csv", index_col=0, parse_dates=True)["oi"]
    oi.index = oi.index + pd.Timedelta(days=1)                                         # снимок 23:55 дня D доступен с D+1 00:00
    oi4 = oi.reindex(b.index.union(oi.index)).sort_index().ffill(limit=D).reindex(b.index)
    X["doi_5d"] = oi4 / oi4.shift(5 * D) - 1

    X = X.replace([np.inf, -np.inf], np.nan)

    return X, primary, vol


def labels(b, vol, hold=HOLD, barrier=BARRIER):
    """Тройной барьер: +-barrier сигм за hold баров; таймаут -> знак доходности. Возвращает (y, индекс бара окончания)."""
    a = np.log(b["close"]).values; v = (vol * np.sqrt(hold) * barrier).values; n = len(a)
    y = np.full(n, np.nan); t_end = np.full(n, -1)
    for i in range(n - hold):
        if np.isnan(v[i]):
            continue
        path = a[i + 1:i + hold + 1] - a[i]
        up = np.argmax(path >= v[i]) if (path >= v[i]).any() else hold + 1
        dn = np.argmax(path <= -v[i]) if (path <= -v[i]).any() else hold + 1
        if up < dn: y[i], t_end[i] = 1, i + 1 + up
        elif dn < up: y[i], t_end[i] = 0, i + 1 + dn
        else: y[i], t_end[i] = float(path[-1] > 0), i + hold
    return pd.Series(y, index=b.index), pd.Series(t_end, index=b.index)


def models(seed_n=5):
    gb = [lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=8, min_child_samples=200, subsample=0.7, subsample_freq=1,
                             colsample_bytree=0.7, reg_lambda=5.0, random_state=s, verbose=-1) for s in range(seed_n)]
    lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
    return gb, lr


def walk_forward(X, primary, y, t_end):
    ok = (primary == 1) & y.notna() & X["dist_sma200"].notna()
    proba = pd.Series(np.nan, index=X.index); parts = {}
    idx_pos = pd.Series(np.arange(len(X)), index=X.index)
    for yr in TEST_YEARS:
        t0, t1 = pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr + 1}-01-01")
        cut = idx_pos[idx_pos.index < t0 - pd.Timedelta(days=5)].iloc[-1]
        tr = ok & (t_end <= cut) & (X.index < t0)                                     # метка целиком до зазора
        te = (primary == 1) & (X.index >= t0) & (X.index < t1) & X["dist_sma200"].notna()
        gb, lr = models()
        ps = [m.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] for m in gb]
        pl = lr.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        proba[te] = (np.mean(ps, axis=0) + pl) / 2
        parts[yr] = {"train_n": int(tr.sum()), "test_n": int(te.sum()), "gb": np.mean(ps, axis=0), "lr": pl, "te": te}
    return proba, parts


def backtest(b, pos):
    held = pos.shift(1).fillna(0)
    return held * b["close"].pct_change() - held.diff().abs().fillna(0) * COST


def sharpe(x):
    return float(x.mean() / x.std() * np.sqrt(ANN)) if x.std() > 0 else 0.0


def metrics(x, pos):
    eq = (1 + x).cumprod(); dd = float((eq / eq.cummax() - 1).min()); cagr = float(eq.iloc[-1] ** (ANN / len(x)) - 1)
    dn = x[x < 0]
    return {"sharpe": sharpe(x), "sortino": float(x.mean() / dn.std() * np.sqrt(ANN)), "cagr": cagr, "max_dd": dd, "calmar": cagr / abs(dd),
            "avg_exposure": float(pos.mean()), "turnover_per_year": float(pos.diff().abs().sum() / len(pos) * ANN),
            "profit_factor": float(x[x > 0].sum() / -x[x < 0].sum())}


def boot_diff(a, c, block=20 * D, reps=2000, seed=42):
    rng = np.random.default_rng(seed); m = np.column_stack([a, c]); n = len(m); nb = int(np.ceil(n / block)); d = []
    for _ in range(reps):
        s = m[(rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n] % n]
        d.append((s[:, 0].mean() / s[:, 0].std() - s[:, 1].mean() / s[:, 1].std()) * np.sqrt(ANN))
    return {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_better": float(np.mean(np.array(d) > 0))}


if __name__ == "__main__":
    b = load_bars(); X, primary, vol = build(b); y, t_end = labels(b, vol)
    proba, parts = walk_forward(X, primary, y, t_end)
    size = (((proba - 0.5) / 0.1).clip(0, 1) * 4).round() / 4
    oos = (b.index >= "2020-01-01")
    daily = b["close"].resample("1D", closed="right", label="right").last(); sma50 = (daily > daily.rolling(50).mean()).astype(float).where(daily.rolling(50).mean().notna())
    pos = {"buyhold": pd.Series(1.0, index=b.index),
           "regime_sma50_daily": sma50.reindex(b.index, method="ffill"),             # baseline: решение раз в сутки в 00:00
           "primary_ensemble": primary.fillna(0),
           "meta": (primary * size).fillna(0)}
    R = {k: backtest(b, v)[oos] for k, v in pos.items()}
    out = {"period": "2020-01-01..2024-12-31", "cost_bps": 10, "strategies": {k: metrics(R[k], pos[k][oos]) for k in R}}
    out["meta_vs_primary"] = boot_diff(R["meta"].values, R["primary_ensemble"].values)
    out["meta_vs_regime_sma50"] = boot_diff(R["meta"].values, R["regime_sma50_daily"].values)
    out["primary_vs_regime_sma50"] = boot_diff(R["primary_ensemble"].values, R["regime_sma50_daily"].values)
    out["by_year_sharpe"] = {int(yr): {k: round(sharpe(R[k][R[k].index.year == yr]), 2) for k in R} for yr in TEST_YEARS}
    te_all = proba.notna() & y.notna()
    out["classifier_oos"] = {"n": int(te_all.sum()), "base_rate": float(y[te_all].mean()), "auc": float(roc_auc_score(y[te_all], proba[te_all])),
                             "logloss": float(log_loss(y[te_all], proba[te_all])),
                             "logloss_constant": float(log_loss(y[te_all], np.full(te_all.sum(), y[te_all].mean()))),
                             "auc_by_year": {}, "hit_rate_by_proba_bucket": {}}
    for yr, pt in parts.items():
        m = pt["te"] & y.notna(); yy = y[m]
        sel = y[pt["te"]].notna().values
        out["classifier_oos"]["auc_by_year"][int(yr)] = {"train_n": pt["train_n"], "test_n": int(m.sum()), "auc_ens": round(float(roc_auc_score(yy, proba[m])), 3),
                                                         "auc_gb": round(float(roc_auc_score(yy, pt["gb"][sel])), 3), "auc_lr": round(float(roc_auc_score(yy, pt["lr"][sel])), 3)}
    for lo, hi in ((0, .45), (.45, .5), (.5, .55), (.55, .6), (.6, 1)):
        m = te_all & (proba >= lo) & (proba < hi)
        out["classifier_oos"]["hit_rate_by_proba_bucket"][f"{lo}-{hi}"] = {"n": int(m.sum()), "label_rate": round(float(y[m].mean()), 3) if m.sum() else None}
    json.dump(out, open("results/phase4/exp8_meta.json", "w"), indent=1)
    proba.to_frame("proba").assign(primary=primary, y=y).to_csv("results/phase4/exp8_meta_proba.csv")
    for k, v in out["strategies"].items():
        print(f"{k:20s}", {m: round(x, 2) for m, x in v.items()})
    for k in ("meta_vs_primary", "meta_vs_regime_sma50", "primary_vs_regime_sma50", "by_year_sharpe", "classifier_oos"):
        print(k, json.dumps(out[k], ensure_ascii=False))
