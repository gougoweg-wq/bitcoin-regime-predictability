"""Усиление 13: нейросети для направления BTC. Цель: знак доходности за следующие 24 часа (решение каждые 4 часа).
Две сети (ансамбль из 5 сидов каждая, архитектура задана заранее):
  A) MLP 64-32 на 25 признаках из meta.py (тренд, Z, ускорение, волатильность, объём, поток покупок, фандинг, OI);
  B) MLP 128-64 на сырых данных: последние 120 часовых доходностей и объёмов в сигмах (сеть сама ищет паттерны).
Walk-forward: тест 2020..2024, обучение на всём прошлом с зазором 2 дня. Точность считается по всем барам
и по 20% самых уверенных прогнозов; ДИ — блочный bootstrap."""
import json, warnings
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from meta import load_bars, build, D, TEST_YEARS

warnings.filterwarnings("ignore")
H = D                                                            # 24 часа = 6 баров


def raw_features(b):
    h = pd.read_csv("data/btc_binance_1h.csv", index_col=0, parse_dates=True)[:"2024-12-31 23:59"]
    r = np.log(h["close"]).diff(); s = r.rolling(24 * 30).std()
    z = (r / s).clip(-8, 8); v = np.log(h["qvol"] / h["qvol"].rolling(24 * 30).mean()).clip(-5, 5)
    z.index = z.index + pd.Timedelta(hours=1); v.index = v.index + pd.Timedelta(hours=1)      # метка = закрытие часа
    cols = {f"r{k}": z.shift(k) for k in range(120)} | {f"v{k}": v.shift(k) for k in range(0, 120, 4)}
    return pd.DataFrame(cols).reindex(b.index)


def net(hidden, seed):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         MLPClassifier(hidden_layer_sizes=hidden, alpha=1e-2, early_stopping=True, validation_fraction=0.15,
                                       n_iter_no_change=10, max_iter=200, random_state=seed))


def walk(X, y, hidden):
    proba = pd.Series(np.nan, index=X.index); ok = y.notna() & X.notna().mean(axis=1).gt(0.7)
    for yr in TEST_YEARS:
        t0, t1 = pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr + 1}-01-01")
        tr = ok & (X.index < t0 - pd.Timedelta(days=2)); te = ok & (X.index >= t0) & (X.index < t1)
        proba[te] = np.mean([net(hidden, s).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1] for s in range(5)], axis=0)
    return proba


def report(proba, y, name):
    m = proba.notna() & y.notna(); p, t = proba[m], y[m]
    hit = ((p > 0.5) == (t == 1)).astype(float); conf = (p - 0.5).abs() >= (p - 0.5).abs().quantile(0.8)
    rng = np.random.default_rng(42)
    def ci(x, blk=60):
        x = np.asarray(x); n = len(x); nb = int(np.ceil(n / blk))
        bs = [x[(rng.integers(0, n, nb)[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n].mean() for _ in range(2000)]
        return [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
    out = {"bars": int(m.sum()), "base_up_rate": round(float(t.mean()), 3), "always_up_accuracy": round(float(max(t.mean(), 1 - t.mean())), 3),
           "accuracy": round(float(hit.mean()), 3), "accuracy_ci95": ci(hit.values), "auc": round(float(roc_auc_score(t, p)), 3),
           "top20pct_confident": {"bars": int(conf.sum()), "accuracy": round(float(hit[conf].mean()), 3), "ci95": ci(hit[conf].values, 20)},
           "accuracy_by_year": {int(yr): round(float(hit[hit.index.year == yr].mean()), 3) for yr in TEST_YEARS}}
    print(name, json.dumps(out, ensure_ascii=False)); return out


if __name__ == "__main__":
    b = load_bars(); X, primary, vol = build(b)
    lp = np.log(b["close"]); fwd = lp.shift(-H) - lp; y = (fwd > 0).astype(float).where(fwd.notna())
    res = {"A_mlp_features": report(walk(X, y, (64, 32)), y, "A: MLP на признаках      "),
           "B_mlp_raw_120h": report(walk(raw_features(b), y, (128, 64)), y, "B: MLP на сырых данных   ")}
    json.dump(res, open("results/phase4/exp13_nn.json", "w"), indent=1)
