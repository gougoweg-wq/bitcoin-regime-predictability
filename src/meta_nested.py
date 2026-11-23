"""Усиление 8b: честный подбор метамодели. Для каждого тестового года конфигурация выбирается ТОЛЬКО по данным до него:
последние 365 дней обучающего окна — внутренняя проверка (AUC), остальное — обучение. Лучшая конфигурация
переобучается на всём прошлом и применяется к тестовому году. Если лучший внутренний AUC < 0.52, метамодель
отключается (остаётся первичная). Сетка: горизонт метки x барьер x модель x набор признаков = 32 конфигурации."""
import itertools, json
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from meta import *

GRID = list(itertools.product((1, 3, 5, 10), (1.0, 2.0), ("gb", "lr"), ("all", "core")))
EXTRA = ["fund_3d", "fund_30d", "doi_5d", "dow", "hour"]


def make(kind):
    gb, lr = models(1)
    return gb[0] if kind == "gb" else lr


def fit_predict(X, y, tr, te, kind):
    return make(kind).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]


if __name__ == "__main__":
    b = load_bars(); X, primary, vol = build(b)
    L = {(h, k): labels(b, vol, h * D, k) for h in (1, 3, 5, 10) for k in (1.0, 2.0)}
    pos_i = pd.Series(np.arange(len(X)), index=X.index); base_ok = (primary == 1) & X["dist_sma200"].notna()
    proba = pd.Series(np.nan, index=X.index); log, pairs = {}, []
    for yr in TEST_YEARS:
        t0, t1 = pd.Timestamp(f"{yr}-01-01"), pd.Timestamp(f"{yr + 1}-01-01"); v0 = t0 - pd.Timedelta(days=365)
        scores = {}
        for h, k, kind, fs in GRID:
            y, t_end = L[(h, k)]; Xf = X.drop(columns=EXTRA) if fs == "core" else X
            emb = pd.Timedelta(days=h)
            cut_v = pos_i[pos_i.index < v0 - emb].iloc[-1]; cut_t = pos_i[pos_i.index < t0 - emb].iloc[-1]
            tr_in = base_ok & y.notna() & (t_end <= cut_v) & (X.index < v0)
            va = base_ok & y.notna() & (X.index >= v0) & (t_end <= cut_t) & (X.index < t0)
            te = base_ok & (X.index >= t0) & (X.index < t1)
            if tr_in.sum() < 300 or va.sum() < 100 or y[va].nunique() < 2:
                continue
            auc_va = roc_auc_score(y[va], fit_predict(Xf, y, tr_in, va, kind))
            tr_full = base_ok & y.notna() & (t_end <= cut_t) & (X.index < t0)
            p_te = pd.Series(fit_predict(Xf, y, tr_full, te, kind), index=X.index[te])
            m = y[te].notna()
            auc_te = roc_auc_score(y[te][m], p_te[m]) if y[te][m].nunique() == 2 else np.nan
            scores[(h, k, kind, fs)] = (auc_va, auc_te, p_te); pairs.append((auc_va, auc_te))
        best = max(scores, key=lambda c: scores[c][0]); auc_va, auc_te, p_te = scores[best]
        use = auc_va >= 0.52
        proba[p_te.index] = p_te if use else 1.0                      # 1.0 -> полный размер, то есть чистая первичная модель
        log[int(yr)] = {"best_config": list(best), "inner_auc": round(float(auc_va), 3), "test_auc": round(float(auc_te), 3), "meta_used": bool(use),
                        "median_test_auc_all_configs": round(float(np.nanmedian([s[1] for s in scores.values()])), 3),
                        "best_test_auc_in_hindsight": round(float(np.nanmax([s[1] for s in scores.values()])), 3)}
        print(yr, log[int(yr)])
    size = (((proba - 0.5) / 0.1).clip(0, 1) * 4).round() / 4
    oos = b.index >= "2020-01-01"
    pos = {"primary_ensemble": primary.fillna(0), "meta_nested": (primary * size).fillna(0)}
    R = {k: backtest(b, v)[oos] for k, v in pos.items()}
    pv = np.array([p for p in pairs if not np.isnan(p[1])])
    out = {"per_year": log, "strategies": {k: metrics(R[k], pos[k][oos]) for k in R},
           "meta_vs_primary": boot_diff(R["meta_nested"].values, R["primary_ensemble"].values),
           "by_year_sharpe": {int(yr): {k: round(sharpe(R[k][R[k].index.year == yr]), 2) for k in R} for yr in TEST_YEARS},
           "n_configs": len(GRID), "all_configs_test_auc": {"mean": float(pv[:, 1].mean()), "share_above_0.5": float((pv[:, 1] > 0.5).mean())},
           "spearman_inner_vs_test_auc": {"rho": float(spearmanr(pv[:, 0], pv[:, 1])[0]), "p": float(spearmanr(pv[:, 0], pv[:, 1])[1])}}
    json.dump(out, open("results/phase4/exp8b_meta_nested.json", "w"), indent=1)
    for k in ("strategies", "meta_vs_primary", "by_year_sharpe", "all_configs_test_auc", "spearman_inner_vs_test_auc"):
        print(k, json.dumps(out[k], ensure_ascii=False, default=lambda x: round(x, 3)))
