"""Усиление 10, шаг B: байесовская формула по типам рынка (исправленная «Master formula»).

Тип рынка s_t — одна из 6 клеток (тренд Bull/Bear x волатильность low/mid/high), известен на закрытии дня t.
Для каждой клетки ведём апостериор средней доходности следующего дня, обновляя его каждый день только прошлым:
    mu_s ~ N(mu0, sigma_s^2 / n0)                           априор: n0 «виртуальных» дней со средней mu0 (общая средняя)
    mu_hat_s(t) = (n_s * xbar_s + n0 * mu0) / (n_s + n0)    апостериорное среднее
    P(mu_s > 0 | данные до t) = Phi( mu_hat_s / (sigma_s / sqrt(n_s + n0)) )
и апостериор вероятности роста (Beta-Binomial):  p_up_s(t) = (k_s + n0 * p0) / (n_s + n0).
Позиция: половина Келли, w_t = clip(0.5 * mu_hat_s / sigma_s^2, 0, 1). n0 = 250 задан заранее.
Обновление последовательное, поэтому каждое решение принимается по данным, которых модель ещё не видела."""
import json, sys
import numpy as np
import pandas as pd
from scipy.stats import norm
sys.path.insert(0, "src")
from exp10_map import state
from lib import metrics, sharpe, boot_sharpe_diff

N0, COST = 250, 10 / 1e4


def posterior(d, n0=N0):
    d = d.copy(); d["s"] = pd.Series(np.where(d["bull"].astype(bool), "bull", "bear"), index=d.index) + "|" + d["vol"].astype(str)
    known = d["r_next"].shift(1)                                    # исход дня t-1 известен на закрытии дня t
    # накопленные статистики по каждой клетке, доступные на день t
    stats = {}
    for s in d["s"].unique():
        m = (d["s"].shift(1) == s)
        x = known.where(m)
        stats[s] = pd.DataFrame({"n": m.cumsum(), "sum": x.fillna(0).cumsum(), "sq": (x**2).fillna(0).cumsum(), "k": (x > 0).cumsum()})
    mu0 = known.expanding(100).mean(); p0 = (known > 0).astype(float).where(known.notna()).expanding(100).mean(); sd0 = known.expanding(100).std()
    n = pd.Series([stats[s]["n"].iloc[i] for i, s in enumerate(d["s"])], index=d.index)
    sm = pd.Series([stats[s]["sum"].iloc[i] for i, s in enumerate(d["s"])], index=d.index)
    sq = pd.Series([stats[s]["sq"].iloc[i] for i, s in enumerate(d["s"])], index=d.index)
    k = pd.Series([stats[s]["k"].iloc[i] for i, s in enumerate(d["s"])], index=d.index)
    sd_s = np.sqrt((sq - sm**2 / n.clip(lower=1)) / (n - 1).clip(lower=1)).where(n >= 30, sd0)
    d["mu_hat"] = (sm + n0 * mu0) / (n + n0)
    d["p_mu_pos"] = norm.cdf(d["mu_hat"] / (sd_s / np.sqrt(n + n0)))
    d["p_up"] = (k + n0 * p0) / (n + n0)
    d["w"] = (0.5 * d["mu_hat"] / sd_s**2).clip(0, 1)
    return d.dropna(subset=["mu_hat", "w"])


def run(path, start, eval_from, n0=N0):
    d = posterior(state(path, start), n0); d = d[d.index >= eval_from]
    simple = np.exp(d["r_next"]) - 1                               # доходность дня t+1
    def pnl(w):
        return (w * simple - w.diff().abs().fillna(0) * COST)
    R = {"bayes": pnl(d["w"]), "regime_sma50": pnl(d["bull"].astype(float)), "buyhold": simple}
    out = {"from": str(d.index[0].date()), "days": len(d), "strategies": {k: metrics(v) for k, v in R.items()},
           "avg_exposure": float(d["w"].mean()), "turnover_per_year": float(d["w"].diff().abs().sum() / len(d) * 365),
           "bayes_vs_regime": boot_sharpe_diff(R["bayes"].values, R["regime_sma50"].values),
           "bayes_vs_buyhold": boot_sharpe_diff(R["bayes"].values, R["buyhold"].values),
           "by_year_sharpe_[bayes, regime, buyhold]": {int(y): [round(sharpe(v[v.index.year == y]), 2) for v in R.values()] for y in sorted(set(d.index.year))}}
    up = (d["r_next"] > 0).astype(float); base = up.expanding().mean().shift(1).fillna(0.5)
    out["direction_forecast"] = {"brier_bayes": float(((d["p_up"] - up) ** 2).mean()), "brier_base_rate": float(((base - up) ** 2).mean()),
                                 "hit_when_p_up>0.55": [int((d.p_up > .55).sum()), round(float(up[d.p_up > .55].mean()), 3)],
                                 "hit_when_p_up<0.50": [int((d.p_up < .50).sum()), round(float(1 - up[d.p_up < .50].mean()), 3)]}
    last = d.groupby("s").tail(1).set_index("s")[["mu_hat", "p_mu_pos", "p_up", "w"]].round(4)
    out["posterior_at_end_2024"] = last.to_dict("index")
    return out, d


if __name__ == "__main__":
    res = {}
    for name, path, start, ev in (("yahoo_long", "data/btc_yahoo_1d_long.csv", "2014-09-17", "2017-01-01"),
                                  ("binance", "data/btc_binance_1d.csv", "2017-08-17", "2019-01-01")):
        res[name], d = run(path, start, ev)
        r = res[name]; print(f"\n===== {name}: с {r['from']}, дней {r['days']}, средняя позиция {r['avg_exposure']:.2f}, оборот {r['turnover_per_year']:.1f}/год")
        for k, v in r["strategies"].items():
            print(f"  {k:13s}", {m: round(x, 2) for m, x in v.items()})
        print("  bayes vs regime :", r["bayes_vs_regime"]); print("  bayes vs buyhold:", r["bayes_vs_buyhold"])
        print("  по годам [bayes, regime, buyhold]:", r["by_year_sharpe_[bayes, regime, buyhold]"])
        print("  прогноз направления:", r["direction_forecast"])
        print("  апостериор на конец 2024:"); [print("   ", k, v) for k, v in sorted(r["posterior_at_end_2024"].items())]
        if name == "yahoo_long":
            d[["s", "mu_hat", "p_mu_pos", "p_up", "w"]].to_csv("results/phase4/exp10_bayes_daily.csv")
    res["sensitivity_n0_yahoo"] = {n0: round(run("data/btc_yahoo_1d_long.csv", "2014-09-17", "2017-01-01", n0)[0]["strategies"]["bayes"]["sharpe"], 3) for n0 in (50, 100, 250, 500, 1000)}
    print("\nчувствительность к n0 (Sharpe, Yahoo):", res["sensitivity_n0_yahoo"])
    json.dump(res, open("results/phase4/exp10_bayes.json", "w"), indent=1, default=float)
