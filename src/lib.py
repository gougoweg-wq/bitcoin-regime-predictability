"""Общий стенд Фазы 4: данные Binance, бэктест с издержками, A/B против текущей лучшей стратегии.

Период разработки 2017-08-17..2024-12-31. Данные 2025+ здесь недоступны намеренно (hold-out).
"""
import json, os
import numpy as np
import pandas as pd

START, END, ANN, COST_BPS = "2017-08-17", "2024-12-31", 365, 10
ASSETS = ("btc", "eth", "sol")


def load(asset):
    return pd.read_csv(f"data/{asset}_binance_1d.csv", index_col=0, parse_dates=True).loc[START:END, "close"]


def ewma_vol(p, halflife=20):
    """Дневная волатильность по данным до t включительно."""
    return np.sqrt((np.log(p).diff() ** 2).ewm(halflife=halflife, min_periods=30).mean())


def regime_pos(p, w=50):
    """Baseline: 1, если P_t > SMA_w, иначе 0."""
    sma = p.rolling(w).mean()
    return (p > sma).astype(float).where(sma.notna())


def strat(p, pos, cost_bps=COST_BPS):
    held = pos.shift(1)
    return (held * p.pct_change() - held.diff().abs().fillna(0) * cost_bps / 1e4).dropna()


def sharpe(x):
    return float(x.mean() / x.std() * np.sqrt(ANN)) if len(x) > 1 and x.std() > 0 else 0.0


def metrics(x):
    eq = (1 + x).cumprod(); dd = float((eq / eq.cummax() - 1).min())
    cagr = float(eq.iloc[-1] ** (ANN / len(x)) - 1); act = x[x != 0]
    return {"sharpe": sharpe(x), "sortino": float(x.mean() / x[x < 0].std() * np.sqrt(ANN)), "cagr": cagr,
            "max_dd": dd, "calmar": cagr / abs(dd) if dd else 0.0, "hit_rate": float((act > 0).mean()),
            "profit_factor": float(act[act > 0].sum() / -act[act < 0].sum())}


def boot_sharpe_diff(a, b, block=20, reps=2000, seed=42):
    """Парный круговой блочный bootstrap: 95% ДИ и доля реплик, где Sharpe(a) > Sharpe(b)."""
    rng = np.random.default_rng(seed); m = np.column_stack([a, b]); n = len(m); nb = int(np.ceil(n / block))
    d = []
    for _ in range(reps):
        s = m[(rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n] % n]
        d.append((s[:, 0].mean() / s[:, 0].std() - s[:, 1].mean() / s[:, 1].std()) * np.sqrt(ANN))
    return {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_new_better": float(np.mean(np.array(d) > 0))}


def ab(name, new_fn, base_fn, start=None, assets=ASSETS):
    """A/B: new_fn(p) и base_fn(p) возвращают позиции. Критерий: Sharpe лучше на >=2 из 3 активов
    и в >=5 из 8 лет на BTC (при издержках 10 б.п.)."""
    out = {"name": name, "assets": {}}
    for a in assets:
        p = load(a); rn, rb = strat(p, new_fn(p)), strat(p, base_fn(p))
        i = rn.index.intersection(rb.index); i = i[i >= start] if start else i
        rn, rb = rn[i], rb[i]
        out["assets"][a] = {"new": metrics(rn), "base": metrics(rb), "sharpe_diff": boot_sharpe_diff(rn.values, rb.values),
                            "turnover_new": float(new_fn(p).diff().abs().sum() / len(p) * ANN),
                            "turnover_base": float(base_fn(p).diff().abs().sum() / len(p) * ANN)}
        if a == "btc":
            yrs = sorted(set(i.year))
            out["btc_years"] = {int(y): [round(sharpe(rn[rn.index.year == y]), 3), round(sharpe(rb[rb.index.year == y]), 3)] for y in yrs}
    wins_a = sum(v["new"]["sharpe"] > v["base"]["sharpe"] for v in out["assets"].values())
    wins_y = sum(n_ > b_ for n_, b_ in out["btc_years"].values())
    out["assets_won"], out["btc_years_won"], out["btc_years_total"] = wins_a, wins_y, len(out["btc_years"])
    out["accepted"] = bool(wins_a >= min(2, len(assets)) and wins_y >= np.ceil(len(out["btc_years"]) * 5 / 8))
    os.makedirs("results/phase4", exist_ok=True)
    json.dump(out, open(f"results/phase4/{name}.json", "w"), indent=1)
    print(f"\n### {name}: accepted={out['accepted']} | активов {wins_a}/{len(assets)}, лет BTC {wins_y}/{len(out['btc_years'])}")
    for a, v in out["assets"].items():
        print(f"  {a}: Sharpe {v['base']['sharpe']:.2f} -> {v['new']['sharpe']:.2f} | maxDD {v['base']['max_dd']:.0%} -> {v['new']['max_dd']:.0%}"
              f" | Calmar {v['base']['calmar']:.2f} -> {v['new']['calmar']:.2f} | CAGR {v['base']['cagr']:.0%} -> {v['new']['cagr']:.0%}"
              f" | dSharpe CI {np.round(v['sharpe_diff']['ci95'], 2)} P(лучше)={v['sharpe_diff']['p_new_better']:.2f}"
              f" | оборот {v['turnover_base']:.0f} -> {v['turnover_new']:.0f}")
    print("  BTC по годам [new, base]:", out["btc_years"])
    return out
