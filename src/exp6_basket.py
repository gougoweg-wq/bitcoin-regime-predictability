"""Усиление 6: корзина BTC+ETH+SOL, каждая монета торгуется по REGIME, веса обратно пропорциональны волатильности
(сумма весов 1, без плеча). Сравнение с REGIME только на BTC и с равновесной корзиной. Период: с появления SOL."""
import json
import numpy as np
import pandas as pd
from lib import *

P = pd.DataFrame({a: load(a) for a in ASSETS}).dropna()
reg = P.apply(regime_pos); vol = P.apply(ewma_vol)
iv = (1 / vol).div((1 / vol).sum(axis=1), axis=0)
ret = P.pct_change()


def port(w):
    held = w.shift(1)
    return ((held * ret).sum(axis=1) - held.diff().abs().sum(axis=1).fillna(0) * COST_BPS / 1e4)[held.notna().all(axis=1)]


runs = {"regime_btc": port(reg[["btc"]].reindex(columns=ASSETS, fill_value=0.0)), "regime_basket_invvol": port(reg * iv),
        "regime_basket_equal": port(reg / 3), "buyhold_basket_invvol": port(iv)}
i = runs["regime_basket_invvol"].index
out = {k: metrics(v[i]) for k, v in runs.items()}
out["sharpe_diff_basket_vs_btc"] = boot_sharpe_diff(runs["regime_basket_invvol"][i].values, runs["regime_btc"][i].values)
out["by_year_sharpe_[basket, btc]"] = {int(y): [round(sharpe(runs["regime_basket_invvol"][i][i.year == y]), 2),
                                                 round(sharpe(runs["regime_btc"][i][i.year == y]), 2)] for y in sorted(set(i.year))}
out["corr_regime_returns"] = pd.DataFrame({a: strat(P[a], reg[a]) for a in ASSETS}).corr().round(2).to_dict()
json.dump(out, open("results/phase4/exp6_basket.json", "w"), indent=1)
for k, v in out.items():
    print(k, {m: round(x, 2) for m, x in v.items()} if "sharpe" in v else v)
