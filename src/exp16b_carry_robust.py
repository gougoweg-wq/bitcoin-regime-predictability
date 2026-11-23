"""Усиление 16b: устойчивость CARRY. Основной вариант (7 дней, 20%, пересборка раз в 7 дней, топ-30, 10 б.п.) был задан заранее;
здесь проверяется, не случайна ли удача: 108 соседних вариантов, издержки 10 и 30 б.п., разбор на ноги и на источники дохода."""
import itertools, json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
import exp16_breadth as E

ANN = 365


def carry(look=7, q=0.2, reb=7, top=30, cost=10 / 1e4, leg="both"):
    uni = E.liq.rank(axis=1, ascending=False) <= top
    sig = E.Fd.rolling(look, min_periods=max(2, look - 2)).mean().where(uni)
    r = sig.rank(axis=1, pct=True); n = sig.notna().sum(axis=1)
    lo, hi = (r <= q).astype(float), (r > 1 - q).astype(float)
    lo = lo.div(lo.sum(axis=1).replace(0, np.nan), axis=0) * 0.5; hi = hi.div(hi.sum(axis=1).replace(0, np.nan), axis=0) * 0.5
    w = {"both": lo - hi, "long_low_funding": lo, "short_high_funding": -hi}[leg].where(n >= top // 2, 0).fillna(0)
    mask = np.zeros(len(w), dtype=bool); mask[::reb] = True
    held = w.where(pd.Series(mask, index=w.index), np.nan).ffill().fillna(0).shift(1).fillna(0)
    price = (held * E.ret.fillna(0)).sum(axis=1); fund = (-held * E.Fd.fillna(0)).sum(axis=1); c = held.diff().abs().sum(axis=1) * cost
    return (price + fund - c)[E.START:], price[E.START:], fund[E.START:]


def sh(x):
    return float(x.mean() / x.std() * np.sqrt(ANN))


if __name__ == "__main__":
    out = {}
    grid = list(itertools.product((3, 7, 14, 30), (0.1, 0.2, 0.3), (1, 7, 14), (20, 30, 50)))
    for cost in (10, 30):
        s = np.array([sh(carry(l, q, rb, tp, cost / 1e4)[0]) for l, q, rb, tp in grid])
        out[f"grid_cost{cost}bps"] = {"configs": len(grid), "share_sharpe_gt0": round(float((s > 0).mean()), 3), "median": round(float(np.median(s)), 2),
                                      "min": round(float(s.min()), 2), "max": round(float(s.max()), 2), "p25": round(float(np.percentile(s, 25)), 2)}
        print(f"сетка, издержки {cost} б.п.:", out[f"grid_cost{cost}bps"])
    x, price, fund = carry()
    out["main_decomposition_annual"] = {"total": round(float(x.mean() * ANN), 3), "price_part": round(float(price.mean() * ANN), 3), "funding_part": round(float(fund.mean() * ANN), 3)}
    for leg in ("long_low_funding", "short_high_funding"):
        y, p_, f_ = carry(leg=leg)
        out[f"leg_{leg}"] = {"annual_return": round(float(y.mean() * ANN), 3), "sharpe": round(sh(y), 2), "price_part": round(float(p_.mean() * ANN), 3), "funding_part": round(float(f_.mean() * ANN), 3)}
    out["main_cost30bps_sharpe"] = round(sh(carry(cost=30 / 1e4)[0]), 2)
    m = (1 + x).resample("ME").prod() - 1
    out["monthly"] = {"months": len(m), "share_positive": round(float((m > 0).mean()), 3), "worst_month": round(float(m.min()), 3), "worst_month_date": str(m.idxmin().date()), "best_month": round(float(m.max()), 3)}
    half = len(x) // 2
    out["halves_sharpe"] = [round(sh(x[:half]), 2), round(sh(x[half:]), 2)]
    for k in ("main_decomposition_annual", "leg_long_low_funding", "leg_short_high_funding", "main_cost30bps_sharpe", "monthly", "halves_sharpe"):
        print(k, out[k])
    json.dump(out, open("results/phase4/exp16b_carry_robust.json", "w"), indent=1)
