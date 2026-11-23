"""Усиление 7, шаг B: стратегия из единственной находки исследования позиционирования.
«Смыв»: Z_t < -1.9 и OI за 5 дней падал (ранг dOI < 1/3) -> плечевые лонги уже ликвидированы -> long на 5 дней,
даже если режим Bear. Иначе обычный REGIME. Правило сформулировано ПОСЛЕ взгляда на данные 2020-2024,
поэтому A/B здесь внутривыборочный; настоящая проверка — hold-out 2025+."""
import pandas as pd
from lib import *
from exp7_positioning_study import exp_rank, W

H, LAM = 5, 1.9


def flush_signal(p, asset):
    oi = pd.read_csv(f"data/{asset}_oi.csv", index_col=0, parse_dates=True)["oi"].reindex(p.index)
    z = (p - p.rolling(W).mean()) / p.rolling(W).std()
    return ((z < -LAM) & (exp_rank(oi.pct_change(5)) < 1 / 3)).astype(float)


def make(asset):
    def pos(p):
        sig = flush_signal(p, asset)
        return pd.concat([regime_pos(p), sig.rolling(H).max()], axis=1).max(axis=1).where(regime_pos(p).notna())
    return pos


if __name__ == "__main__":
    import json
    out = {}
    for a in ASSETS:
        p = load(a); start = "2021-06-01" if a == "btc" else "2022-09-01"      # с момента, когда у ранга OI есть история
        rn, rb = strat(p, make(a)(p))[start:], strat(p, regime_pos(p))[start:]
        out[a] = {"new": metrics(rn), "base": metrics(rb), "sharpe_diff": boot_sharpe_diff(rn.values, rb.values),
                  "signal_days": int(flush_signal(p, a)[start:].sum()),
                  "years_[new, base]": {int(y): [round(sharpe(rn[rn.index.year == y]), 2), round(sharpe(rb[rb.index.year == y]), 2)] for y in sorted(set(rn.index.year))}}
        v = out[a]
        print(f"{a}: Sharpe {v['base']['sharpe']:.2f} -> {v['new']['sharpe']:.2f} | maxDD {v['base']['max_dd']:.0%} -> {v['new']['max_dd']:.0%} | "
              f"CAGR {v['base']['cagr']:.0%} -> {v['new']['cagr']:.0%} | dSharpe CI {[round(x, 2) for x in v['sharpe_diff']['ci95']]} "
              f"P(лучше)={v['sharpe_diff']['p_new_better']:.2f} | дней с сигналом {v['signal_days']} | по годам {v['years_[new, base]']}")
    json.dump(out, open("results/phase4/exp7_flush_overlay.json", "w"), indent=1)
