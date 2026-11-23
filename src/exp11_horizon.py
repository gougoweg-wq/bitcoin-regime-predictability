"""Усиление 11: байесовская формула направления на горизонте h дней.
Идея из раздела 6.1 работы: сигнал растёт как h, шум как sqrt(h), значит P(рост за h дней) = Phi(sqrt(h)*mu/sigma) растёт с h.
Последовательный Beta-Binomial по типам рынка: p_up_s(t) = (k_s + n0*p0) / (n_s + n0). Исход дня u известен только на день u+h,
поэтому каждый прогноз сделан без знания будущего. Оценка: дни 2018-01-01..2024-12-31 (до этого модель только копит статистику).
Два набора типов рынка:  S6  = тренд (Bull/Bear) x волатильность (low/mid/high);
                         S12 = сила тренда (T < -1, -1..0, 0..1, > 1 сигм) x волатильность."""
import json, sys
import numpy as np
import pandas as pd
sys.path.insert(0, "src")
from exp10_map import state

N0, EVAL_FROM, END = 250, "2018-01-01", "2024-12-31"


def run(d, h, scheme):
    lp = np.log(d["p"]); fwd = (lp.shift(-h) - lp)
    up = (fwd > 0).astype(float).where(fwd.notna())
    if scheme == "S6":
        s = np.where(d["bull"].astype(bool), "bull", "bear") + pd.Series("|", index=d.index) + d["vol"].astype(str)
    else:
        s = pd.cut(d["T"], [-np.inf, -1, 0, 1, np.inf], labels=["T<-1", "T-1..0", "T0..1", "T>1"]).astype(str) + "|" + d["vol"].astype(str)
    s = s.values; u = up.values; n = len(d)
    cnt, k = {}, {}; tot_n = tot_k = 0; p_hat = np.full(n, np.nan); n_s = np.zeros(n)
    for t in range(n):
        j = t - h                                                   # исход дня j стал известен сегодня
        if j >= 0 and not np.isnan(u[j]):
            cnt[s[j]] = cnt.get(s[j], 0) + 1; k[s[j]] = k.get(s[j], 0) + u[j]; tot_n += 1; tot_k += u[j]
        if tot_n >= 200:
            p0 = tot_k / tot_n
            p_hat[t] = (k.get(s[t], 0) + N0 * p0) / (cnt.get(s[t], 0) + N0); n_s[t] = cnt.get(s[t], 0)
    out = pd.DataFrame({"s": s, "p_hat": p_hat, "up": u, "fwd": fwd.values}, index=d.index).dropna()
    return out[(out.index >= EVAL_FROM)]


def block_ci(x, block, reps=2000, seed=42):
    rng = np.random.default_rng(seed); n = len(x); nb = int(np.ceil(n / block))
    m = [x[(rng.integers(0, n, nb)[:, None] + np.arange(block)[None, :]).ravel()[:n] % n].mean() for _ in range(reps)]
    return [round(float(np.percentile(m, 2.5)), 3), round(float(np.percentile(m, 97.5)), 3)]


if __name__ == "__main__":
    d = state("data/btc_yahoo_1d_long.csv", "2014-09-17", END)
    res = {}
    for scheme in ("S6", "S12"):
        for h in (1, 5, 10, 20, 30):
            o = run(d, h, scheme); base = float(o.up.mean())
            row = {"days": len(o), "base_up_rate": round(base, 3), "brier": round(float(((o.p_hat - o.up) ** 2).mean()), 4),
                   "brier_const": round(float(((base - o.up) ** 2).mean()), 4)}
            for thr in (0.60, 0.65, 0.70):
                m = o.p_hat >= thr
                if m.sum() >= 50:
                    x = (o.up[m].values - base)
                    row[f"p>={thr}"] = {"days": int(m.sum()), "share": round(float(m.mean()), 2), "hit": round(float(o.up[m].mean()), 3),
                                        "lift_vs_base": round(float(x.mean()), 3), "lift_ci95": block_ci(x, block=max(h, 5) * 2),
                                        "mean_fwd_pct": round(float(o.fwd[m].mean() * 100), 2)}
            lo = o.p_hat <= o.p_hat.quantile(0.2)                   # 20% дней с самой низкой прогнозной вероятностью
            row["lowest_20pct"] = {"days": int(lo.sum()), "up_rate": round(float(o.up[lo].mean()), 3), "mean_fwd_pct": round(float(o.fwd[lo].mean() * 100), 2)}
            years = {int(y): round(float(o.up[(o.p_hat >= 0.60) & (o.index.year == y)].mean()), 2) for y in sorted(set(o.index.year)) if ((o.p_hat >= 0.60) & (o.index.year == y)).sum() >= 20}
            row["hit_by_year_p>=0.60"] = years
            res[f"{scheme}|h{h}"] = row
            print(f"\n{scheme} h={h:2d} | база {base:.3f} | Brier {row['brier']} против {row['brier_const']}")
            for kk, vv in row.items():
                if isinstance(vv, dict):
                    print("   ", kk, vv)
    json.dump(res, open("results/phase4/exp11_horizon.json", "w"), indent=1)
