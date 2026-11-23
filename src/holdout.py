"""Финальная проверка на 2025-01-01..2026-10-31. Запускается ОДИН раз, критерии в results/holdout/PREREGISTRATION.md.
Код экспериментов берётся из тех же файлов, что и в разработке: меняются только даты периода и папка вывода."""
import io, json, sys, contextlib, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
HO0, HO1, ANN = "2025-01-01", "2026-10-31", 365
OUT = {}


def run_patched(path, repl, argv=None):
    """Исполняет файл эксперимента как __main__ с заменой дат; возвращает напечатанный текст."""
    src = open(path).read()
    for a, b in repl:
        assert a in src, (path, a)
        src = src.replace(a, b)
    src = src.replace("results/phase4/", "results/holdout/")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(compile(src, path, "exec"), {"__name__": "__main__", "__file__": path})
    return buf.getvalue()


def sharpe(x):
    return float(x.mean() / x.std() * np.sqrt(ANN))


def boot(x, fn, blk=20, reps=2000):
    rng = np.random.default_rng(42); x = np.asarray(x); n = len(x); nb = int(np.ceil(n / blk))
    s = [fn(x[(rng.integers(0, n, nb)[:, None] + np.arange(blk)[None, :]).ravel()[:n] % n]) for _ in range(reps)]
    return [round(float(np.percentile(s, 2.5)), 3), round(float(np.percentile(s, 97.5)), 3)]


def nw_t(x, L=5):
    x = np.asarray(x); n = len(x); e = x - x.mean()
    v = e @ e / n + 2 * sum((1 - j / (L + 1)) * (e[j:] @ e[:-j]) / n for j in range(1, L + 1))
    return float(x.mean() / np.sqrt(v / n))


# ---------- 1, 2, 12: BTC, Binance, дневные
p = pd.read_csv("data/btc_binance_1d.csv", index_col=0, parse_dates=True)["close"][:HO1]
lp = np.log(p); r = lp.diff(); sma = p.rolling(50).mean(); bull = (p > sma).astype(float)
held = bull.shift(1); reg = (held * p.pct_change() - held.diff().abs().fillna(0) * 10 / 1e4)[HO0:]; bh = p.pct_change()[HO0:]


def m(x):
    eq = (1 + x).cumprod(); dd = float((eq / eq.cummax() - 1).min()); c = float(eq.iloc[-1] ** (ANN / len(x)) - 1)
    return {"sharpe": round(sharpe(x), 2), "sharpe_ci95": boot(x.values, lambda a: a.mean() / a.std() * np.sqrt(ANN)), "cagr": round(c, 3), "max_dd": round(dd, 3),
            "total_return": round(float(eq.iloc[-1] - 1), 3)}


OUT["1_regime_vs_buyhold"] = {"days": len(reg), "regime": m(reg), "buyhold": m(bh), "time_in_market": round(float(held[HO0:].mean()), 2),
                              "sharpe_diff_ci95": boot(np.column_stack([reg.values, bh.values]), lambda a: (a[:, 0].mean() / a[:, 0].std() - a[:, 1].mean() / a[:, 1].std()) * np.sqrt(ANN))}
nxt = r.shift(-1)[HO0:].dropna(); b_ = bull[nxt.index]
OUT["2_next_day_bull_minus_bear"] = {"mu_bull_pct": round(float(nxt[b_ == 1].mean() * 100), 3), "mu_bear_pct": round(float(nxt[b_ == 0].mean() * 100), 3),
                                     "days_bull": int((b_ == 1).sum()), "days_bear": int((b_ == 0).sum()), "t_bull": round(nw_t(nxt[b_ == 1].values), 2), "t_bear": round(nw_t(nxt[b_ == 0].values), 2)}
z = (p - p.rolling(20).mean()) / p.rolling(20).std(); f5 = lp.shift(-5) - lp; ev = (z.abs() > 2.4) & f5.notna() & (z.index >= HO0)
OUT["12_paper_z_momentum"] = {"events": int(ev.sum()), "accuracy": round(float((np.sign(z[ev]) == np.sign(f5[ev])).mean()), 3) if ev.sum() else None,
                              "always_long_accuracy": round(float((f5[ev] > 0).mean()), 3) if ev.sum() else None, "base_up_rate_5d": round(float((f5[HO0:].dropna() > 0).mean()), 3)}

# ---------- 3: байесовская формула (последовательный апостериор продолжает обновляться)
from exp10_map import state
from exp10_bayes import posterior
d = posterior(state("data/btc_yahoo_1d_long.csv", "2014-09-17", HO1)); d = d[d.index >= HO0]
up = (d.r_next > 0).astype(float); cells = {}
for s, g in d.groupby("s"):
    cells[s] = {"days": len(g), "up_rate": round(float((g.r_next > 0).mean()), 3), "mean_next_pct": round(float(g.r_next.mean() * 100), 3), "t": round(nw_t(g.r_next.values), 2) if len(g) > 20 else None}
hi = d.p_up > 0.55
OUT["3_bayes_formula"] = {"days": len(d), "base_up_rate": round(float(up.mean()), 3), "cells": cells,
                          "p_up>0.55": {"days": int(hi.sum()), "accuracy": round(float(up[hi].mean()), 3) if hi.sum() else None, "ci95": boot(up[hi].values, np.mean) if hi.sum() > 40 else None},
                          "brier_bayes": round(float(((d.p_up - up) ** 2).mean()), 4), "brier_const_0.53": round(float(((0.53 - up) ** 2).mean()), 4)}

# ---------- 4: размах
txt = run_patched("src/exp12_vol_predict.py", [('"2014-09-17":"2024-12-31"', f'"2014-09-17":"{HO1}"'), ('.loc["2018-01-01":]', f'.loc["{HO0}":]')])
OUT["4_vol_forecast"] = json.load(open("results/holdout/exp12_vol_predict.json"))

# ---------- 5: нейросеть, дневная, 3 слоя (и 2 слоя справочно)
import exp14_nn_long as N
N.END = HO1
X, y = N.daily_features(); yrs = range(2025, 2027)
OUT["5_nn_daily"] = {name: N.report(N.walk(X, y, h, yrs, 2), y, yrs, 20) for name, h in N.ARCH.items()}

# ---------- 6: уровень фандинга BTC; 7: керри BTC
f = pd.read_csv("data/btc_funding.csv", index_col=0, parse_dates=True)["last_funding_rate"][:HO1]; f.index = f.index.floor("s")
trail = f.rolling(9).mean().shift(1); med = f.rolling(1095, min_periods=270).median().shift(1)
mm = (f.index >= HO0); pred, act = (trail > med)[mm], (f > med)[mm]
OUT["6_funding_level_forecast"] = {"prints": int(mm.sum()), "accuracy": round(float((pred == act).mean()), 3), "ci95": boot((pred == act).astype(float).values, np.mean, blk=30),
                                   "share_above_actual": round(float(act.mean()), 3), "always_majority_accuracy": round(float(max(act.mean(), 1 - act.mean())), 3),
                                   "funding_annualized": round(float(f[mm].mean() * 1095), 4), "share_positive": round(float((f[mm] > 0).mean()), 3)}
spot = pd.read_csv("data/btc_binance_1d.csv", index_col=0, parse_dates=True)["close"][:HO1]
K = pd.read_parquet("data/perp_1d.parquet"); Fq = pd.read_parquet("data/perp_funding.parquet")
perp = K[K.sym == "BTC"].set_index("date")["close"]; fd = Fq[Fq.sym == "BTC"].set_index("date")["funding"]
c = pd.DataFrame({"s": spot, "p": perp, "f": fd}).dropna(); rc = ((c.s.pct_change() - c.p.pct_change() + c.f) / 1.5)[HO0:]; eq = (1 + rc).cumprod()
OUT["7_btc_carry"] = {"days": len(rc), "annual_return": round(float(eq.iloc[-1] ** (ANN / len(rc)) - 1), 4), "annual_vol": round(float(rc.std() * np.sqrt(ANN)), 4),
                      "max_dd": round(float((eq / eq.cummax() - 1).min()), 4), "by_year": {int(yy): round(float((1 + rc[rc.index.year == yy]).prod() - 1), 4) for yy in (2025, 2026)}}

# ---------- 8, 9, 10: ширина
txt = run_patched("src/exp16_breadth.py", [('"2020-07-01", "2024-12-31"', f'"{HO0}", "{HO1}"')])
OUT["8_9_10_breadth"] = json.load(open("results/holdout/exp16_breadth.json")); OUT["8_9_10_breadth"].pop("correlations", None)

# ---------- 11: «смыв»
import lib
lib.END = HO1
import exp7_positioning_study as S
D_ = pd.concat([S.features(a) for a in ("btc", "eth", "sol")]).dropna(subset=["z", "bull", "fwd_ex"]); D_ = D_[D_.index >= HO0]
e = D_[D_.z < -1.9].dropna(subset=["dOI_rank"]); lo, hi_ = e[e.dOI_rank < 1 / 3], e[e.dOI_rank > 2 / 3]
OUT["11_flush"] = {"events": len(e), "oi_fell": {"n": len(lo), "up_rate": round(float((lo.fwd > 0).mean()), 3) if len(lo) else None, "mean_fwd_pct": round(float(lo.fwd.mean() * 100), 2) if len(lo) else None},
                   "oi_rose": {"n": len(hi_), "up_rate": round(float((hi_.fwd > 0).mean()), 3) if len(hi_) else None, "mean_fwd_pct": round(float(hi_.fwd.mean() * 100), 2) if len(hi_) else None}}

json.dump(OUT, open("results/holdout/holdout.json", "w"), indent=1, default=float, ensure_ascii=False)
print(json.dumps(OUT, indent=1, default=float, ensure_ascii=False))
