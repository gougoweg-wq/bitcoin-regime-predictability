"""Цифры для примеров и графики для переписанной работы. Все значения считаются из тех же данных, что и эксперименты."""
import json, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom, norm
sys.path.insert(0, "src")
from exp10_map import state
from exp10_bayes import posterior

BLUE, ORANGE, INK, MUTED, GRID = "#0072B2", "#D55E00", "#1a1a1a", "#666666", "#e6e6e6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.frameon": False})
DEV_END, HO0, HO1 = "2024-12-31", "2025-01-01", "2026-10-31"
N = {}

d = state("data/btc_yahoo_1d_long.csv", "2014-09-17", HO1)
d["s"] = np.where(d.bull.astype(bool), "Bull", "Bear") + pd.Series(" / ", index=d.index) + d.vol.astype(str)
dev, ho = d[:DEV_END], d[HO0:]
ORDER = ["Bull / low", "Bull / mid", "Bull / high", "Bear / low", "Bear / mid", "Bear / high"]

# --- цифры для разобранного примера (клетка Bull / mid на конец 2024)
c = dev[dev.s == "Bull / mid"]; n_s, k_s, xbar = len(c), int((c.r_next > 0).sum()), float(c.r_next.mean()); sd_s = float(c.r_next.std())
mu0, p0 = float(dev.r_next.mean()), float((dev.r_next > 0).mean()); n0 = 250
mu_hat = (n_s * xbar + n0 * mu0) / (n_s + n0); se = sd_s / np.sqrt(n_s + n0); p_up = (k_s + n0 * p0) / (n_s + n0)
N["worked_example_bull_mid_end_2024"] = {"n_s": n_s, "k_s": k_s, "xbar": xbar, "sd_s": sd_s, "mu0": mu0, "p0": p0, "n0": n0, "mu_hat": mu_hat, "se": se,
                                         "z": mu_hat / se, "P_mu_pos": float(norm.cdf(mu_hat / se)), "p_up": p_up, "days_dev_total": len(dev)}
h = ho[ho.s == "Bull / mid"]; kh, nh = int((h.r_next > 0).sum()), len(h)
N["holdout_bull_mid"] = {"n": nh, "k": kh, "up_rate": kh / nh, "binom_p_one_sided": float(binom.sf(kh - 1, nh, 0.5)), "mean": float(h.r_next.mean()), "sd": float(h.r_next.std()),
                         "t_plain": float(h.r_next.mean() / (h.r_next.std() / np.sqrt(nh)))}
N["cells_dev"] = {s: {"days": int((dev.s == s).sum()), "up": float((dev[dev.s == s].r_next > 0).mean()), "mean_pct": float(dev[dev.s == s].r_next.mean() * 100)} for s in ORDER}
N["cells_holdout"] = {s: {"days": int((ho.s == s).sum()), "up": float((ho[ho.s == s].r_next > 0).mean()), "mean_pct": float(ho[ho.s == s].r_next.mean() * 100)} for s in ORDER}
N["base_up_dev"], N["base_up_ho"] = float((dev.r_next > 0).mean()), float((ho.r_next > 0).mean())

# --- Рис. 1: цена и режим
p = pd.read_csv("data/btc_yahoo_1d_long.csv", index_col=0, parse_dates=True)["close"]["2017-01-01":HO1]; sma = p.rolling(50).mean()
fig, ax = plt.subplots(figsize=(8, 3.6)); ax.semilogy(p.index, p, color=INK, lw=1.0, label="BTC close")
ax.semilogy(sma.index, sma, color=ORANGE, lw=1.4, label="50-day SMA")
bull = (p > sma).fillna(False).values
ax.fill_between(p.index, p.min() * 0.8, p.max() * 1.3, where=bull, color=BLUE, alpha=0.10, lw=0, label="Bull regime (P > SMA50)")
ax.axvline(pd.Timestamp(HO0), color=MUTED, lw=1, ls="--"); ax.text(pd.Timestamp(HO0), p.max() * 1.15, " hold-out", color=MUTED, fontsize=9, va="top")
ax.set_ylabel("Price, USD (log scale)"); ax.set_ylim(p.min() * 0.8, p.max() * 1.3); ax.legend(loc="lower right", fontsize=9)
ax.set_title("Figure 1. Bitcoin price and the regime indicator, 2017–2026", loc="left", fontsize=11, color=INK); fig.savefig("paper/figures/fig1_regime.png"); plt.close(fig)

# --- Рис. 2: средняя доходность следующего дня по шести типам рынка
fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(6); w = 0.38
a = [N["cells_dev"][s]["mean_pct"] for s in ORDER]; b = [N["cells_holdout"][s]["mean_pct"] for s in ORDER]
ax.bar(x - w / 2 - 0.01, a, w, color=BLUE, label="Development, 2015–2024"); ax.bar(x + w / 2 + 0.01, b, w, color=ORANGE, label="Hold-out, 2025–2026")
ax.axhline(0, color=MUTED, lw=1); ax.set_xticks(x); ax.set_xticklabels(ORDER); ax.set_ylabel("Mean next-day log return, %"); ax.legend(fontsize=9, loc="upper right")
for xi, v in zip(x - w / 2 - 0.01, a): ax.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
for xi, v in zip(x + w / 2 + 0.01, b): ax.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=INK)
ax.set_ylim(-0.52, 0.58); ax.set_title("Figure 2. Mean next-day return by market type", loc="left", fontsize=11, color=INK); fig.savefig("paper/figures/fig2_cells.png"); plt.close(fig)

# --- Рис. 3: точность прогноза размаха
dv = json.load(open("results/phase4/exp12_vol_predict.json")); hv = json.load(open("results/holdout/exp12_vol_predict.json")); hs = [1, 5, 10, 20]
fig, ax = plt.subplots(figsize=(6.5, 3.4))
for src, col, lab, off in ((dv, BLUE, "Development, 2018–2024", -0.12), (hv, ORANGE, "Hold-out, 2025–2026", 0.12)):
    acc = [src[f"h{k}"]["accuracy"] * 100 for k in hs]; lo = [src[f"h{k}"]["ci95"][0] * 100 for k in hs]; hi = [src[f"h{k}"]["ci95"][1] * 100 for k in hs]
    xs = np.arange(4) + off; ax.errorbar(xs, acc, yerr=[np.array(acc) - lo, np.array(hi) - acc], fmt="o", color=col, ms=6, lw=1.6, capsize=3, label=lab)
    for xi, v in zip(xs, acc): ax.text(xi + (-0.07 if off < 0 else 0.07), v, f"{v:.1f}", ha="right" if off < 0 else "left", va="center", fontsize=8, color=INK)
ax.set_xlim(-0.65, 3.6); ax.axhline(50, color=MUTED, lw=1, ls="--"); ax.text(3.55, 50.5, "coin flip", color=MUTED, fontsize=8, ha="right")
ax.set_xticks(range(4)); ax.set_xticklabels([f"{k} day" + ("s" if k > 1 else "") for k in hs]); ax.set_xlabel("Forecast horizon"); ax.set_ylabel("Accuracy, % (95% CI)"); ax.set_ylim(45, 80); ax.legend(fontsize=9, loc="upper left")
ax.set_title("Figure 3. Accuracy of the volatility-regime forecast", loc="left", fontsize=11, color=INK); fig.savefig("paper/figures/fig3_volatility.png"); plt.close(fig)

# --- Рис. 4: кривые капитала (Binance, 10 б.п.)
q = pd.read_csv("data/btc_binance_1d.csv", index_col=0, parse_dates=True)["close"][:HO1]; b_ = (q > q.rolling(50).mean()).astype(float); held = b_.shift(1)
reg = (held * q.pct_change() - held.diff().abs().fillna(0) * 10 / 1e4)["2017-10-06":]; bh = q.pct_change()["2017-10-06":]
fig, ax = plt.subplots(figsize=(8, 3.6)); e1, e2 = (1 + reg).cumprod(), (1 + bh).cumprod()
ax.semilogy(e2.index, e2, color=MUTED, lw=1.4, label="Buy and hold"); ax.semilogy(e1.index, e1, color=BLUE, lw=1.6, label="Regime rule (long only when P > SMA50)")
ax.axvline(pd.Timestamp(HO0), color=MUTED, lw=1, ls="--"); ax.text(pd.Timestamp(HO0), ax.get_ylim()[1] * 0.9, " hold-out", color=MUTED, fontsize=9, va="top")
ax.text(e1.index[-1], e1.iloc[-1], f" ×{e1.iloc[-1]:.0f}", color=INK, fontsize=9, va="center"); ax.text(e2.index[-1], e2.iloc[-1], f" ×{e2.iloc[-1]:.0f}", color=INK, fontsize=9, va="center")
ax.set_ylabel("Growth of 1 unit (log scale)"); ax.legend(fontsize=9, loc="upper left")
ax.set_title("Figure 4. Regime rule versus buy and hold, after 0.10% costs per trade", loc="left", fontsize=11, color=INK); fig.savefig("paper/figures/fig4_equity.png"); plt.close(fig)
dd = lambda e: float((e / e.cummax() - 1).min())
N["equity_full_2017_2026"] = {"regime_final": float(e1.iloc[-1]), "buyhold_final": float(e2.iloc[-1]), "regime_maxdd": dd(e1), "buyhold_maxdd": dd(e2)}
r_all = np.log(p).diff().dropna()[:DEV_END]
N["returns_2017_2024"] = {"n": len(r_all), "share_abs_gt_10pct": float((r_all.abs() > 0.10).mean()), "count_abs_gt_10pct": int((r_all.abs() > 0.10).sum()),
                          "worst": float(r_all.min()), "worst_date": str(r_all.idxmin().date()), "best": float(r_all.max()), "best_date": str(r_all.idxmax().date())}
json.dump(N, open("paper/numbers.json", "w"), indent=1); print(json.dumps(N, indent=1))

# --- Рис. 5: вся процедура на рядах без предсказуемости
sim = pd.read_csv("results/phase4/exp17b_null_sims.csv")
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.scatter(sim.dev_max_abs_t, sim.holdout_t_same_sign, s=7, color=MUTED, alpha=0.35, lw=0, label="2,000 simulated histories with no predictability")
ax.axvline(2.64, color=MUTED, lw=1, ls="--"); ax.axhline(1.645, color=MUTED, lw=1, ls="--")
ax.scatter([3.48], [2.72], s=110, color=ORANGE, edgecolor="white", lw=1.5, zorder=5, label="Observed: Bull / mid")
ax.annotate("observed\n(3.48, 2.72)", (3.48, 2.72), xytext=(3.85, 3.25), fontsize=9, color=INK, arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
found = float((sim.dev_max_abs_t > 2.64).mean()); conf = float(((sim.dev_max_abs_t > 2.64) & (sim.holdout_t_same_sign > 1.645)).mean())
ax.text(2.68, -3.6, f"discovery threshold\n{found:.0%} of null runs pass", fontsize=8, color=MUTED, va="bottom")
ax.text(1.05, 1.75, f"replication threshold: {conf:.1%} pass both", fontsize=8, color=MUTED, va="bottom")
ax.set_xlabel("Best |t| among six market types, development"); ax.set_ylabel("t of the same type on the hold-out"); ax.set_xlim(1, 4.6); ax.set_ylim(-3.8, 3.8)
ax.legend(fontsize=8, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=1)
ax.set_title("Figure 5. Discovery and replication under the null hypothesis", loc="left", fontsize=11, color=INK, pad=38)
fig.savefig("paper/figures/fig5_null_simulation.png"); plt.close(fig)
N["null_simulation"] = {"found": found, "confirmed": conf, "as_strong": float(((sim.dev_max_abs_t >= 3.48) & (sim.holdout_t_same_sign >= 2.72)).mean())}
json.dump(N, open("paper/numbers.json", "w"), indent=1); print(N["null_simulation"])
