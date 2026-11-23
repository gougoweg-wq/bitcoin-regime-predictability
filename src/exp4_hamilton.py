"""Усиление 4: честный байесовский фильтр режима вместо «Master formula».
Параметры переоцениваются 1 января каждого года по всем прошлым данным (минимум 500 дней),
внутри года фильтр идёт с фиксированными параметрами. Позиция на день t+1 использует только данные до t.
Варианты: A — только фильтр (long, если P(bull завтра) > 0.5); B — SMA50 И фильтр."""
import json
import numpy as np
import pandas as pd
from lib import *
import hmm

_cache = {}


def hmm_prob(p):
    """P(bull в день t+1 | данные до t), индекс = день t."""
    key = (p.index[0], len(p), float(p.iloc[-1]))
    if key in _cache:
        return _cache[key]
    r = np.log(p).diff().dropna(); lab = regime_pos(p).shift(1).reindex(r.index)
    out = pd.Series(np.nan, index=p.index); params = {}
    for y in sorted(set(r.index.year)):
        tr = r[r.index.year < y]; tl = lab[tr.index]; ok = tl.notna()
        if ok.sum() < 500:
            continue
        th = hmm.fit(tr[ok].values, tl[ok].values)
        upto = r[r.index.year <= y]
        _, pred = hmm.hamilton(upto.values, th)
        out[upto.index[upto.index.year == y]] = pred[1:][upto.index.year == y]
        mu, sig, p00, p11, nu = hmm._unpack(th)
        params[int(y)] = {"mu_bear": float(mu[0]), "mu_bull": float(mu[1]), "sd_bear": float(sig[0]), "sd_bull": float(sig[1]),
                          "p_stay_bear": float(p00), "p_stay_bull": float(p11), "nu": float(nu)}
    _cache[key] = out; _cache[("params",) + key] = params
    return out


def pos_A(p):
    q = hmm_prob(p); return (q > 0.5).astype(float).where(q.notna())


def pos_B(p):
    return pos_A(p) * regime_pos(p)


def base(p):                                   # baseline на том же периоде, где есть фильтр
    return regime_pos(p).where(hmm_prob(p).notna())


if __name__ == "__main__":
    ab("exp4A_hamilton_only", pos_A, base)
    ab("exp4B_sma_and_hamilton", pos_B, base)
    pr = {a: _cache[("params", load(a).index[0], len(load(a)), float(load(a).iloc[-1]))] for a in ASSETS}
    json.dump(pr, open("results/phase4/exp4_params.json", "w"), indent=1)
    print(json.dumps(pr["btc"], indent=1))
