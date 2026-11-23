"""Усиление 2: гистерезис на границе режима.
Вход, когда ln(P/SMA50) > +k*sigma_hat; выход, когда < -k*sigma_hat; внутри полосы держим прежнее состояние.
k = 0.5 задан заранее (половина дневной сигмы)."""
import numpy as np
import pandas as pd
from lib import *


def hysteresis_pos(p, k=0.5, w=50):
    sma = p.rolling(w).mean(); dev = np.log(p / sma); band = k * ewma_vol(p)
    state = pd.Series(np.nan, index=p.index)
    state[dev > band] = 1.0
    state[dev < -band] = 0.0
    return state.ffill().where(sma.notna() & band.notna())


if __name__ == "__main__":
    ab("exp2_hysteresis", hysteresis_pos, regime_pos)
    for k in (0.25, 1.0):   # чувствительность, в решение не входит
        ab(f"exp2_sens_k{k}", lambda p, k=k: hysteresis_pos(p, k), regime_pos)
