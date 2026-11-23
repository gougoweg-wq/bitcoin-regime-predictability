"""Усиление 9: ансамбль режимов на дневных данных. Позиция = доля окон {10,20,50,100,200}, где P > SMA
(дробная, от 0 до 1). Окна заданы заранее. Baseline — SMA50 на том же периоде."""
import pandas as pd
from lib import *

WINS = (10, 20, 50, 100, 200)


def ensemble_pos(p):
    return pd.concat([regime_pos(p, w) for w in WINS], axis=1).mean(axis=1, skipna=False)


if __name__ == "__main__":
    ab("exp9_ensemble_daily", ensemble_pos, lambda p: regime_pos(p).where(ensemble_pos(p).notna()))
