"""Усиление 1: таргетирование волатильности поверх режима.
w_t = regime_t * min(1, sigma*_t / sigma_hat_t); sigma*_t = медиана sigma_hat по прошлому (расширяющееся окно).
Параметры заданы заранее: halflife 20 дней, плечо не выше 1."""
import sys
from lib import *


def voltarget_pos(p, halflife=20):
    s = ewma_vol(p, halflife)
    return regime_pos(p) * (s.expanding(60).median() / s).clip(upper=1.0)


if __name__ == "__main__":
    ab("exp1_voltarget", voltarget_pos, regime_pos)
    for hl in (10, 40):   # чувствительность, в решение не входит
        ab(f"exp1_sens_hl{hl}", lambda p, hl=hl: voltarget_pos(p, hl), regime_pos)
