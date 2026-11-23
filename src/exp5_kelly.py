"""Усиление 5: дробный Келли. w_t = regime_t * clip(c * mu_bull_t / sigma_hat_t^2, 0, 1), c = 0.5 задан заранее.
mu_bull_t — средняя доходность следующего дня в режиме Bull, оценённая только по прошлому (расширяющееся окно).
Апостериор Гамильтона не используется: exp4 показал, что он не различает направление."""
import numpy as np
from lib import *


def kelly_pos(p, c=0.5):
    reg = regime_pos(p); r = p.pct_change()
    nxt_in_bull = r.where(reg.shift(1) == 1)                 # доходность дня t при Bull накануне: известна на close t
    mu = nxt_in_bull.expanding(100).mean()
    return reg * (c * mu / ewma_vol(p) ** 2).clip(0, 1)


if __name__ == "__main__":
    ab("exp5_kelly", kelly_pos, lambda p: regime_pos(p).where(kelly_pos(p).notna()))
