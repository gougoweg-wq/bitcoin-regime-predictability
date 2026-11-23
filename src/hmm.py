"""Двухрежимная марковская модель доходностей (Hamilton 1989) с t-распределением Стьюдента.
Оценка: максимум правдоподобия через фильтр Гамильтона. Инициализация — из режимов по SMA50 (как в работе)."""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import t as student


def _unpack(th):
    mu = th[0:2]; sig = np.exp(th[2:4]); p00, p11 = 1 / (1 + np.exp(-th[4:6])); nu = 2.05 + np.exp(th[6])
    return mu, sig, p00, p11, nu


def hamilton(r, th):
    """Возвращает (loglik, pred), pred[t] = P(состояние 1 в день t | данные до t-1)."""
    mu, sig, p00, p11, nu = _unpack(th)
    L = student.pdf((r[:, None] - mu) / sig, nu) / sig + 1e-300
    l0, l1 = L[:, 0].tolist(), L[:, 1].tolist()
    q = (1 - p00) / (2 - p00 - p11)          # стационарная вероятность состояния 1
    ll, pred = 0.0, np.empty(len(r) + 1)
    for i in range(len(r)):
        pred[i] = q
        a, b = (1 - q) * l0[i], q * l1[i]; s = a + b
        ll += np.log(s); f = b / s
        q = f * p11 + (1 - f) * (1 - p00)
    pred[len(r)] = q                          # прогноз на следующий после выборки день
    return ll, pred


def fit(r, bull_label):
    """r — лог-доходности, bull_label — метка режима по SMA накануне (для стартовой точки). Состояние 1 = bull."""
    r0, r1 = r[bull_label == 0], r[bull_label == 1]
    th0 = np.array([r0.mean(), r1.mean(), np.log(r0.std()), np.log(r1.std()), 3.0, 3.0, np.log(3.0)])
    res = minimize(lambda th: -hamilton(r, th)[0], th0, method="L-BFGS-B")
    th = res.x
    if th[0] > th[1]:                         # состояние 1 всегда то, где средняя доходность выше
        th = th[[1, 0, 3, 2, 5, 4, 6]]
    return th
