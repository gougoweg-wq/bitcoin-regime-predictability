"""Inference tools for serially dependent returns and for selection among many backtests."""
import numpy as np
from scipy.stats import kurtosis, norm, skew


def newey_west_t(x, lags=5):
    """t-statistic of the mean with a Newey-West (Bartlett kernel) variance."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    e = x - x.mean()
    var = e @ e / n
    for j in range(1, lags + 1):
        var += 2 * (1 - j / (lags + 1)) * (e[j:] @ e[:-j]) / n
    return float(x.mean() / np.sqrt(var / n))


def block_bootstrap_ci(x, statistic=np.mean, block=20, reps=2000, level=0.95, seed=42):
    """Percentile confidence interval from a circular block bootstrap."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    n = len(x)
    n_blocks = int(np.ceil(n / block))
    draws = []
    for _ in range(reps):
        starts = rng.integers(0, n, n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]).ravel()[:n] % n
        draws.append(statistic(x[idx]))
    lo, hi = np.percentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(lo), float(hi)


def probabilistic_sharpe(returns, benchmark=0.0):
    """P(true Sharpe > benchmark), adjusting for skewness and kurtosis.

    `benchmark` is a per-period Sharpe ratio. Bailey and Lopez de Prado (2014).
    """
    r = np.asarray(returns, dtype=float)
    sr = r.mean() / r.std(ddof=1)
    denom = np.sqrt(1 - skew(r) * sr + (kurtosis(r, fisher=False) - 1) / 4 * sr**2)
    return float(norm.cdf((sr - benchmark) * np.sqrt(len(r) - 1) / denom))


def expected_max_sharpe(trial_sharpes):
    """Expected maximum Sharpe ratio among N trials with no true skill."""
    trials = np.asarray(trial_sharpes, dtype=float)
    n = len(trials)
    euler = 0.5772156649
    return float(trials.std(ddof=1) * ((1 - euler) * norm.ppf(1 - 1 / n) + euler * norm.ppf(1 - 1 / (n * np.e))))


def deflated_sharpe(returns, trial_sharpes):
    """Probabilistic Sharpe ratio against the best result expected by chance.

    `trial_sharpes` are per-period Sharpe ratios of every configuration tried.
    """
    return probabilistic_sharpe(returns, expected_max_sharpe(trial_sharpes))
