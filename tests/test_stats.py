import numpy as np
import pytest

from btcregime import block_bootstrap_ci, deflated_sharpe, newey_west_t, probabilistic_sharpe


def test_newey_west_reduces_to_ordinary_t_without_lags():
    x = np.random.default_rng(1).normal(0.1, 1, 500)
    ordinary = x.mean() / (x.std(ddof=0) / np.sqrt(len(x)))
    assert newey_west_t(x, lags=0) == pytest.approx(ordinary)


def test_newey_west_is_more_cautious_for_persistent_series():
    rng = np.random.default_rng(2)
    shocks = rng.normal(0, 1, 2000)
    persistent = np.empty(2000); persistent[0] = shocks[0]
    for i in range(1, 2000):
        persistent[i] = 0.8 * persistent[i - 1] + shocks[i]
    persistent += 0.3
    assert abs(newey_west_t(persistent, lags=20)) < abs(newey_west_t(persistent, lags=0))


def test_bootstrap_interval_contains_the_sample_mean():
    x = np.random.default_rng(3).normal(0.5, 1, 1000)
    lo, hi = block_bootstrap_ci(x)
    assert lo < x.mean() < hi


def test_deflated_sharpe_never_exceeds_the_plain_probability():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.001, 0.02, 1500)
    trials = rng.normal(0, 0.03, 200)
    assert deflated_sharpe(returns, trials) <= probabilistic_sharpe(returns)
