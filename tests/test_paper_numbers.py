"""Headline numbers of the paper, recomputed from the committed data."""
import numpy as np
import pytest
from scipy.stats import binom

from btcregime import high_volatility_forecast, market_types, newey_west_t, trend_regime
from btcregime.data import load_close


def test_regime_counts_and_look_ahead_trap(root):
    close = load_close(f"{root}/data/btc_yahoo_1d.csv", "2017-01-01", "2024-12-31")
    bull = trend_regime(close).dropna().astype(bool)
    assert (int(bull.sum()), int((~bull).sum())) == (1591, 1282)
    returns = np.log(close).diff()
    same_day, next_day = returns[bull.index], returns.shift(-1)[bull.index]
    assert same_day[bull].mean() == pytest.approx(0.00711, abs=1e-5)       # first edition: inflated
    assert next_day[bull].mean() == pytest.approx(0.00335, abs=1e-5)       # corrected
    assert next_day[~bull].mean() == pytest.approx(-0.00065, abs=1e-5)


def test_original_momentum_filter_is_57_not_60(root):
    close = load_close(f"{root}/data/btc_yahoo_1d.csv", "2017-01-01", "2024-12-31")
    log_price = np.log(close)
    z = (close - close.rolling(20).mean()) / close.rolling(20).std()
    forward = log_price.shift(-5) - log_price
    events = (z.abs() > 2.4) & forward.notna()
    hits = int((np.sign(z[events]) == np.sign(forward[events])).sum())
    assert (hits, int(events.sum())) == (76, 133)
    assert binom.sf(hits - 1, 133, 0.5) == pytest.approx(0.059, abs=0.001)
    assert (forward[events] > 0).mean() == pytest.approx(0.669, abs=0.001)  # always predicting a rise does better


def test_bull_mid_drift_in_development_and_holdout(yahoo_long):
    types = market_types(yahoo_long)
    dev = types[:"2024-12-31"]
    holdout = types["2025-01-01":]
    dev_cell = dev[dev["type"] == "Bull / mid"]["r_next"]
    ho_cell = holdout[holdout["type"] == "Bull / mid"]["r_next"]
    assert (len(dev_cell), int((dev_cell > 0).sum())) == (645, 363)
    assert newey_west_t(dev_cell.values) == pytest.approx(3.48, abs=0.02)
    assert (len(ho_cell), int((ho_cell > 0).sum())) == (95, 53)
    assert newey_west_t(ho_cell.values) == pytest.approx(2.72, abs=0.02)


@pytest.mark.parametrize("start,end,expected", [("2018-01-01", "2024-12-31", 0.646), ("2025-01-01", "2026-10-31", 0.627)])
def test_five_day_volatility_forecast_accuracy(yahoo_long, start, end, expected):
    close = yahoo_long[:end]
    returns = np.log(close).diff()
    future = returns.rolling(5).std().shift(-5)
    threshold = returns.rolling(5).std().rolling(365, min_periods=180).median()
    signal = high_volatility_forecast(close).astype("boolean")
    valid = future.notna() & signal.notna()
    hit = (signal == (future > threshold))[valid][start:]
    assert float(hit.mean()) == pytest.approx(expected, abs=0.002)
