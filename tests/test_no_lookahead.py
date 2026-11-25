"""The central guarantee: nothing known at day t changes if the future is rewritten."""
import numpy as np
import pandas as pd

from btcregime import high_volatility_forecast, market_types, sequential_posterior


def test_features_ignore_future_prices(yahoo_long):
    close = yahoo_long.iloc[:1600]
    cut = 1200
    rng = np.random.default_rng(0)
    tampered = close.copy()
    tampered.iloc[cut + 1:] *= np.exp(np.cumsum(rng.normal(0, 0.2, len(close) - cut - 1)))   # rewrite everything after `cut`

    original, altered = market_types(close, keep_last=True), market_types(tampered, keep_last=True)
    upto = close.index[cut]
    for column in ("bull", "vol", "type", "V", "T"):
        pd.testing.assert_series_equal(original.loc[:upto, column], altered.loc[:upto, column])

    post_a, post_b = sequential_posterior(original), sequential_posterior(altered)
    pd.testing.assert_frame_equal(post_a.loc[:upto], post_b.loc[:upto])

    vol_a, vol_b = high_volatility_forecast(close), high_volatility_forecast(tampered)
    pd.testing.assert_series_equal(vol_a.loc[:upto], vol_b.loc[:upto])


def test_outcome_is_strictly_the_next_day(yahoo_long):
    close = yahoo_long.iloc[:800]
    types = market_types(close)
    log_return = np.log(close).diff()
    day = types.index[300]
    next_day = close.index[close.index.get_loc(day) + 1]
    assert np.isclose(types.loc[day, "r_next"], log_return.loc[next_day])
