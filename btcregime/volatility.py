"""A parameter-free forecast of the volatility regime."""
import numpy as np


def ewma_volatility(close, halflife=10, min_periods=30):
    """Exponentially weighted daily volatility using returns up to day t."""
    returns = np.log(close).diff()
    return np.sqrt((returns**2).ewm(halflife=halflife, min_periods=min_periods).mean())


def high_volatility_forecast(close, lookback=365, min_periods=180):
    """True where current volatility is above its own trailing-year median.

    Read as: "the coming days will be more volatile than usual".
    """
    vol = ewma_volatility(close)
    return (vol > vol.rolling(lookback, min_periods=min_periods).median()).where(vol.notna())
