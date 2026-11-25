"""Observable market regimes: trend (Bull/Bear) and volatility state (low/mid/high)."""
import numpy as np
import pandas as pd

MARKET_TYPES = ["Bull / low", "Bull / mid", "Bull / high", "Bear / low", "Bear / mid", "Bear / high"]


def trend_regime(close, window=50):
    """True where the close is above its `window`-day simple moving average.

    The average includes the current close, so the label is known at the close of
    day t and must only ever be paired with returns from day t+1 onward.
    """
    sma = close.rolling(window).mean()
    return (close > sma).where(sma.notna())


def volatility_state(close, vol_window=30, rank_window=365, min_periods=180):
    """Tercile of realised volatility relative to its own trailing year.

    Returns (rank, state): rank in (0, 1], state in {"low", "mid", "high"}.
    """
    returns = np.log(close).diff()
    realised = returns.rolling(vol_window).std()
    rank = realised.rolling(rank_window, min_periods=min_periods).rank(pct=True)
    state = pd.cut(rank, [0, 1 / 3, 2 / 3, 1.0001], labels=["low", "mid", "high"], right=False)
    return rank, state


def market_types(close, keep_last=False):
    """One row per day with the observable market type and the next day's log return.

    Columns: bull (bool), vol (low/mid/high), type (one of MARKET_TYPES), r_next.
    `r_next` is the outcome, not a feature: it is the return of day t+1.
    With keep_last=True the final day (whose outcome is not yet known) is retained.
    """
    log_price = np.log(close)
    returns = log_price.diff()
    sma = close.rolling(50).mean()
    realised = returns.rolling(30).std()
    rank, state = volatility_state(close)
    frame = pd.DataFrame({
        "p": close,
        "r_next": returns.shift(-1),
        "bull": trend_regime(close),
        "T": np.log(close / sma) / (realised * np.sqrt(50)),
        "V": rank,
    })
    frame["vol"] = state
    required = ["p", "bull", "T", "V"] + ([] if keep_last else ["r_next"])
    frame = frame.dropna(subset=required)
    if keep_last:
        frame = frame[frame["r_next"].notna() | (frame.index == frame.index[-1])]
    trend = np.where(frame["bull"].astype(bool), "Bull", "Bear")
    frame["type"] = pd.Series(trend, index=frame.index) + " / " + frame["vol"].astype(str)
    return frame
