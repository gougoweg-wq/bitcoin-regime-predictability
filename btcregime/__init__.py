"""btcregime: regime-based statistical modelling of Bitcoin returns.

Every function that produces a forecast uses only data available at the close of
day t. This property is enforced by tests/test_no_lookahead.py.
"""
from .regime import trend_regime, volatility_state, market_types, MARKET_TYPES
from .bayes import posterior_update, sequential_posterior
from .volatility import ewma_volatility, high_volatility_forecast
from .stats import newey_west_t, block_bootstrap_ci, probabilistic_sharpe, expected_max_sharpe, deflated_sharpe

__version__ = "1.0.0"
