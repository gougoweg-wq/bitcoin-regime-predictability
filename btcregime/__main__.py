"""Command line: `python -m btcregime status` and `python -m btcregime forecast`."""
import argparse
import os

import pandas as pd

from .bayes import sequential_posterior
from .data import fetch_binance_daily, load_close
from .regime import market_types
from .volatility import high_volatility_forecast


def snapshot(close):
    """Forecast made at the latest close, for the following day."""
    types = market_types(close, keep_last=True)
    post = sequential_posterior(types).iloc[-1]
    return {
        "as_of_close": types.index[-1].date().isoformat(),
        "forecast_for": (types.index[-1] + pd.Timedelta(days=1)).date().isoformat(),
        "close": round(float(close.iloc[-1]), 2),
        "market_type": post["type"],
        "p_up": round(float(post["p_up"]), 4),
        "mu_hat_pct": round(float(post["mu_hat"]) * 100, 4),
        "p_mu_pos": round(float(post["p_mu_pos"]), 4),
        "high_vol_forecast": bool(high_volatility_forecast(close).iloc[-1]),
    }


def main():
    parser = argparse.ArgumentParser(prog="btcregime", description="Regime-based forecast for Bitcoin.")
    parser.add_argument("command", choices=["status", "forecast"])
    parser.add_argument("--csv", help="read daily closes from this CSV instead of fetching from Binance")
    parser.add_argument("--append", default="forecasts/forward_log.csv", help="log file for the `forecast` command")
    args = parser.parse_args()

    close = load_close(args.csv) if args.csv else fetch_binance_daily()["close"]
    snap = snapshot(close)

    if args.command == "status":
        print(f"As of close {snap['as_of_close']}  (BTC {snap['close']:,.2f})")
        print(f"  Market type            {snap['market_type']}")
        print(f"  P(rise on {snap['forecast_for']})   {snap['p_up']:.1%}")
        print(f"  Expected return        {snap['mu_hat_pct']:+.2f}% per day")
        print(f"  P(drift > 0)           {snap['p_mu_pos']:.3f}")
        print(f"  Volatility outlook     {'above' if snap['high_vol_forecast'] else 'below'} its one-year median")
        print("Research output, not investment advice.")
        return

    log = pd.read_csv(args.append) if os.path.exists(args.append) else pd.DataFrame()
    if len(log) and snap["as_of_close"] in set(log["as_of_close"]):
        print("already logged", snap["as_of_close"])
        return
    snap["logged_at_utc"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M")
    os.makedirs(os.path.dirname(args.append) or ".", exist_ok=True)
    pd.concat([log, pd.DataFrame([snap])]).to_csv(args.append, index=False)
    print("logged", snap)


if __name__ == "__main__":
    main()
