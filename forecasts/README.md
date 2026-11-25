# Forward test

`forward_log.csv` receives one row per day, written at the close of day *t* with the forecast for day *t+1*:
market type, probability of a rise, expected return, and the volatility outlook.

Each row is committed to git when it is made. Because a commit cannot be back-dated on GitHub without
rewriting public history, the log is a tamper-evident record that forecasts precede outcomes. This extends
the paper's hold-out indefinitely: every new day is out of sample.

It runs daily at 00:20 UTC (`.github/workflows/forward-test.yml`) and can also be started by hand from the
Actions tab (`forward-test` → Run workflow). Locally:

```bash
python -m btcregime forecast --append forecasts/forward_log.csv
```

The live model is fitted on Binance BTCUSDT (from August 2017). The paper's model uses Yahoo BTC-USD from
2014, so posterior values differ slightly between the two.
