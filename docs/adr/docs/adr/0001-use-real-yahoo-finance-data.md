# 1. Use real Yahoo Finance data instead of synthetic simulation

## Status
Accepted

## Context
The original CryptoPulse README described synthetic, simulated market data
(Pandas/NumPy), but the project's resume description claimed real BTC-USD
data from Yahoo Finance. The two didn't match, and only one of them is
defensible in a technical interview.

## Decision
Fetch real daily OHLCV data for BTC-USD via the `yfinance` library
(`src/cryptopulse/data/fetch.py`). All network I/O is isolated behind a
single function so the rest of the pipeline can be tested without a live
API dependency.

## Consequences
- The pipeline now depends on Yahoo Finance's availability and data
  quality; `fetch_ohlcv` raises `ValueError` on empty responses or
  unexpected schema rather than silently proceeding with bad data.
- Tests mock `yf.download` entirely (`tests/test_fetch.py`), so the suite
  runs offline and deterministically. The live API is only exercised when
  running `make pipeline` manually.
- Real market data is non-stationary and noisy compared to a synthetic
  simulation; this directly motivated ADR 0002 (predicting direction
  rather than price level).
