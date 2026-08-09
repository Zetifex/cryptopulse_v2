"""Market data ingestion from Yahoo Finance.

This module isolates all network I/O behind one function so the rest of
the pipeline never talks to yfinance directly. That boundary is what lets
us unit-test everything downstream with mocked data instead of hitting a
live API in CI.
"""

from __future__ import annotations

import logging
from typing import cast

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_ohlcv(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Fetch daily OHLCV data for a ticker from Yahoo Finance.

    Args:
        ticker: Ticker symbol, e.g. "BTC-USD".
        start: Start date as "YYYY-MM-DD".
        end: Optional end date as "YYYY-MM-DD". Defaults to today when None.

    Returns:
        DataFrame indexed by date with columns Open, High, Low, Close, Volume,
        sorted chronologically ascending.

    Raises:
        ValueError: If no data is returned for the given ticker/date range.
    """
    logger.info("Fetching %s from %s to %s", ticker, start, end or "today")
    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker={ticker!r}, start={start!r}, "
            f"end={end!r}. Check the ticker symbol and date range."
        )

    # Recent yfinance versions return MultiIndex columns (ticker, field)
    # even for a single-ticker download. Flatten to a plain column index.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.title)
    missing = set(REQUIRED_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing expected columns: {missing}")

    # yfinance has no type stubs, so `raw` is typed Any from here on --
    #  cast() draws an explicit line: everything past this point is our
    # typed contract, not yfinance's untyped output leaking through.
    return cast(pd.DataFrame, raw[REQUIRED_COLUMNS].sort_index())
