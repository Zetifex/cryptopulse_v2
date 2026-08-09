"""Tests for cryptopulse.data.fetch.

We mock yf.download in every test here -- tests must never depend on a
live external API. That would make CI flaky, slow, and rate-limited, and
it's exactly the kind of dependency a production test suite avoids.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from cryptopulse.data.fetch import fetch_ohlcv


def _flat_ohlcv(n: int = 5) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame(
        {
            "Open": range(n),
            "High": range(n),
            "Low": range(n),
            "Close": range(n),
            "Volume": range(n),
        },
        index=dates,
    )


def test_fetch_ohlcv_returns_expected_columns():
    with patch("cryptopulse.data.fetch.yf.download", return_value=_flat_ohlcv()):
        result = fetch_ohlcv("BTC-USD", "2024-01-01")

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert len(result) == 5


def test_fetch_ohlcv_flattens_multiindex_columns():
    flat = _flat_ohlcv()
    multi = flat.copy()
    multi.columns = pd.MultiIndex.from_product([flat.columns, ["BTC-USD"]])

    with patch("cryptopulse.data.fetch.yf.download", return_value=multi):
        result = fetch_ohlcv("BTC-USD", "2024-01-01")

    assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]


def test_fetch_ohlcv_raises_on_empty_response():
    with patch("cryptopulse.data.fetch.yf.download", return_value=pd.DataFrame()):
        with pytest.raises(ValueError, match="No data returned"):
            fetch_ohlcv("NOT-A-REAL-TICKER", "2024-01-01")


def test_fetch_ohlcv_raises_on_missing_columns():
    incomplete = _flat_ohlcv().drop(columns=["Volume"])
    with patch("cryptopulse.data.fetch.yf.download", return_value=incomplete):
        with pytest.raises(ValueError, match="missing expected columns"):
            fetch_ohlcv("BTC-USD", "2024-01-01")
