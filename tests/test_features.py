"""Tests for cryptopulse.data.features.

The leakage test below is the important one: it doesn't just eyeball the
rolling-window math, it asserts a property -- changing a future price
must not change any feature computed at an earlier row. That's the kind
of test a resume can actually point to as evidence of care.
"""

from __future__ import annotations

import pandas as pd
import pytest

from cryptopulse.data.features import FEATURE_COLUMNS, engineer_features


def _make_price_df(closes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({"Close": closes}, index=dates)


def test_target_up_matches_expected_direction():
    # 8 points gives room for long_window=3 to warm up (drops first 2 rows)
    # before we start checking direction, so the assertion isn't entangled
    # with the rolling-window NaN-drop behavior.
    closes = [100, 105, 103, 110, 108, 115, 112, 120]
    df = _make_price_df(closes)
    features = engineer_features(df, short_window=2, long_window=3)

    assert list(features["target_up"]) == [1, 0, 1, 0, 1]


def test_final_row_is_dropped_no_future_label():
    df = _make_price_df([100.0 + i for i in range(40)])
    features = engineer_features(df, short_window=5, long_window=10)

    assert features.index.max() < df.index.max()


def test_no_leakage_future_prices_do_not_affect_past_features():
    """Changing prices after a cutoff must not change features at or before it."""
    base_closes = [100 + i + (i % 5) for i in range(60)]
    df_a = _make_price_df(base_closes)

    cutoff = 40
    mutated_closes = base_closes[:cutoff] + [c + 500 for c in base_closes[cutoff:]]
    df_b = _make_price_df(mutated_closes)

    features_a = engineer_features(df_a, short_window=5, long_window=10)
    features_b = engineer_features(df_b, short_window=5, long_window=10)

    shared_dates = features_a.index[features_a.index < df_a.index[cutoff]]

    pd.testing.assert_frame_equal(
        features_a.loc[shared_dates, FEATURE_COLUMNS],
        features_b.loc[shared_dates, FEATURE_COLUMNS],
    )


def test_missing_close_column_raises():
    df = pd.DataFrame({"Open": [1, 2, 3]})
    with pytest.raises(ValueError, match="Close"):
        engineer_features(df)
