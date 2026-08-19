"""Tests for cryptopulse.training.split.

The leakage test here mirrors the one in test_features.py, but targets a
different mechanism: not the rolling windows, but the scaler statistics.
Mutating test-period values must not change a single scaled training value.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryptopulse.data.features import FEATURE_COLUMNS
from cryptopulse.training.split import chronological_split, scale_splits


def _make_feature_df(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    data = {col: rng.normal(loc=100.0, scale=10.0, size=n) for col in FEATURE_COLUMNS}
    data["target_up"] = rng.integers(0, 2, size=n)
    return pd.DataFrame(data, index=dates)


def test_split_size_match_fractions():
    df = _make_feature_df(100)
    train, val, test = chronological_split(df, train_frac=0.70, val_frac=0.15)

    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    assert len(train) + len(val) + len(test) == len(df)


def test_split_preserves_all_rows():
    df = _make_feature_df(97)  # deliberately not divisible by the fractions
    train, val, test = chronological_split(df)

    rejoined = pd.concat([train, val, test])
    pd.testing.assert_frame_equal(df, rejoined)


def test_split_rejects_unsorted_index():
    df = _make_feature_df(100).iloc[::-1]  # reverse chronological
    with pytest.raises(ValueError, match="date-sorted index"):
        chronological_split(df)


def test_split_rejects_invalid_fractions():
    df = _make_feature_df(5)
    with pytest.raises(ValueError, match="empty partition"):
        chronological_split(df, train_frac=0.95, val_frac=0.04)


def test_scaler_fitted_on_train_only():
    """Train features standardize to mean 0 / std 1; val and test do not."""
    df = _make_feature_df(200)
    train, val, test = chronological_split(df)
    result = scale_splits(train, val, test)

    train_means = result.train[FEATURE_COLUMNS].mean()
    np.testing.assert_allclose(train_means, 0.0, atol=1e-10)

    train_stds = result.train[FEATURE_COLUMNS].std(ddof=0)
    np.testing.assert_allclose(train_stds, 1.0, atol=1e-10)


def test_scaling_does_not_touch_target_column():
    df = _make_feature_df(200)
    train, val, test = chronological_split(df)
    result = scale_splits(train, val, test)

    pd.testing.assert_series_equal(train["target_up"], result.train["target_up"])
    pd.testing.assert_series_equal(test["target_up"], result.test["target_up"])


def test_no_leakage_test_values_do_not_affect_scaled_training_data():
    """Changing the test partition must not alter any scaled training value."""
    df_a = _make_feature_df(200)

    df_b = df_a.copy()
    df_b.iloc[170:, :4] = df_b.iloc[170:, :4] + 5000.0  # distort test period only

    result_a = scale_splits(*chronological_split(df_a))
    result_b = scale_splits(*chronological_split(df_b))

    pd.testing.assert_frame_equal(
        result_a.train[FEATURE_COLUMNS],
        result_b.train[FEATURE_COLUMNS],
    )


def test_scaler_is_returned_for_reuse_at_inference():
    df = _make_feature_df(200)
    result = scale_splits(*chronological_split(df))

    assert hasattr(result.scaler, "mean_")
    assert len(result.scaler.mean_) == len(FEATURE_COLUMNS)

def test_split_partitions_are_chronological():
    df = _make_feature_df(100)
    train, val, test = chronological_split(df)

    assert train.index.max() < val.index.min()
    assert val.index.max() < test.index.min()


def test_split_rejects_empty_partition():
    df = _make_feature_df(5)
    with pytest.raises(ValueError, match="empty partition"):
        chronological_split(df, train_frac=0.95, val_frac=0.04)