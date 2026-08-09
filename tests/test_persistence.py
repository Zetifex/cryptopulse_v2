"""Tests for cryptopulse.data.persistence."""

from __future__ import annotations

import pandas as pd

from cryptopulse.data.persistence import load_features, save_features


def test_save_and_load_round_trip(tmp_path):
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame({"short_ma": [1.0, 2.0, 3.0], "target_up": [1, 0, 1]}, index=dates)
    df.index.name = "date"

    db_path = str(tmp_path / "test.db")
    save_features(df, db_path, table="market_features")
    loaded = load_features(db_path, table="market_features")

    pd.testing.assert_frame_equal(df, loaded, check_dtype=False, check_freq=False)


def test_save_replace_existing_data(tmp_path):
    dates = pd.date_range("2024-01-01", periods=2, freq="D")
    df_v1 = pd.DataFrame({"target_up": [1, 0]}, index=dates)
    df_v1.index.name = "date"
    df_v2 = pd.DataFrame(
        {"target_up": [0, 1, 1]},
        index=pd.date_range("2024-02-01", periods=3, freq="D"),
    )
    df_v2.index.name = "date"

    db_path = str(tmp_path / "test.db")

    db_path = str(tmp_path / "test.db")
    save_features(df_v1, db_path, table="market_features")
    save_features(df_v2, db_path, table="market_features")
    loaded = load_features(db_path, table="market_features")

    assert len(loaded) == 3
