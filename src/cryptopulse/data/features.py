"""Feature engineering and label construction for CryptoPulse.

Design constraint: every feature at row t must be computable using only
data available up to and including t. The label looks one day forward.
Getting this ordering backwards is the single most common way a
time-series model silently cheats.
"""

from __future__ import annotations

import pandas as pd

TARGET_COLUMN = "target_up"
# Columns fed to the model. Deliberately excludes short_ma/long_ma: those
# are absolute price levels, which drift far outside the training
# distribution as the market trends. Their *ratio* carries the same
# crossover signal while remaining scale-invariant.
FEATURE_COLUMNS = ["ma_ratio", "price_to_long_ma", "daily_return", "volatility"]


def engineer_features(
    df: pd.DataFrame,
    short_window: int = 10,
    long_window: int = 30,
) -> pd.DataFrame:
    """Compute technical indicators and the next-day direction label.

    All indicators use trailing (non-centered) rolling windows, so nothing
    at row t depends on data from t+1 onward. The label depends on t+1, so
    the final row in the input (which has no known "tomorrow") is dropped,
    along with any leading rows that don't yet have a full rolling window.

    Args:
        df: OHLCV DataFrame indexed by date, must contain a "Close" column.
        short_window: Window size in days for the short moving average and
            the rolling volatility calculation.
        long_window: Window size in days for the long moving average.

    Returns:
        DataFrame with engineered feature columns (short_ma, long_ma,
        daily_return, volatility) and an integer "target_up" label column.
    """
    if "Close" not in df.columns:
        raise ValueError("engineer_features requires a 'Close' column")

    out = df.copy()
    out["short_ma"] = out["Close"].rolling(window=short_window).mean()
    out["long_ma"] = out["Close"].rolling(window=long_window).mean()
    out["ma_ratio"] = out["short_ma"] / out["long_ma"] - 1.0
    out["price_to_long_ma"] = out["Close"] / out["long_ma"] - 1.0
    out["daily_return"] = out["Close"].pct_change()
    out["volatility"] = out["daily_return"].rolling(window=short_window).std()

    # Label: does tomorrow's close exceed today's close? Uses shift(-1),
    # i.e. looks forward -- this is the ONLY forward-looking line in the
    # module, and it belongs on the label, never on a feature column.
    #
    # Careful: `NaN > x` evaluates to False in pandas/numpy, not NaN. If we
    # cast the raw comparison directly, the last row (which has no "next
    # close") silently gets labeled 0 instead of missing -- a fabricated
    # label that would leak straight into training. We compute the shift
    # once, then explicitly null out rows where it's undefined.
    next_close = out["Close"].shift(-1)
    out[TARGET_COLUMN] = (next_close > out["Close"]).astype("int64")
    out.loc[next_close.isna(), TARGET_COLUMN] = pd.NA

    out = out.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    return out.loc[:, FEATURE_COLUMNS + [TARGET_COLUMN]].astype({TARGET_COLUMN: int})
