"""Chronological train/validation/test splitting and feature scaling.

Two invariants this module exists to enforce:

1. Splits are strictly chronological. A random shuffle would place future
   rows into the training set, inflating every metric downstream.
2. The scaler is fitted on training rows only. Fitting on the full dataset
   would leak the mean and standard deviation of the test period into the
   training inputs.

Sequence construction happens *after* splitting (see models/dataset.py), so
no input window ever spans a partition boundary. That costs roughly 2% of
the available sequences and removes the possibility of boundary leakage.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.preprocessing import StandardScaler

from cryptopulse.data.features import FEATURE_COLUMNS


@dataclass(frozen=True)
class SplitData:
    """Scaled partitions plus the scaler that produced them.

    The scaler travels with the data because inference must apply the
    identical transformation. Refitting a scaler at serving time is a
    production bug that is very difficult to detect from metrics alone.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    scaler: StandardScaler


def chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered DataFrame into three contiguous partitions.

    Args:
        df: Feature DataFrame indexed by date, sorted ascending.
        train_frac: Fraction of rows assigned to the training partition.
        val_frac: Fraction assigned to validation. The remainder becomes
            the test partition.

    Returns:
        Tuple of (train, val, test) DataFrames in chronological order.

    Raises:
        ValueError: If the index is not sorted ascending, if the fractions
            are not a valid split, or if any partition would be empty.
    """
    if not df.index.is_monotonic_increasing:
        raise ValueError(
            "chronological_split requires a date-sorted index; "
            "sort the DataFrame before splitting"
        )
    if train_frac <= 0 or val_frac <= 0 or train_frac + val_frac >= 1.0:
        raise ValueError(
            f"invalid split fractions: train_frac={train_frac}, val_frac={val_frac}. "
            "Both must be positive and sum to less than 1.0"
        )

    n = len(df)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    train = df.iloc[:train_end]
    val = df.iloc[train_end:val_end]
    test = df.iloc[val_end:]

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            f"split produced an empty partition from {n} rows "
            f"(train={len(train)}, val={len(val)}, test={len(test)})"
        )

    return train, val, test


def scale_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> SplitData:
    """Standardize features using statistics from the training partition only.

    Args:
        train: Training partition, used to fit the scaler.
        val: Validation partition, transformed with the fitted scaler.
        test: Test partition, transformed with the fitted scaler.
        feature_columns: Columns to scale. Defaults to FEATURE_COLUMNS.
            The label column is never scaled.

    Returns:
        SplitData holding the three scaled partitions and the fitted scaler.
    """
    columns = list(FEATURE_COLUMNS) if feature_columns is None else list(feature_columns)

    scaler = StandardScaler()

    train_scaled = train.copy()
    val_scaled = val.copy()
    test_scaled = test.copy()

    # fit_transform on train ONLY. val/test get transform, which applies the
    # stored training statistics rather than computing their own.
    train_scaled[columns] = scaler.fit_transform(train[columns])
    val_scaled[columns] = scaler.transform(val[columns])
    test_scaled[columns] = scaler.transform(test[columns])

    return SplitData(
        train=train_scaled,
        val=val_scaled,
        test=test_scaled,
        scaler=scaler,
    )
