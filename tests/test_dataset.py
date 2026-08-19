"""Tests for cryptopulse.models.dataset.

The two tests that matter most here are test_label_aligns_with_window_end
and test_no_lookahead_window_excludes_future_rows. Everything else checks
shapes and guard rails; those two check that the dataset can't cheat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from cryptopulse.data.features import FEATURE_COLUMNS, TARGET_COLUMN
from cryptopulse.models.dataset import SequenceDataset


def _make_scaled_df(n: int = 100) -> pd.DataFrame:
    """Feature frame with distinguishable values: feature j at row i is i + j/10."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    data = {col: [i + j / 10 for i in range(n)] for j, col in enumerate(FEATURE_COLUMNS)}
    data[TARGET_COLUMN] = [i % 2 for i in range(n)]
    return pd.DataFrame(data, index=dates)


def test_length_accounts_for_window_warmup():
    df = _make_scaled_df(100)
    ds = SequenceDataset(df, sequence_length=30)

    assert len(ds) == 100 - 30 + 1 == 71


def test_item_shapes_and_dtypes():
    df = _make_scaled_df(100)
    ds = SequenceDataset(df, sequence_length=30)
    sequence, label = ds[0]

    assert sequence.shape == (30, len(FEATURE_COLUMNS))
    assert sequence.dtype == torch.float32
    assert label.shape == ()
    assert label.dtype == torch.float32


def test_label_aligns_with_window_end():
    """Window [s, s+L-1] must pair with the label at row s+L-1, not s+L."""
    df = _make_scaled_df(50)
    ds = SequenceDataset(df, sequence_length=10)

    for idx in (0, 1, 17, len(ds) - 1):
        _, label = ds[idx]
        expected_row = idx + 10 - 1
        assert float(label) == float(
            df[TARGET_COLUMN].iloc[expected_row]
        ), f"window {idx} should carry the label from row {expected_row}"


def test_no_lookahead_window_excludes_future_rows():
    """A window must contain rows [s, s+L-1] and nothing beyond."""
    df = _make_scaled_df(50)
    ds = SequenceDataset(df, sequence_length=10)

    sequence, _ = ds[5]
    # First feature column at row i equals i exactly (j=0 offset).
    first_col = sequence[:, 0].numpy()

    np.testing.assert_allclose(first_col, np.arange(5, 15, dtype=np.float32))
    assert first_col.max() == 14.0  # row 14 = s+L-1, never row 15


def test_windows_are_contiguous_and_ordered():
    df = _make_scaled_df(50)
    ds = SequenceDataset(df, sequence_length=10)

    seq_a, _ = ds[0]
    seq_b, _ = ds[1]

    # Consecutive windows overlap by L-1 rows, shifted forward by exactly one.
    np.testing.assert_allclose(seq_a[1:, 0].numpy(), seq_b[:-1, 0].numpy())


def test_dataloader_collates_to_batch_first_shape():
    df = _make_scaled_df(100)
    ds = SequenceDataset(df, sequence_length=30)
    loader = DataLoader(ds, batch_size=8, shuffle=False)

    sequences, labels = next(iter(loader))

    assert sequences.shape == (8, 30, len(FEATURE_COLUMNS))
    assert labels.shape == (8,)


def test_shuffling_is_safe_because_windows_are_self_contained():
    """Shuffling reorders windows but never reorders rows within a window."""
    df = _make_scaled_df(100)
    ds = SequenceDataset(df, sequence_length=30)
    loader = DataLoader(ds, batch_size=4, shuffle=True, generator=torch.Generator())

    sequences, _ = next(iter(loader))
    for sequence in sequences:
        diffs = np.diff(sequence[:, 0].numpy())
        np.testing.assert_allclose(diffs, np.ones(29, dtype=np.float32))


def test_rejects_too_few_rows():
    df = _make_scaled_df(5)
    with pytest.raises(ValueError, match="at least sequence_length"):
        SequenceDataset(df, sequence_length=30)


def test_rejects_missing_columns():
    df = _make_scaled_df(100).drop(columns=[FEATURE_COLUMNS[0]])
    with pytest.raises(ValueError, match="missing required columns"):
        SequenceDataset(df, sequence_length=10)


def test_rejects_nulls():
    df = _make_scaled_df(100)
    df.iloc[42, 0] = np.nan
    with pytest.raises(ValueError, match="contains nulls"):
        SequenceDataset(df, sequence_length=10)


def test_rejects_invalid_sequence_length():
    df = _make_scaled_df(100)
    with pytest.raises(ValueError, match="sequence_length must be >= 1"):
        SequenceDataset(df, sequence_length=0)


def test_index_out_of_range_raises():
    df = _make_scaled_df(50)
    ds = SequenceDataset(df, sequence_length=10)
    with pytest.raises(IndexError):
        ds[len(ds)]


def test_positive_rate_counts_only_labeled_windows():
    df = _make_scaled_df(50)  # labels alternate 0,1,0,1,...
    ds = SequenceDataset(df, sequence_length=10)

    assert 0.0 < ds.positive_rate < 1.0


def test_pos_weight_is_near_one_for_balanced_labels():
    df = _make_scaled_df(50)
    ds = SequenceDataset(df, sequence_length=10)

    assert abs(float(ds.pos_weight()) - 1.0) < 0.15
