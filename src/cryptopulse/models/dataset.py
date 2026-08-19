"""Sliding-window Dataset for sequence models.

Label alignment is the critical detail in this module. A window covering
rows [s, s+L-1] is paired with the label at row s+L-1 -- the *last* row of
the window. Because `target_up` at row t already encodes "does close(t+1)
exceed close(t)", the model sees history up to and including t, and predicts
the move from t to t+1. It never sees row t+1 itself.

Pairing the window with the label at row s+L would be a one-day lookahead
bug: the window's final row would already contain the outcome being
predicted. See test_dataset.py::test_label_aligns_with_window_end.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cryptopulse.data.features import FEATURE_COLUMNS, TARGET_COLUMN


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Map-style Dataset yielding (sequence, label) pairs from a feature frame.

    Expects a DataFrame that has already been scaled (see
    training/split.scale_splits). Build one instance per partition so no
    window ever spans a train/val/test boundary.

    Attributes:
        sequence_length: Number of timesteps per input window.
        feature_columns: Columns used as model inputs, in order.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        sequence_length: int = 30,
        feature_columns: list[str] | None = None,
        target_column: str = TARGET_COLUMN,
    ) -> None:
        """Initialize the dataset from a scaled feature DataFrame.

        Args:
            df: Scaled features indexed by date, sorted ascending.
            sequence_length: Timesteps per window. Must be >= 1.
            feature_columns: Model input columns. Defaults to FEATURE_COLUMNS.
            target_column: Binary label column.

        Raises:
            ValueError: If sequence_length is invalid, required columns are
                missing, the frame holds too few rows for one window, or any
                feature/label value is null.
        """
        columns = list(FEATURE_COLUMNS) if feature_columns is None else list(feature_columns)

        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")

        # Parentheses matter: `-` binds tighter than `|`, so without them this
        # would compute columns | ({target} - df.columns), which is wrong.
        missing = (set(columns) | {target_column}) - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

        if len(df) < sequence_length:
            raise ValueError(
                f"need at least sequence_length={sequence_length} rows to build one "
                f"window, got {len(df)}"
            )

        subset = df[columns + [target_column]]
        if subset.isna().to_numpy().any():
            raise ValueError(
                "DataFrame contains nulls in feature or label columns; "
                "clean the data before constructing the dataset"
            )

        self.sequence_length = sequence_length
        self.feature_columns = columns

        # Materialize to contiguous float32 arrays once, in __init__, rather
        # than calling .iloc per item. pandas indexing is orders of magnitude
        # slower than numpy slicing, and __getitem__ runs once per sample per
        # epoch -- this is the difference between a fast and a stalled loader.
        self._features: np.ndarray = np.ascontiguousarray(
            subset[columns].to_numpy(dtype=np.float32)
        )
        # float32 labels, NOT int64: BCEWithLogitsLoss expects float targets.
        # (CrossEntropyLoss is the one that wants integer class indices.)
        self._labels: np.ndarray = subset[target_column].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        """Number of complete windows available."""
        return len(self._features) - self.sequence_length + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one (sequence, label) pair.

        Args:
            index: Window index in [0, len(self)).

        Returns:
            Tuple of:
                sequence: float32 tensor of shape (sequence_length, n_features)
                label: float32 scalar tensor, 0.0 or 1.0

            A DataLoader collates these into (batch, sequence_length,
            n_features) and (batch,) respectively.

        Raises:
            IndexError: If index is out of range.
        """
        if not 0 <= index < len(self):
            raise IndexError(f"index {index} out of range for {len(self)} windows")

        end = index + self.sequence_length
        window = self._features[index:end]
        # Label at the window's LAST row, not one past it. See module docstring.
        label = self._labels[end - 1]

        return torch.from_numpy(window.copy()), torch.tensor(label, dtype=torch.float32)

    @property
    def positive_rate(self) -> float:
        """Fraction of windows labeled 1, i.e. the majority-class baseline."""
        labels = self._labels[self.sequence_length - 1 :]
        return float(labels.mean())

    def pos_weight(self) -> torch.Tensor:
        """Weight for the positive class, for BCEWithLogitsLoss(pos_weight=...).

        Returns negatives/positives over this partition's labels. A value near
        1.0 means the classes are balanced and no reweighting is needed.

        Raises:
            ValueError: If the partition contains no positive labels.
        """
        rate = self.positive_rate
        if rate == 0.0:
            raise ValueError("partition contains no positive labels; cannot weight")
        return torch.tensor((1.0 - rate) / rate, dtype=torch.float32)
