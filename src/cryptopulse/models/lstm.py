"""LSTM classifier for next-day direction prediction.

Architecture notes that matter:

- The head emits a RAW LOGIT with no activation. BCEWithLogitsLoss fuses the
  sigmoid into its own computation for numerical stability; adding a sigmoid
  here would double-apply it, which trains badly and raises no error.
- Dropout sits between the LSTM and the head, not inside nn.LSTM's `dropout`
  argument. That argument applies *between* stacked layers and is a silent
  no-op when num_layers=1 (PyTorch warns about it).
- The final representation comes from `h_n[-1]`, the last layer's final
  hidden state. For this configuration that equals `output[:, -1, :]`, but
  h_n stays correct if the layer ever becomes bidirectional or uses packed
  variable-length sequences.
"""

from __future__ import annotations

import torch
from torch import nn


class LSTMDirectionClassifier(nn.Module):
    """Single-layer LSTM followed by dropout and a linear classification head.

    Deliberately small: with roughly 1,900 training sequences available, a
    larger network memorizes the training set rather than learning a
    transferable signal.

    Attributes:
        input_size: Number of features per timestep.
        hidden_size: LSTM hidden dimension.
        num_layers: Number of stacked LSTM layers.
    """

    def __init__(
        self,
        input_size: int = 4,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.2,
    ) -> None:
        """Build the network.

        Args:
            input_size: Features per timestep. Must match len(FEATURE_COLUMNS).
            hidden_size: LSTM hidden dimension.
            num_layers: Stacked LSTM layers.
            dropout: Dropout probability applied to the final hidden state,
                in [0.0, 1.0).

        Raises:
            ValueError: If any dimension is non-positive or dropout is out of
                range.
        """
        super().__init__()

        if input_size < 1 or hidden_size < 1 or num_layers < 1:
            raise ValueError(
                f"input_size, hidden_size and num_layers must all be >= 1; got "
                f"{input_size}, {hidden_size}, {num_layers}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {dropout}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # batch_first=True gives (batch, seq, feature) for input and output,
        # matching every other layer in the stack. Note it does NOT change the
        # layout of h_n / c_n, which stay (num_layers, batch, hidden_size).
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map a batch of sequences to one raw logit per sequence.

        Args:
            x: float tensor of shape (batch, seq_len, input_size).

        Returns:
            Raw logits of shape (batch,). Apply torch.sigmoid at inference to
            obtain probabilities; do NOT apply it before BCEWithLogitsLoss.

        Raises:
            ValueError: If x is not 3-dimensional or its feature dimension
                does not equal input_size.
        """
        # Cheap guard against the classic transposition bug: passing
        # (batch, features, seq_len) would not error inside nn.LSTM if the two
        # dimensions happened to be compatible -- it would silently treat
        # features as timesteps and train to a mediocre result.
        if x.ndim != 3:
            raise ValueError(f"expected a 3D tensor (batch, seq_len, features), got {x.ndim}D")
        if x.shape[-1] != self.input_size:
            raise ValueError(
                f"expected {self.input_size} features in the last dimension, "
                f"got {x.shape[-1]}; check for a transposed (batch, features, seq) input"
            )

        # h_n: (num_layers, batch, hidden_size) -- batch is dim 1, not 0.
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)

        logits: torch.Tensor = self.head(self.dropout(last_hidden))  # (batch, 1)

        # squeeze(-1), not squeeze(): a bare squeeze() would also collapse the
        # batch dimension when batch_size == 1, producing a 0-d tensor and a
        # broadcasting bug in the loss.
        return logits.squeeze(-1)

    def count_parameters(self) -> int:
        """Total number of trainable parameters."""

        return sum(p.numel() for p in self.parameters() if p.requires_grad)
