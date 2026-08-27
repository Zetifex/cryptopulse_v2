"""Tests for cryptopulse.models.lstm."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from cryptopulse.models.lstm import LSTMDirectionClassifier


def test_forward_returns_one_logit_per_sequence():
    model = LSTMDirectionClassifier(input_size=4, hidden_size=32)
    x = torch.randn(8, 30, 4)

    logits = model(x)

    assert logits.shape == (8,)
    assert logits.dtype == torch.float32


def test_batch_size_one_keeps_batch_dimension():
    """squeeze(-1) must not collapse a single-sample batch to a scalar."""
    model = LSTMDirectionClassifier()
    logits = model(torch.randn(1, 30, 4))

    assert logits.shape == (1,)


def test_output_is_raw_logits_not_probabilities():
    """A sigmoid in forward() would confine outputs to (0, 1)."""
    model = LSTMDirectionClassifier()
    # Push the head to produce large-magnitude values.
    with torch.no_grad():
        model.head.weight.fill_(10.0)
        model.head.bias.fill_(-5.0)

    logits = model(torch.randn(64, 30, 4))

    assert logits.min() < 0.0 or logits.max() > 1.0


def test_hidden_state_matches_output_last_timestep():
    """Verifies h_n[-1] == output[:, -1, :] for this configuration.

    True for a unidirectional LSTM over fixed-length sequences. Documented as
    a test so the assumption is checked rather than assumed.
    """
    lstm = nn.LSTM(input_size=4, hidden_size=32, num_layers=1, batch_first=True)
    lstm.eval()
    x = torch.randn(8, 30, 4)

    with torch.no_grad():
        output, (h_n, _) = lstm(x)

    assert output.shape == (8, 30, 32)
    assert h_n.shape == (1, 8, 32)  # batch is dim 1, unaffected by batch_first
    torch.testing.assert_close(h_n[-1], output[:, -1, :])


def test_bidirectional_breaks_the_equivalence():
    """Counterexample: with bidirectional=True the two are different tensors."""
    lstm = nn.LSTM(input_size=4, hidden_size=32, batch_first=True, bidirectional=True)
    lstm.eval()
    x = torch.randn(8, 30, 4)

    with torch.no_grad():
        output, (h_n, _) = lstm(x)

    assert output.shape == (8, 30, 64)  # forward + backward concatenated
    assert h_n.shape == (2, 8, 32)
    # The backward direction's final state is at timestep 0, not timestep -1.
    assert not torch.allclose(h_n[1], output[:, -1, 32:])


def test_eval_mode_is_deterministic_train_mode_is_not():
    """Dropout must be active in train() and disabled in eval()."""
    torch.manual_seed(0)
    model = LSTMDirectionClassifier(dropout=0.5)
    x = torch.randn(16, 30, 4)

    model.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), model(x))

    model.train()
    with torch.no_grad():
        assert not torch.allclose(model(x), model(x))


def test_gradients_flow_to_all_parameters():
    model = LSTMDirectionClassifier()
    criterion = nn.BCEWithLogitsLoss()

    logits = model(torch.randn(8, 30, 4))
    loss = criterion(logits, torch.randint(0, 2, (8,), dtype=torch.float32))
    loss.backward()

    for name, param in model.named_parameters():
        assert param.grad is not None, f"{name} received no gradient"
        assert torch.isfinite(param.grad).all(), f"{name} has non-finite gradients"


def test_loss_accepts_model_output_without_reshaping():
    """Model output (batch,) and float labels (batch,) must align exactly.

    A (batch, 1) vs (batch,) mismatch would broadcast into a (batch, batch)
    loss matrix -- which runs, returns a number, and trains nonsense.
    """
    model = LSTMDirectionClassifier()
    logits = model(torch.randn(8, 30, 4))
    labels = torch.randint(0, 2, (8,), dtype=torch.float32)

    loss = nn.BCEWithLogitsLoss()(logits, labels)

    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_parameter_count_is_small_relative_to_dataset():
    model = LSTMDirectionClassifier(input_size=4, hidden_size=32)

    # 4 gates * ((4+32)*32 + 32 + 32) + (32 + 1)
    assert model.count_parameters() == 4897


def test_rejects_wrong_feature_dimension():
    model = LSTMDirectionClassifier(input_size=4)
    with pytest.raises(ValueError, match="expected 4 features"):
        model(torch.randn(8, 4, 30))  # transposed


def test_rejects_non_3d_input():
    model = LSTMDirectionClassifier()
    with pytest.raises(ValueError, match="expected a 3D tensor"):
        model(torch.randn(30, 4))


def test_rejects_invalid_hyperparameters():
    with pytest.raises(ValueError, match="must all be >= 1"):
        LSTMDirectionClassifier(hidden_size=0)
    with pytest.raises(ValueError, match="dropout must be in"):
        LSTMDirectionClassifier(dropout=1.0)


def test_seeding_makes_initialization_reproducible():
    torch.manual_seed(42)
    a = LSTMDirectionClassifier()
    torch.manual_seed(42)
    b = LSTMDirectionClassifier()

    for pa, pb in zip(a.parameters(), b.parameters(), strict=True):
        torch.testing.assert_close(pa, pb)
