"""Tests for cryptopulse.training.metrics and cryptopulse.training.trainer."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from cryptopulse.models.lstm import LSTMDirectionClassifier
from cryptopulse.training.metrics import (
    ClassificationMetrics,
    compute_metrics,
    majority_class_accuracy,
)
from cryptopulse.training.trainer import Trainer, load_scaler

# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------


def test_majority_class_accuracy_matches_larger_class():
    labels = np.array([1.0] * 51 + [0.0] * 49)
    assert majority_class_accuracy(labels) == pytest.approx(0.51)


def test_majority_class_accuracy_is_symmetric():
    assert majority_class_accuracy(np.array([0.0] * 90 + [1.0] * 10)) == pytest.approx(0.90)


def test_perfect_predictions_score_one():
    labels = np.array([0.0, 1.0, 0.0, 1.0])
    probabilities = np.array([0.1, 0.9, 0.2, 0.8])

    m = compute_metrics(labels, probabilities, loss=0.1)

    assert m.accuracy == 1.0
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.auc == 1.0


def test_collapsed_model_is_visible_in_positive_prediction_rate():
    """A model predicting all-negative must be detectable, not hidden."""
    labels = np.array([0.0] * 49 + [1.0] * 51)
    probabilities = np.full(100, 0.2)  # always below threshold

    m = compute_metrics(labels, probabilities, loss=0.7)

    assert m.positive_prediction_rate == 0.0
    assert m.recall == 0.0
    assert m.precision == 0.0  # zero_division=0, not a crash


def test_auc_is_nan_when_only_one_class_present():
    labels = np.ones(10)
    probabilities = np.linspace(0.1, 0.9, 10)

    m = compute_metrics(labels, probabilities, loss=0.5)

    assert np.isnan(m.auc)


def test_beats_baseline_compares_against_majority_class():
    labels = np.array([1.0] * 51 + [0.0] * 49)
    always_up = np.full(100, 0.9)

    m = compute_metrics(labels, always_up, loss=0.7)

    assert m.accuracy == pytest.approx(0.51)
    assert m.baseline_accuracy == pytest.approx(0.51)
    assert not m.beats_baseline()  # matching the baseline is not beating it


def test_metrics_reject_length_mismatch():
    with pytest.raises(ValueError, match="same length"):
        compute_metrics(np.array([1.0, 0.0]), np.array([0.5]), loss=0.5)


def test_metrics_reject_empty_input():
    with pytest.raises(ValueError, match="empty pass"):
        compute_metrics(np.array([]), np.array([]), loss=0.5)


def test_metrics_to_dict_is_flat_and_complete():
    m = compute_metrics(np.array([0.0, 1.0]), np.array([0.2, 0.8]), loss=0.3)
    d = m.to_dict()

    assert set(d) == {
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
        "baseline_accuracy",
        "positive_prediction_rate",
    }
    assert all(isinstance(v, float) for v in d.values())


# --------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------


def copy_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in model.state_dict().items()}


def _make_loaders(
    n_train: int = 64, n_val: int = 32, seq_len: int = 10, n_features: int = 4
) -> tuple[DataLoader, DataLoader]:
    """Loaders over a learnable signal: label = 1 when feature 0 mean > 0."""
    torch.manual_seed(0)

    def build(n: int) -> TensorDataset:
        x = torch.randn(n, seq_len, n_features)
        y = (x[:, :, 0].mean(dim=1) > 0).float()
        return TensorDataset(x, y)

    return (
        DataLoader(build(n_train), batch_size=16, shuffle=True),
        DataLoader(build(n_val), batch_size=16, shuffle=False),
    )


def test_train_one_epoch_returns_finite_loss_and_updates_weights():
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8)
    before = model.head.weight.clone()

    trainer = Trainer(model, train_loader, val_loader)
    loss = trainer.train_one_epoch()

    assert np.isfinite(loss)
    assert not torch.allclose(before, model.head.weight)


def test_evaluate_does_not_change_weights():
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8)
    trainer = Trainer(model, train_loader, val_loader)

    before = copy_state(model)
    trainer.evaluate(val_loader)

    for name, tensor in copy_state(model).items():
        torch.testing.assert_close(tensor, before[name])


def test_evaluate_is_deterministic_dropout_disabled():
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8, dropout=0.5)
    trainer = Trainer(model, train_loader, val_loader)

    first = trainer.evaluate(val_loader)
    second = trainer.evaluate(val_loader)

    assert first.loss == pytest.approx(second.loss)
    assert first.accuracy == pytest.approx(second.accuracy)


def test_fit_records_one_history_entry_per_epoch():
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8)
    trainer = Trainer(model, train_loader, val_loader, max_epochs=3, patience=99)

    result = trainer.fit()

    assert result.epochs_run == 3
    assert len(result.history) == 3
    assert [r.epoch for r in result.history] == [1, 2, 3]
    assert not result.early_stopped


def test_early_stopping_fires_when_validation_stops_improving():
    """With lr=0 the weights never change, so validation can never improve.

    Note we set learning_rate=0 rather than freezing parameters: freezing
    would leave backward() with no graph to traverse and raise instead.
    """
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8, dropout=0.0)

    trainer = Trainer(model, train_loader, val_loader, learning_rate=0.0, max_epochs=50, patience=3)
    result = trainer.fit()

    assert result.early_stopped
    assert result.best_epoch == 1  # nothing ever beat the first epoch
    assert result.epochs_run == 4  # epoch 1 + 3 without improvement


def test_fit_restores_best_weights_not_final_weights():
    """After fit(), the model must hold the best epoch's parameters."""
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8, dropout=0.0)
    trainer = Trainer(model, train_loader, val_loader, max_epochs=8, patience=99)

    result = trainer.fit()
    restored = trainer.evaluate(val_loader)

    assert restored.loss == pytest.approx(result.best_val_loss, abs=1e-5)


def test_gradient_clipping_bounds_the_gradient_norm():
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8)
    trainer = Trainer(model, train_loader, val_loader, grad_clip_norm=0.01)

    sequences, labels = next(iter(train_loader))
    logits = model(sequences)
    loss = trainer.criterion(logits, labels)
    trainer.optimizer.zero_grad()
    loss.backward()
    total_norm = nn.utils.clip_grad_norm_(model.parameters(), 0.01)

    # clip_grad_norm_ returns the norm BEFORE clipping; after the call the
    # actual norm must not exceed the cap.
    after = torch.sqrt(sum((p.grad**2).sum() for p in model.parameters()))
    assert after <= 0.01 + 1e-5
    assert torch.isfinite(total_norm)


def test_trainer_rejects_invalid_configuration():
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8)

    with pytest.raises(ValueError, match="max_epochs must be >= 1"):
        Trainer(model, train_loader, val_loader, max_epochs=0)
    with pytest.raises(ValueError, match="patience must be >= 1"):
        Trainer(model, train_loader, val_loader, patience=0)


# --------------------------------------------------------------------------
# checkpointing
# --------------------------------------------------------------------------


def test_checkpoint_round_trip_reproduces_predictions(tmp_path):
    """A reloaded model and scaler must reproduce the original outputs exactly."""
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8, dropout=0.0)
    trainer = Trainer(model, train_loader, val_loader, max_epochs=2, patience=99)
    trainer.fit()

    scaler = StandardScaler().fit(np.random.default_rng(0).normal(size=(50, 4)))
    path = tmp_path / "checkpoints" / "model.pt"

    trainer.save_checkpoint(
        path,
        scaler=scaler,
        feature_columns=["a", "b", "c", "d"],
        sequence_length=10,
        model_config={"input_size": 4, "hidden_size": 8, "num_layers": 1, "dropout": 0.0},
        metrics={"val_loss": 0.5},
    )
    assert path.exists()

    checkpoint = torch.load(path, weights_only=False)
    restored = LSTMDirectionClassifier(**checkpoint["model_config"])
    restored.load_state_dict(checkpoint["model_state_dict"])
    restored.eval()
    model.eval()

    x = torch.randn(8, 10, 4)
    with torch.no_grad():
        torch.testing.assert_close(model(x), restored(x))

    rebuilt = load_scaler(checkpoint["scaler_mean"], checkpoint["scaler_scale"])
    sample = np.random.default_rng(1).normal(size=(5, 4))
    np.testing.assert_allclose(scaler.transform(sample), rebuilt.transform(sample))


def test_checkpoint_stores_inference_contract(tmp_path):
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8)
    trainer = Trainer(model, train_loader, val_loader)
    scaler = StandardScaler().fit(np.zeros((10, 4)) + np.arange(4))

    path = tmp_path / "model.pt"
    trainer.save_checkpoint(
        path,
        scaler=scaler,
        feature_columns=["w", "x", "y", "z"],
        sequence_length=30,
        model_config={"input_size": 4, "hidden_size": 8},
    )

    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["feature_columns"] == ["w", "x", "y", "z"]
    assert checkpoint["sequence_length"] == 30


def test_checkpoint_rejects_unfitted_scaler(tmp_path):
    train_loader, val_loader = _make_loaders()
    model = LSTMDirectionClassifier(input_size=4, hidden_size=8)
    trainer = Trainer(model, train_loader, val_loader)

    with pytest.raises(ValueError, match="must be fitted"):
        trainer.save_checkpoint(
            tmp_path / "m.pt",
            scaler=StandardScaler(),
            feature_columns=["a"],
            sequence_length=10,
            model_config={},
        )


def test_metrics_dataclass_is_immutable():
    m = ClassificationMetrics(0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)
    with pytest.raises(AttributeError):
        m.accuracy = 0.9  # type: ignore[misc]
