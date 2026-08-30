"""Training loop with early stopping, gradient clipping and checkpointing.

The Trainer owns the epoch loop so it can be tested as an object rather than
existing as top-level script code that only runs end to end.

Three details that are easy to get wrong and expensive to debug:

1. `model.train()` / `model.eval()` must bracket every pass. Dropout stays
   active in train mode and must be off during validation, or the validation
   loss is noisy and early stopping fires at the wrong epoch.
2. Early stopping restores the *best* weights, not the last epoch's. Without
   the restore you keep whichever overfitted state the loop happened to end on.
3. The scaler is checkpointed alongside the weights. Inference must apply the
   same transformation; refitting at serving time is a silent production bug.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader

from cryptopulse.training.metrics import ClassificationMetrics, compute_metrics

logger = logging.getLogger(__name__)


@dataclass
class EpochRecord:
    """Metrics for a single epoch."""

    epoch: int
    train_loss: float
    val_metrics: ClassificationMetrics


@dataclass
class TrainingResult:
    """Outcome of a full training run.

    Attributes:
        best_epoch: 1-indexed epoch with the lowest validation loss.
        best_val_loss: That epoch's validation loss.
        epochs_run: How many epochs actually executed before stopping.
        early_stopped: Whether patience was exhausted before max_epochs.
        history: Per-epoch records, in order.
    """

    best_epoch: int
    best_val_loss: float
    epochs_run: int
    early_stopped: bool
    history: list[EpochRecord] = field(default_factory=list)


class Trainer:
    """Trains an LSTM classifier with early stopping on validation loss.

    Attributes:
        model: The network being trained. Mutated in place.
        device: Device tensors are moved to.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        learning_rate: float = 1e-3,
        max_epochs: int = 100,
        patience: int = 10,
        grad_clip_norm: float | None = 1.0,
        device: str = "cpu",
    ) -> None:
        """Configure the training run.

        Args:
            model: Network producing raw logits of shape (batch,).
            train_loader: Training batches.
            val_loader: Validation batches. Must not shuffle.
            criterion: Loss function. Defaults to BCEWithLogitsLoss.
            optimizer: Optimizer. Defaults to Adam over model.parameters().
            learning_rate: Used only when optimizer is None.
            max_epochs: Upper bound on epochs.
            patience: Epochs without validation improvement before stopping.
            grad_clip_norm: Max gradient norm, or None to disable clipping.
            device: "cpu" or "cuda".

        Raises:
            ValueError: If max_epochs or patience is not positive.
        """
        if max_epochs < 1:
            raise ValueError(f"max_epochs must be >= 1, got {max_epochs}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")

        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # BCEWithLogitsLoss, not Sigmoid + BCELoss: it fuses the sigmoid using
        # log-sum-exp, which stays finite for large-magnitude logits where a
        # separate sigmoid would saturate to exactly 0.0 or 1.0 and produce
        # log(0) = -inf.
        self.criterion = criterion if criterion is not None else nn.BCEWithLogitsLoss()
        self.optimizer = (
            optimizer
            if optimizer is not None
            else torch.optim.Adam(model.parameters(), lr=learning_rate)
        )

        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip_norm = grad_clip_norm

        self._best_state: dict[str, torch.Tensor] | None = None

    def train_one_epoch(self) -> float:
        """Run one pass over the training set.

        Returns:
            Mean training loss, weighted by batch size so a smaller final
            batch does not distort the average.
        """
        self.model.train()  # dropout ON
        total_loss = 0.0
        total_samples = 0

        for sequences, labels in self.train_loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(sequences)
            loss = self.criterion(logits, labels)

            # zero_grad BEFORE backward: PyTorch accumulates gradients, so
            # skipping this sums every batch's gradients across the epoch.
            self.optimizer.zero_grad()
            loss.backward()

            # Clip AFTER backward, BEFORE step. Backprop through 30 timesteps
            # multiplies gradients repeatedly; products above 1 compound
            # exponentially and spike the loss to NaN. Clipping rescales the
            # gradient vector when its norm exceeds the cap, preserving
            # direction while bounding magnitude.
            if self.grad_clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

            self.optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

        return total_loss / total_samples

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> ClassificationMetrics:
        """Evaluate the model over a loader without updating weights.

        Args:
            loader: Batches to evaluate. Should not shuffle.

        Returns:
            Full metric set for the pass.
        """
        self.model.eval()  # dropout OFF -- deterministic outputs
        total_loss = 0.0
        total_samples = 0
        all_probabilities: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for sequences, labels in loader:
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)

            logits = self.model(sequences)
            loss = self.criterion(logits, labels)

            # sigmoid applied HERE, at inference, never inside the model.
            probabilities = torch.sigmoid(logits)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            all_probabilities.append(probabilities.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        return compute_metrics(
            labels=np.concatenate(all_labels),
            probabilities=np.concatenate(all_probabilities),
            loss=total_loss / total_samples,
        )

    def fit(self) -> TrainingResult:
        """Train until validation loss stops improving or max_epochs is hit.

        Restores the best-performing weights into self.model before returning,
        so the trained model is the best one seen rather than the last one.

        Returns:
            TrainingResult with the best epoch and full per-epoch history.
        """
        history: list[EpochRecord] = []
        best_val_loss = float("inf")
        best_epoch = 0
        epochs_without_improvement = 0
        early_stopped = False

        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.train_one_epoch()
            val_metrics = self.evaluate(self.val_loader)
            history.append(EpochRecord(epoch, train_loss, val_metrics))

            logger.info(
                "epoch %3d | train_loss %.4f | val_loss %.4f | val_acc %.4f "
                "(baseline %.4f) | pos_rate %.3f",
                epoch,
                train_loss,
                val_metrics.loss,
                val_metrics.accuracy,
                val_metrics.baseline_accuracy,
                val_metrics.positive_prediction_rate,
            )

            if val_metrics.loss < best_val_loss:
                best_val_loss = val_metrics.loss
                best_epoch = epoch
                epochs_without_improvement = 0
                # deepcopy, not a reference: state_dict() returns tensors that
                # the optimizer will mutate in place on the next step.
                self._best_state = copy.deepcopy(self.model.state_dict())
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= self.patience:
                    logger.info(
                        "early stopping at epoch %d; best was epoch %d (val_loss %.4f)",
                        epoch,
                        best_epoch,
                        best_val_loss,
                    )
                    early_stopped = True
                    break

        if self._best_state is not None:
            self.model.load_state_dict(self._best_state)

        return TrainingResult(
            best_epoch=best_epoch,
            best_val_loss=best_val_loss,
            epochs_run=len(history),
            early_stopped=early_stopped,
            history=history,
        )

    def save_checkpoint(
        self,
        path: str | Path,
        scaler: StandardScaler,
        feature_columns: list[str],
        sequence_length: int,
        model_config: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> None:
        """Persist weights, scaler parameters and the architecture config.

        The scaler is stored as plain lists rather than a pickled sklearn
        object. Pickled estimators break across sklearn versions and require
        torch.load(weights_only=False), which is unsafe for untrusted files.
        Plain numbers reconstruct anywhere.

        Args:
            path: Destination file.
            scaler: Fitted scaler whose statistics to persist.
            feature_columns: Input column order the model expects.
            sequence_length: Timesteps per window.
            model_config: Kwargs needed to rebuild the architecture.
            metrics: Optional final metrics to record alongside.

        Raises:
            ValueError: If the scaler has not been fitted.
        """
        if not hasattr(scaler, "mean_"):
            raise ValueError("scaler must be fitted before checkpointing")

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "model_config": model_config,
                "scaler_mean": scaler.mean_.tolist(),
                "scaler_scale": scaler.scale_.tolist(),
                "feature_columns": list(feature_columns),
                "sequence_length": sequence_length,
                "metrics": metrics or {},
            },
            destination,
        )
        logger.info("checkpoint saved to %s", destination)


def load_scaler(mean: list[float], scale: list[float]) -> StandardScaler:
    """Rebuild a StandardScaler from persisted statistics.

    Args:
        mean: Per-feature means from the training partition.
        scale: Per-feature standard deviations.

    Returns:
        A scaler that reproduces the original transform exactly.
    """
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(mean, dtype=np.float64)
    scaler.scale_ = np.asarray(scale, dtype=np.float64)
    scaler.var_ = scaler.scale_**2
    scaler.n_features_in_ = len(mean)
    return scaler
