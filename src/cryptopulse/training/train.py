"""Training entry point: load features, split, train, evaluate, checkpoint.

Run with:

    make train

The test partition is evaluated exactly once, at the very end. Every
hyperparameter decision must be made against the validation set; repeatedly
checking test performance and tuning against it turns the test set into a
second validation set and the reported number stops meaning anything.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cryptopulse.config import settings
from cryptopulse.data.features import FEATURE_COLUMNS
from cryptopulse.data.persistence import load_features
from cryptopulse.models.dataset import SequenceDataset
from cryptopulse.models.lstm import LSTMDirectionClassifier
from cryptopulse.training.metrics import ClassificationMetrics
from cryptopulse.training.split import chronological_split, scale_splits
from cryptopulse.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Seed every RNG the training path touches.

    torch.manual_seed alone is not enough: DataLoader shuffling draws from
    Python's `random`, and any numpy-based augmentation draws from numpy.
    Seeding one and not the others gives runs that look reproducible until
    they suddenly aren't.

    Args:
        seed: Value applied to random, numpy and torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _log_metrics(label: str, metrics: ClassificationMetrics) -> None:
    """Log a metric set with the baseline alongside, never accuracy alone."""
    verdict = "BEATS" if metrics.beats_baseline() else "does NOT beat"
    logger.info(
        "%s | loss %.4f | acc %.4f (%s baseline %.4f) | prec %.4f | rec %.4f "
        "| f1 %.4f | auc %.4f | pos_rate %.3f",
        label,
        metrics.loss,
        metrics.accuracy,
        verdict,
        metrics.baseline_accuracy,
        metrics.precision,
        metrics.recall,
        metrics.f1,
        metrics.auc,
        metrics.positive_prediction_rate,
    )


def run() -> None:
    """Execute one full training run and persist the best checkpoint."""
    set_seed(settings.random_seed)

    features = load_features(settings.db_path, settings.features_table)
    logger.info("loaded %d feature rows from %s", len(features), settings.db_path)

    splits = scale_splits(*chronological_split(features, settings.train_frac, settings.val_frac))

    train_ds = SequenceDataset(splits.train, settings.sequence_length)
    val_ds = SequenceDataset(splits.val, settings.sequence_length)
    test_ds = SequenceDataset(splits.test, settings.sequence_length)
    logger.info(
        "sequences: train=%d val=%d test=%d | train positive rate %.4f",
        len(train_ds),
        len(val_ds),
        len(test_ds),
        train_ds.positive_rate,
    )

    # Shuffle training only. Each window is a self-contained ordered snapshot,
    # so shuffling their order helps convergence without disturbing time order
    # inside any window. Validation and test stay in order for reproducibility.
    train_loader = DataLoader(train_ds, batch_size=settings.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=settings.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=settings.batch_size, shuffle=False)

    model_config: dict[str, Any] = {
        "input_size": len(FEATURE_COLUMNS),
        "hidden_size": settings.hidden_size,
        "num_layers": settings.num_layers,
        "dropout": settings.dropout,
    }
    model = LSTMDirectionClassifier(**model_config)
    logger.info("model: %s | %d parameters", model_config, model.count_parameters())

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=settings.learning_rate,
        max_epochs=settings.max_epochs,
        patience=settings.patience,
        grad_clip_norm=settings.grad_clip_norm,
    )

    result = trainer.fit()
    logger.info(
        "training finished after %d epochs; best epoch %d (val_loss %.4f)",
        result.epochs_run,
        result.best_epoch,
        result.best_val_loss,
    )

    val_metrics = trainer.evaluate(val_loader)
    _log_metrics("VAL ", val_metrics)

    # Test evaluated ONCE, after all decisions are locked in.
    test_metrics = trainer.evaluate(test_loader)
    _log_metrics("TEST", test_metrics)

    trainer.save_checkpoint(
        settings.checkpoint_path,
        scaler=splits.scaler,
        feature_columns=FEATURE_COLUMNS,
        sequence_length=settings.sequence_length,
        model_config=model_config,
        metrics={f"test_{k}": v for k, v in test_metrics.to_dict().items()},
    )

    if not test_metrics.beats_baseline():
        logger.warning(
            "test accuracy %.4f does not beat the %.4f majority-class baseline. "
            "This is a legitimate result for next-day direction prediction, not "
            "necessarily a bug.",
            test_metrics.accuracy,
            test_metrics.baseline_accuracy,
        )
    if test_metrics.accuracy > 0.60:
        logger.warning(
            "test accuracy %.4f is implausibly high for this task. Suspect "
            "leakage before celebrating.",
            test_metrics.accuracy,
        )


if __name__ == "__main__":
    run()
