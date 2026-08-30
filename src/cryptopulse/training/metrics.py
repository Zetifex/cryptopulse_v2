"""Classification metrics for the direction classifier.

Accuracy alone is not reported anywhere in this project. With a 50.9%
majority class, a model that always predicts "up" scores 50.9% while
learning nothing -- so every accuracy figure travels alongside the baseline
it must beat, plus precision, recall, F1 and AUC.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Evaluation metrics for one pass over a dataset.

    Attributes:
        loss: Mean BCE loss over the pass.
        accuracy: Fraction of correct predictions at the decision threshold.
        precision: Of predicted positives, the fraction that were positive.
        recall: Of actual positives, the fraction that were predicted.
        f1: Harmonic mean of precision and recall.
        auc: Area under the ROC curve, threshold-independent. NaN when the
            labels contain only one class.
        baseline_accuracy: Accuracy of always predicting the majority class.
            The number `accuracy` has to beat to mean anything.
        positive_prediction_rate: Fraction predicted positive. A value near
            0.0 or 1.0 means the model collapsed to a constant prediction --
            the most common silent failure on a near-balanced problem.
    """

    loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    baseline_accuracy: float
    positive_prediction_rate: float

    def to_dict(self) -> dict[str, float]:
        """Flatten to a plain dict, suitable for experiment-tracking loggers."""
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "baseline_accuracy": self.baseline_accuracy,
            "positive_prediction_rate": self.positive_prediction_rate,
        }

    def beats_baseline(self) -> bool:
        """Whether accuracy exceeds always-predict-majority."""
        return self.accuracy > self.baseline_accuracy


def majority_class_accuracy(labels: np.ndarray) -> float:
    """Accuracy achieved by always predicting whichever class is more common.

    Args:
        labels: Binary labels, 0.0 or 1.0.

    Returns:
        The larger of the positive rate and its complement.
    """
    positive_rate = float(np.mean(labels))
    return max(positive_rate, 1.0 - positive_rate)


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    loss: float,
    threshold: float = 0.5,
) -> ClassificationMetrics:
    """Compute the full metric set from labels and predicted probabilities.

    Args:
        labels: Binary ground truth, shape (n,).
        probabilities: Predicted P(up), shape (n,), already sigmoid-applied.
        loss: Mean loss over the same pass, computed by the caller.
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        ClassificationMetrics for this pass.

    Raises:
        ValueError: If the arrays have mismatched or zero length.
    """
    if len(labels) != len(probabilities):
        raise ValueError(
            f"labels and probabilities must be the same length; "
            f"got {len(labels)} and {len(probabilities)}"
        )
    if len(labels) == 0:
        raise ValueError("cannot compute metrics over an empty pass")

    predictions = (probabilities >= threshold).astype(np.int64)
    int_labels = labels.astype(np.int64)

    # zero_division=0: on a collapsed model that predicts all-negative there
    # are no predicted positives, making precision undefined. Reporting 0.0
    # is more useful than raising, and the collapse is visible anyway in
    # positive_prediction_rate.
    precision, recall, f1, _ = precision_recall_fscore_support(
        int_labels, predictions, average="binary", zero_division=0
    )

    # roc_auc_score raises when only one class is present, which happens on
    # small validation slices. NaN propagates honestly rather than pretending
    # to a score we can't compute.
    try:
        auc = float(roc_auc_score(int_labels, probabilities))
    except ValueError:
        auc = float("nan")

    return ClassificationMetrics(
        loss=loss,
        accuracy=float(accuracy_score(int_labels, predictions)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auc=auc,
        baseline_accuracy=majority_class_accuracy(labels),
        positive_prediction_rate=float(np.mean(predictions)),
    )
