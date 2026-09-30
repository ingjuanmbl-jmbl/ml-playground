"""Classification metrics computed from held-out training outputs only."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_playground.evaluation.results import (
    ClassificationResult,
    MetricAvailability,
)
from ml_playground.models.specifications import ProblemType
from ml_playground.training.contracts import TrainingOutput

AVERAGING_STRATEGY: Literal["macro"] = "macro"
ZERO_DIVISION: int = 0


def evaluate_classification(output: TrainingOutput) -> ClassificationResult:
    """Evaluate held-out labels/predictions without fitting or reading training data.

    Precision, recall, and F1 use macro averaging so every class contributes equally.
    Undefined per-class divisions are set to zero (scikit-learn ``zero_division=0``).
    Multiclass ROC-AUC uses one-vs-rest with macro averaging.
    """
    if output.y_test is None:
        raise ValueError("Classification evaluation requires y_test.")
    if output.predictions is None:
        raise ValueError("Classification evaluation requires predictions.")

    y_true = _as_vector(output.y_test, "y_test")
    y_pred = _as_vector(output.predictions, "predictions")
    if y_true.size == 0 or y_pred.size == 0:
        raise ValueError("y_test and predictions cannot be empty.")
    if y_true.size != y_pred.size:
        raise ValueError("y_test and predictions must have the same length.")
    if _contains_invalid_labels(y_true) or _contains_invalid_labels(y_pred):
        raise ValueError("y_test and predictions must contain valid, non-missing labels.")

    labels = tuple(np.unique(np.concatenate((y_true, y_pred))).tolist())
    n_classes = len(np.unique(y_true))
    accuracy = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(
        y_true, y_pred, labels=list(labels), average=AVERAGING_STRATEGY,
        zero_division=ZERO_DIVISION,
    ))
    recall = float(recall_score(
        y_true, y_pred, labels=list(labels), average=AVERAGING_STRATEGY,
        zero_division=ZERO_DIVISION,
    ))
    f1 = float(f1_score(
        y_true, y_pred, labels=list(labels), average=AVERAGING_STRATEGY,
        zero_division=ZERO_DIVISION,
    ))
    # Construct by the explicit label order. Besides making ordering unambiguous, this avoids
    # scikit-learn's single-observed-label warning for valid one-class test splits.
    label_indices = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred, strict=True):
        matrix[label_indices[actual], label_indices[predicted]] += 1

    auc, auc_availability = _calculate_roc_auc(
        y_true,
        labels=labels,
        probabilities=output.probabilities,
        scores=output.scores,
    )
    metrics: dict[str, float | None] = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }
    availability = {
        "accuracy": MetricAvailability(True),
        "precision": MetricAvailability(True),
        "recall": MetricAvailability(True),
        "f1": MetricAvailability(True),
        "confusion_matrix": MetricAvailability(True),
        "roc_auc": auc_availability,
    }
    configuration = {
        "dataset_id": output.configuration.dataset_id,
        "model_id": output.configuration.model_id,
        "dataset_parameters": dict(output.configuration.dataset_parameters),
        "model_parameters": dict(output.configuration.model_parameters),
        "test_size": output.configuration.test_size,
        "split_random_state": output.configuration.split_random_state,
    }
    return ClassificationResult(
        model_id=output.configuration.model_id,
        model_name=str(output.metadata.get("model_name", output.configuration.model_id)),
        problem_type=ProblemType.CLASSIFICATION,
        configuration=configuration,
        training_seconds=output.training_seconds,
        timestamp=_timestamp(output.metadata.get("started_at")),
        metadata={key: value for key, value in output.metadata.items()},
        predictions=y_pred.copy(),
        probabilities=_copy_optional(output.probabilities),
        scores=_copy_optional(output.scores),
        confusion_matrix=matrix,
        feature_importances=output.feature_importances,
        coefficients=output.coefficients,
        metrics=metrics,
        class_labels=labels,
        averaging_strategy=AVERAGING_STRATEGY,
        metric_availability=availability,
    )


def _calculate_roc_auc(
    y_true: np.ndarray,
    *,
    labels: tuple[Any, ...],
    probabilities: np.ndarray | None,
    scores: np.ndarray | None,
) -> tuple[float | None, MetricAvailability]:
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        return None, MetricAvailability(False, "ROC-AUC requires at least two classes in y_test.")
    source = probabilities if probabilities is not None else scores
    source_name = "predict_proba" if probabilities is not None else "decision_function"
    if source is None:
        return None, MetricAvailability(False, "No probabilities or decision scores were provided.")

    try:
        values = np.asarray(source)
    except (TypeError, ValueError) as error:
        return None, MetricAvailability(False, f"{source_name} output has an incompatible shape: {error}")
    if values.ndim not in (1, 2) or values.shape[0] != y_true.size:
        return None, MetricAvailability(False, f"{source_name} output has an incompatible shape.")
    if not np.issubdtype(values.dtype, np.number) or not np.isfinite(values).all():
        return None, MetricAvailability(False, f"{source_name} output must contain finite numeric values.")
    try:
        if n_classes == 2:
            if values.ndim == 2:
                if values.shape[1] != 2:
                    raise ValueError("binary scores must contain exactly two class columns")
                positive_scores = values[:, 1]
            else:
                positive_scores = values
            auc = roc_auc_score(y_true, positive_scores)
        else:
            if values.ndim != 2 or values.shape[1] != len(labels):
                raise ValueError("multiclass scores must have one column per class")
            auc = roc_auc_score(
                y_true,
                values,
                labels=list(labels),
                multi_class="ovr",
                average="macro",
            )
    except (ValueError, TypeError) as error:
        return None, MetricAvailability(False, f"{source_name} cannot be used for ROC-AUC: {error}")
    return float(auc), MetricAvailability(True)


def _as_vector(values: Any, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return array


def _contains_invalid_labels(values: np.ndarray) -> bool:
    try:
        return bool(np.asarray([value is None for value in values]).any()) or bool(
            np.asarray(values != values).any()
        )
    except (TypeError, ValueError):
        return True


def _copy_optional(values: np.ndarray | None) -> np.ndarray | None:
    return None if values is None else np.asarray(values).copy()


def _timestamp(value: object) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


__all__ = ["AVERAGING_STRATEGY", "evaluate_classification"]
