"""Clustering metrics computed from feature observations and fitted cluster labels."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

from ml_playground.evaluation.results import ClusteringResult, MetricAvailability
from ml_playground.models.specifications import ProblemType
from ml_playground.training.contracts import TrainingOutput

METRIC_FUNCTIONS = {
    "silhouette": silhouette_score,
    "davies_bouldin": davies_bouldin_score,
    "calinski_harabasz": calinski_harabasz_score,
}


def evaluate_clustering(output: TrainingOutput) -> ClusteringResult:
    """Evaluate fitted clustering labels against the observations used for fitting.

    The original target is never accessed. When the model pipeline scales features, metrics use
    the same transformed feature space seen by the estimator.
    """
    if output.X_used is None:
        raise ValueError("Clustering evaluation requires X_used observations.")
    if output.cluster_labels is None:
        raise ValueError("Clustering evaluation requires cluster_labels.")
    preprocessor = output.trained_model.named_steps["preprocessing"]
    transform = getattr(preprocessor, "transform", None)
    X = transform(output.X_used) if callable(transform) else output.X_used.to_numpy()
    values = np.asarray(X, dtype=float)
    labels = np.asarray(output.cluster_labels)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Clustering observations must be a non-empty 2D feature matrix.")
    if not np.isfinite(values).all():
        raise ValueError("Clustering observations must contain finite numeric values.")
    if labels.ndim != 1:
        raise ValueError("cluster_labels must be one-dimensional.")
    if labels.size != values.shape[0]:
        raise ValueError("cluster_labels and X_used must have the same number of observations.")
    if labels.size == 0 or _contains_invalid_labels(labels):
        raise ValueError("cluster_labels must contain valid, non-missing values.")

    unique_count = int(np.unique(labels).size)
    availability: dict[str, MetricAvailability] = {}
    metrics: dict[str, float | None] = {}
    if unique_count < 2:
        invalid_reason = "Clustering metrics require at least two distinct clusters."
    elif unique_count >= labels.size:
        invalid_reason = "Clustering metrics require fewer clusters than observations."
    else:
        invalid_reason = ""

    for name, function in METRIC_FUNCTIONS.items():
        if invalid_reason:
            metrics[name] = None
            availability[name] = MetricAvailability(False, invalid_reason)
            continue
        try:
            score = float(function(values, labels))
            if not np.isfinite(score):
                raise ValueError("metric returned a non-finite value")
        except (ValueError, TypeError) as error:
            metrics[name] = None
            availability[name] = MetricAvailability(False, str(error))
        else:
            metrics[name] = score
            availability[name] = MetricAvailability(True)

    configuration = {
        "dataset_id": output.configuration.dataset_id,
        "model_id": output.configuration.model_id,
        "dataset_parameters": dict(output.configuration.dataset_parameters),
        "model_parameters": dict(output.configuration.model_parameters),
        "split_random_state": output.configuration.split_random_state,
        "split_performed": output.configuration.split_performed,
    }
    return ClusteringResult(
        model_id=output.configuration.model_id,
        model_name=str(output.metadata.get("model_name", output.configuration.model_id)),
        problem_type=ProblemType.CLUSTERING,
        configuration=configuration,
        training_seconds=output.training_seconds,
        timestamp=_timestamp(output.metadata.get("started_at")),
        metadata=dict(output.metadata),
        cluster_labels=labels.copy(),
        metrics=metrics,
        centroids=None if output.centroids is None else np.asarray(output.centroids).copy(),
        n_clusters=unique_count,
        metric_availability=availability,
    )


def _contains_invalid_labels(labels: np.ndarray) -> bool:
    try:
        return bool(np.asarray([label is None for label in labels]).any()) or bool(
            np.asarray(labels != labels).any()
        )
    except (TypeError, ValueError):
        return True


def _timestamp(value: object) -> datetime:
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            pass
    return datetime.now(timezone.utc)


__all__ = ["evaluate_clustering"]
