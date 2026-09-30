"""Framework-independent experiment result contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping

from ml_playground.models.specifications import ProblemType


@dataclass(frozen=True, slots=True, kw_only=True)
class ExperimentResult:
    """Fields shared by results from any supported problem type."""

    model_id: str
    model_name: str
    problem_type: ProblemType
    configuration: Mapping[str, Any]
    training_seconds: float
    timestamp: datetime
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id.strip() or not self.model_name.strip():
            raise ValueError("Experiment results require a model id and visible name.")
        if self.training_seconds < 0:
            raise ValueError("Training time cannot be negative.")


@dataclass(frozen=True, slots=True)
class MetricAvailability:
    """Availability of one metric, including a reason when it cannot be computed."""

    available: bool
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available and self.reason is not None:
            raise ValueError("Available metrics cannot have an unavailability reason.")
        if not self.available and not self.reason:
            raise ValueError("Unavailable metrics require a reason.")


@dataclass(frozen=True, slots=True)
class ClassificationResult(ExperimentResult):
    """Classification-specific predictions, scores, and metrics."""

    predictions: Any
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    probabilities: Any | None = None
    scores: Any | None = None
    confusion_matrix: Any | None = None
    feature_importances: Any | None = None
    coefficients: Any | None = None
    class_labels: tuple[Any, ...] = ()
    averaging_strategy: str = "macro"
    metric_availability: Mapping[str, MetricAvailability] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ExperimentResult.__post_init__(self)
        if self.problem_type is not ProblemType.CLASSIFICATION:
            raise ValueError("ClassificationResult requires classification problem_type.")


@dataclass(frozen=True, slots=True)
class ClusteringResult(ExperimentResult):
    """Clustering labels, a per-observation noise mask, optional geometry, and metrics."""

    cluster_labels: Any
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    noise_mask: Any | None = None
    centroids: Any | None = None
    n_clusters: int = 0
    metric_availability: Mapping[str, MetricAvailability] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ExperimentResult.__post_init__(self)
        if self.problem_type is not ProblemType.CLUSTERING:
            raise ValueError("ClusteringResult requires clustering problem_type.")
        if self.n_clusters < 0:
            raise ValueError("n_clusters cannot be negative.")
