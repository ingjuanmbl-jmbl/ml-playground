"""Model evaluation and experiment result contracts."""

from ml_playground.evaluation.results import (
    ClassificationResult,
    ClusteringResult,
    ExperimentResult,
    MetricAvailability,
)
from ml_playground.evaluation.classification import evaluate_classification

__all__ = [
    "ClassificationResult",
    "ClusteringResult",
    "ExperimentResult",
    "MetricAvailability",
    "evaluate_classification",
]

