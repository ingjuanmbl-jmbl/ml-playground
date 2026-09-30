"""Model evaluation and experiment result contracts."""

from ml_playground.evaluation.results import (
    ClassificationResult,
    ClusteringResult,
    ExperimentResult,
    MetricAvailability,
)
from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.evaluation.clustering import evaluate_clustering

__all__ = [
    "ClassificationResult",
    "ClusteringResult",
    "ExperimentResult",
    "MetricAvailability",
    "evaluate_classification",
    "evaluate_clustering",
]

