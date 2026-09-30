"""Registered clustering model specifications."""

from __future__ import annotations

from sklearn.cluster import KMeans

from ml_playground.models.specifications import (
    HyperparameterSpec,
    ModelCapability,
    ModelSpecification,
    ProblemType,
)


def kmeans_specification() -> ModelSpecification:
    """Return the declarative specification for scikit-learn K-Means."""
    return ModelSpecification(
        id="kmeans",
        display_name="K-Means",
        problem_type=ProblemType.CLUSTERING,
        estimator_factory=KMeans,
        hyperparameters=(
            HyperparameterSpec(
                "n_clusters", int, default=3, minimum=2, maximum=100,
                description="Number of clusters to find.",
            ),
            HyperparameterSpec(
                "init", str, default="k-means++", choices=("k-means++", "random"),
                description="Initialization method for cluster centers.",
            ),
            HyperparameterSpec(
                "n_init", int, default=10, minimum=1, maximum=100,
                description="Number of centroid initializations; keep an integer for explicit reproducibility.",
            ),
            HyperparameterSpec(
                "max_iter", int, default=300, minimum=1, maximum=1000,
                description="Maximum number of iterations per initialization.",
            ),
            HyperparameterSpec(
                "random_state", int, default=42, minimum=0, maximum=2**32 - 1,
                description="Seed used for centroid initialization.",
            ),
        ),
        capabilities=frozenset(
            {ModelCapability.CLUSTER_LABELS, ModelCapability.CENTROIDS}
        ),
        requires_scaling=True,
        description=(
            "Partitions observations into a requested number of clusters by iteratively assigning "
            "points to nearby centroids. Features are standardized within the training pipeline."
        ),
    )


__all__ = ["kmeans_specification"]
