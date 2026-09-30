"""Default catalog of built-in and synthetic datasets."""

from __future__ import annotations

from functools import partial

from ml_playground.data.generators import (
    generate_blobs,
    generate_circles,
    generate_classification,
    generate_moons,
    generate_regression,
)
from ml_playground.data.loaders import load_sklearn_dataset
from ml_playground.data.registry import DatasetRegistry
from ml_playground.data.specifications import DatasetSpecification
from ml_playground.models.specifications import HyperparameterSpec, ProblemType


def _integer(name: str, default: int, minimum: int, maximum: int) -> HyperparameterSpec:
    return HyperparameterSpec(
        name=name,
        value_type=int,
        default=default,
        minimum=minimum,
        maximum=maximum,
        description=f"{name.replace('_', ' ').capitalize()}.",
    )


def _number(
    name: str,
    default: float,
    minimum: float,
    maximum: float,
    *,
    step: float | None = None,
) -> HyperparameterSpec:
    return HyperparameterSpec(
        name=name,
        value_type=float,
        default=default,
        minimum=minimum,
        maximum=maximum,
        step=step,
        description=f"{name.replace('_', ' ').capitalize()}.",
    )


def _validate_classification_configuration(parameters: dict[str, object]) -> None:
    n_features = int(parameters["n_features"])
    n_informative = int(parameters["n_informative"])
    n_redundant = int(parameters["n_redundant"])
    n_repeated = int(parameters["n_repeated"])
    n_classes = int(parameters["n_classes"])
    clusters = int(parameters["n_clusters_per_class"])
    if n_informative + n_redundant + n_repeated > n_features:
        raise ValueError(
            "n_informative + n_redundant + n_repeated cannot exceed n_features."
        )
    if n_classes * clusters > 2**n_informative:
        raise ValueError("n_classes * n_clusters_per_class cannot exceed 2**n_informative.")


def _validate_regression_configuration(parameters: dict[str, object]) -> None:
    if int(parameters["n_informative"]) > int(parameters["n_features"]):
        raise ValueError("n_informative cannot exceed n_features.")


def _validate_blobs_configuration(parameters: dict[str, object]) -> None:
    if int(parameters["centers"]) > int(parameters["n_samples"]):
        raise ValueError("centers cannot exceed n_samples.")


def create_default_dataset_registry() -> DatasetRegistry:
    """Create a registry containing the requested sklearn and synthetic datasets."""
    specifications = [
        DatasetSpecification(
            id=dataset_id,
            display_name=name,
            problem_type=ProblemType.CLASSIFICATION,
            factory=partial(load_sklearn_dataset, dataset_id=dataset_id, dataset_name=name),
            description=f"Built-in classification dataset provided by scikit-learn: {name}.",
        )
        for dataset_id, name in (
            ("iris", "Iris"),
            ("wine", "Wine"),
            ("breast_cancer", "Breast Cancer"),
            ("digits", "Digits"),
        )
    ]

    specifications.extend(
        [
            DatasetSpecification(
                id="make_classification",
                display_name="Synthetic Classification",
                problem_type=ProblemType.CLASSIFICATION,
                factory=generate_classification,
                parameters=(
                    _integer("n_samples", 300, 2, 5000),
                    _integer("n_features", 2, 2, 100),
                    _integer("n_informative", 2, 1, 100),
                    _integer("n_redundant", 0, 0, 100),
                    _integer("n_repeated", 0, 0, 100),
                    _integer("n_classes", 2, 2, 20),
                    _integer("n_clusters_per_class", 1, 1, 10),
                    _number("class_sep", 1.0, 0.0, 10.0),
                    _number("flip_y", 0.01, 0.0, 1.0),
                    _integer("random_state", 42, 0, 2**32 - 1),
                ),
                description="A configurable synthetic classification dataset.",
                configuration_validator=_validate_classification_configuration,
            ),
            DatasetSpecification(
                id="make_moons",
                display_name="Synthetic Moons",
                problem_type=ProblemType.CLASSIFICATION,
                factory=generate_moons,
                parameters=(
                    _integer("n_samples", 200, 2, 5000),
                    _number("noise", 0.15, 0.0, 5.0),
                    _integer("random_state", 42, 0, 2**32 - 1),
                ),
                description="Two interleaving half circles for binary classification.",
            ),
            DatasetSpecification(
                id="make_circles",
                display_name="Synthetic Circles",
                problem_type=ProblemType.CLASSIFICATION,
                factory=generate_circles,
                parameters=(
                    _integer("n_samples", 200, 2, 5000),
                    _number("noise", 0.05, 0.0, 5.0),
                    _number("factor", 0.5, 0.01, 0.99, step=0.01),
                    _integer("random_state", 42, 0, 2**32 - 1),
                ),
                description="Concentric circles for binary classification.",
            ),
            DatasetSpecification(
                id="make_blobs",
                display_name="Synthetic Blobs",
                problem_type=ProblemType.CLUSTERING,
                factory=generate_blobs,
                parameters=(
                    _integer("n_samples", 300, 2, 5000),
                    _integer("n_features", 2, 1, 100),
                    _integer("centers", 3, 1, 100),
                    _number("cluster_std", 1.0, 0.0, 10.0),
                    _integer("random_state", 42, 0, 2**32 - 1),
                ),
                description="Gaussian blobs with generated cluster labels.",
                configuration_validator=_validate_blobs_configuration,
            ),
            DatasetSpecification(
                id="make_regression",
                display_name="Synthetic Regression",
                problem_type=ProblemType.REGRESSION,
                factory=generate_regression,
                parameters=(
                    _integer("n_samples", 200, 2, 5000),
                    _integer("n_features", 5, 1, 100),
                    _integer("n_informative", 3, 1, 100),
                    _number("noise", 0.0, 0.0, 100.0),
                    _number("bias", 0.0, -1000.0, 1000.0),
                    _integer("random_state", 42, 0, 2**32 - 1),
                ),
                description="A configurable synthetic regression dataset.",
                configuration_validator=_validate_regression_configuration,
            ),
        ]
    )
    return DatasetRegistry(specifications)


DEFAULT_DATASET_REGISTRY = create_default_dataset_registry()
