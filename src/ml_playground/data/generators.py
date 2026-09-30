"""Reproducible synthetic dataset factories."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.datasets import (
    make_blobs as sklearn_make_blobs,
    make_circles as sklearn_make_circles,
    make_classification as sklearn_make_classification,
    make_moons as sklearn_make_moons,
    make_regression as sklearn_make_regression,
)

from ml_playground.data.contracts import Dataset
from ml_playground.models.specifications import ProblemType


def _dataset_from_arrays(
    *,
    X: Any,
    y: Any,
    dataset_name: str,
    problem_type: ProblemType,
    parameters: dict[str, Any],
    target_names: tuple[str, ...] | None = None,
) -> Dataset:
    feature_names = tuple(f"feature_{index}" for index in range(X.shape[1]))
    return Dataset(
        X=pd.DataFrame(X, columns=feature_names),
        y=pd.Series(y, name="target"),
        feature_names=feature_names,
        target_names=target_names,
        dataset_name=dataset_name,
        problem_type=problem_type,
        metadata={"source": "scikit-learn synthetic generator"},
        parameters=dict(parameters),
    )


def _class_names(n_classes: int) -> tuple[str, ...]:
    return tuple(f"class_{index}" for index in range(n_classes))


def generate_classification(
    *,
    n_samples: int,
    n_features: int,
    n_informative: int,
    n_redundant: int,
    n_repeated: int,
    n_classes: int,
    n_clusters_per_class: int,
    class_sep: float,
    flip_y: float,
    random_state: int,
) -> Dataset:
    """Create a classification dataset with a recorded, reproducible configuration."""
    parameters = locals().copy()
    X, y = sklearn_make_classification(**parameters)
    return _dataset_from_arrays(
        X=X,
        y=y,
        dataset_name="Synthetic Classification",
        problem_type=ProblemType.CLASSIFICATION,
        parameters=parameters,
        target_names=_class_names(n_classes),
    )


def generate_moons(*, n_samples: int, noise: float, random_state: int) -> Dataset:
    """Create a two-moons binary classification dataset."""
    parameters = locals().copy()
    X, y = sklearn_make_moons(**parameters)
    return _dataset_from_arrays(
        X=X,
        y=y,
        dataset_name="Synthetic Moons",
        problem_type=ProblemType.CLASSIFICATION,
        parameters=parameters,
        target_names=_class_names(2),
    )


def generate_circles(
    *, n_samples: int, noise: float, factor: float, random_state: int
) -> Dataset:
    """Create a concentric-circles binary classification dataset."""
    parameters = locals().copy()
    X, y = sklearn_make_circles(**parameters)
    return _dataset_from_arrays(
        X=X,
        y=y,
        dataset_name="Synthetic Circles",
        problem_type=ProblemType.CLASSIFICATION,
        parameters=parameters,
        target_names=_class_names(2),
    )


def generate_blobs(
    *, n_samples: int, n_features: int, centers: int, cluster_std: float, random_state: int
) -> Dataset:
    """Create a labeled blob dataset intended for clustering exploration."""
    parameters = locals().copy()
    X, y = sklearn_make_blobs(**parameters)
    return _dataset_from_arrays(
        X=X,
        y=y,
        dataset_name="Synthetic Blobs",
        problem_type=ProblemType.CLUSTERING,
        parameters=parameters,
        target_names=tuple(f"cluster_{index}" for index in range(centers)),
    )


def generate_regression(
    *,
    n_samples: int,
    n_features: int,
    n_informative: int,
    noise: float,
    bias: float,
    random_state: int,
) -> Dataset:
    """Create a regression dataset without fitting a model."""
    parameters = locals().copy()
    X, y = sklearn_make_regression(**parameters)
    return _dataset_from_arrays(
        X=X,
        y=y,
        dataset_name="Synthetic Regression",
        problem_type=ProblemType.REGRESSION,
        parameters=parameters,
    )
