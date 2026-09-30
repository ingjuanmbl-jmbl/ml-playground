"""Load named datasets supplied by scikit-learn."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn.datasets import (
    load_breast_cancer,
    load_digits,
    load_iris,
    load_wine,
)

from ml_playground.data.contracts import Dataset
from ml_playground.models.specifications import ProblemType

_SKLEARN_LOADERS: dict[str, Callable[..., Any]] = {
    "iris": load_iris,
    "wine": load_wine,
    "breast_cancer": load_breast_cancer,
    "digits": load_digits,
}


def load_sklearn_dataset(dataset_id: str, dataset_name: str) -> Dataset:
    """Load one registered sklearn classification dataset with target kept separate."""
    try:
        loader = _SKLEARN_LOADERS[dataset_id]
    except KeyError as error:
        raise KeyError(f"Unsupported scikit-learn dataset '{dataset_id}'.") from error

    bunch = loader(as_frame=True)
    feature_names = tuple(map(str, bunch.feature_names))
    X = pd.DataFrame(bunch.data, columns=feature_names)
    y = pd.Series(bunch.target, name="target")
    raw_target_names = getattr(bunch, "target_names", None)
    target_names = tuple(map(str, raw_target_names)) if raw_target_names is not None else None
    description = getattr(bunch, "DESCR", "")
    return Dataset(
        X=X,
        y=y,
        feature_names=feature_names,
        target_names=target_names,
        dataset_name=dataset_name,
        problem_type=ProblemType.CLASSIFICATION,
        metadata={
            "source": "scikit-learn",
            "source_id": dataset_id,
            "description": description.splitlines()[0] if description else "",
        },
        parameters={"as_frame": True},
    )
