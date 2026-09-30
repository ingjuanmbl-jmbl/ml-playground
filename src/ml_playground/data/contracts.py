"""Shared, framework-independent dataset contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd

from ml_playground.models.specifications import ProblemType


@dataclass(frozen=True, slots=True)
class Dataset:
    """A tabular feature matrix, optional target, and provenance metadata.

    ``X`` is represented as a DataFrame so feature names remain attached to columns. A target, when
    present, is a separate Series and is never included in ``X``.
    """

    X: pd.DataFrame
    y: pd.Series | None
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...] | None
    dataset_name: str
    problem_type: ProblemType
    metadata: Mapping[str, Any] = field(default_factory=dict)
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.X, pd.DataFrame):
            raise TypeError("Dataset X must be a pandas DataFrame.")
        if self.y is not None and not isinstance(self.y, pd.Series):
            raise TypeError("Dataset y must be a pandas Series or None.")
        if self.y is not None and len(self.y) != len(self.X):
            raise ValueError("Dataset X and y must contain the same number of observations.")
        if len(self.feature_names) != self.X.shape[1]:
            raise ValueError("feature_names must contain one name for each column in X.")
        if tuple(map(str, self.X.columns)) != self.feature_names:
            raise ValueError("feature_names must match the columns of X in order.")
        if not self.dataset_name.strip():
            raise ValueError("dataset_name cannot be empty.")

    @property
    def n_observations(self) -> int:
        """Number of rows in the feature matrix."""
        return int(self.X.shape[0])

    @property
    def n_features(self) -> int:
        """Number of feature columns, excluding the target."""
        return int(self.X.shape[1])
