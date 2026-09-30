"""Structured, framework-independent training request and output contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from typing import Mapping, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass(frozen=True, slots=True)
class TrainingSplit:
    """Positional train/test indices and the settings used to create them."""

    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    n_samples: int
    test_size: float
    random_state: int
    stratified: bool

    def __post_init__(self) -> None:
        if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, int):
            raise TypeError("n_samples must be an integer.")
        if isinstance(self.test_size, bool) or not isinstance(self.test_size, (int, float)):
            raise TypeError("test_size must be a number between 0 and 1.")
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be greater than 0 and less than 1.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer.")
        if not 0 <= self.random_state <= 2**32 - 1:
            raise ValueError("random_state must be between 0 and 2**32 - 1.")
        if not isinstance(self.stratified, bool):
            raise TypeError("stratified must be a bool.")
        if not isinstance(self.train_indices, tuple) or not isinstance(self.test_indices, tuple):
            raise TypeError("Training split indices must be tuples.")
        if any(
            isinstance(index, bool) or not isinstance(index, int)
            for index in (*self.train_indices, *self.test_indices)
        ):
            raise TypeError("Training split indices must be integers.")
        train = set(self.train_indices)
        test = set(self.test_indices)
        expected = set(range(self.n_samples))
        if self.n_samples < 2 or not train or not test:
            raise ValueError("A training split requires non-empty train and test partitions.")
        if len(train) != len(self.train_indices) or len(test) != len(self.test_indices):
            raise ValueError("Training split indices must not contain duplicates.")
        if train & test or train | test != expected:
            raise ValueError("Training split indices must partition all observations exactly once.")


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """Dataset/model identifiers and execution options for one training run."""

    dataset_id: str
    model_id: str
    dataset_parameters: Mapping[str, object] = field(default_factory=dict)
    model_parameters: Mapping[str, object] = field(default_factory=dict)
    test_size: float = 0.2
    random_state: int = 42
    stratify: bool = True
    split: TrainingSplit | None = None

    def __post_init__(self) -> None:
        if not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be empty.")
        if not self.model_id.strip():
            raise ValueError("model_id cannot be empty.")
        if isinstance(self.test_size, bool) or not isinstance(self.test_size, (int, float)):
            raise TypeError("test_size must be a number between 0 and 1.")
        if not isfinite(self.test_size) or not 0 < self.test_size < 1:
            raise ValueError("test_size must be greater than 0 and less than 1.")
        if isinstance(self.random_state, bool) or not isinstance(self.random_state, int):
            raise TypeError("random_state must be an integer.")
        if not 0 <= self.random_state <= 2**32 - 1:
            raise ValueError("random_state must be between 0 and 2**32 - 1.")
        if not isinstance(self.stratify, bool):
            raise TypeError("stratify must be a bool.")
        if self.split is not None and not isinstance(self.split, TrainingSplit):
            raise TypeError("split must be a TrainingSplit or None.")


@dataclass(frozen=True, slots=True)
class TrainingConfiguration:
    """Resolved, reproducible configuration actually used for a run."""

    dataset_id: str
    model_id: str
    dataset_parameters: Mapping[str, object]
    model_parameters: Mapping[str, object]
    test_size: float
    split_random_state: int
    estimator_random_state: object | None
    stratified: bool
    split_performed: bool = True
    split: TrainingSplit | None = None


@dataclass(frozen=True, slots=True)
class TrainingOutput:
    """Fitted pipeline, held-out outputs, split data, timing, and provenance."""

    trained_model: Pipeline
    predictions: np.ndarray | None
    probabilities: np.ndarray | None
    scores: np.ndarray | None
    training_seconds: float
    metadata: Mapping[str, object]
    configuration: TrainingConfiguration
    X_train: pd.DataFrame | None = None
    X_test: pd.DataFrame | None = None
    y_train: pd.Series | None = None
    y_test: pd.Series | None = None
    feature_importances: object | None = None
    coefficients: object | None = None
    cluster_labels: np.ndarray | None = None
    noise_mask: np.ndarray | None = None
    centroids: np.ndarray | None = None
    X_used: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        if self.training_seconds < 0:
            raise ValueError("Training time cannot be negative.")


@runtime_checkable
class TrainingRunner(Protocol):
    """Framework-independent interface implemented by a training runner."""

    def run(self, request: TrainingRequest) -> TrainingOutput:
        """Fit the requested estimator and return its structured outputs."""
        ...
