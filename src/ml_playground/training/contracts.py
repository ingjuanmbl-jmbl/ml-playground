"""Input and output data contracts for model training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from ml_playground.models.specifications import ModelSpecification


@dataclass(frozen=True, slots=True)
class TrainingRequest:
    """Data and configuration supplied to a runner, independent of any UI framework."""

    specification: ModelSpecification
    parameters: Mapping[str, Any]
    X_train: Any
    y_train: Any | None = None
    X_predict: Any | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingOutput:
    """Structured output of one fit/predict run."""

    trained_model: Any
    predictions: Any | None
    probabilities: Any | None
    scores: Any | None
    training_seconds: float
    metadata: Mapping[str, Any]
    configuration: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.training_seconds < 0:
            raise ValueError("Training time cannot be negative.")


@runtime_checkable
class TrainingRunner(Protocol):
    """Framework-independent interface implemented by a training runner."""

    def run(self, request: TrainingRequest) -> TrainingOutput:
        """Fit the requested estimator and return its structured outputs."""
        ...
