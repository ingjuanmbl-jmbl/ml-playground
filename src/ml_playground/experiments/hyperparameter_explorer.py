"""One-parameter sensitivity exploration through the generic training contracts."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.data.registry import DatasetRegistry
from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.evaluation.clustering import evaluate_clustering
from ml_playground.evaluation.results import MetricAvailability
from ml_playground.models.catalog import DEFAULT_MODEL_REGISTRY
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import HyperparameterSpec, ProblemType, UNSET
from ml_playground.training.contracts import TrainingRequest, TrainingRunner
from ml_playground.training.runner import GenericTrainingRunner


@dataclass(frozen=True, slots=True)
class HyperparameterRun:
    """Evaluation and timing from one value of the selected hyperparameter."""

    value: Any
    metrics: dict[str, float | None]
    metric_availability: dict[str, MetricAvailability]
    training_seconds: float


@dataclass(frozen=True, slots=True)
class HyperparameterExploration:
    """Comparable runs produced by varying exactly one model parameter."""

    model_id: str
    model_name: str
    dataset_id: str
    dataset_name: str
    problem_type: ProblemType
    hyperparameter: str
    metric: str
    values: tuple[Any, ...]
    results: tuple[HyperparameterRun, ...]
    base_parameters: dict[str, Any]
    dataset_parameters: dict[str, Any]
    test_size: float
    random_state: int
    stratify: bool


CLASSIFICATION_METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")
CLUSTERING_METRICS = ("silhouette", "davies_bouldin", "calinski_harabasz")


def is_explorable(parameter: HyperparameterSpec) -> bool:
    """Return whether a parameter is a safe scalar control for a one-value-at-a-time sweep."""
    if parameter.name == "random_state":
        return False
    types = parameter.value_type if isinstance(parameter.value_type, tuple) else (parameter.value_type,)
    scalar_types = {int, float, str, bool, type(None)}
    if not set(types) <= scalar_types:
        return False
    if str in types:
        return parameter.choices is not None and all(
            isinstance(choice, str) for choice in parameter.choices
        )
    if bool in types:
        return parameter.choices is None or all(
            isinstance(choice, bool) for choice in parameter.choices
        )
    return int in types or float in types


class HyperparameterExplorer:
    """Run a controlled sensitivity exploration using existing runner/evaluator components."""

    def __init__(
        self,
        model_registry: ModelRegistry = DEFAULT_MODEL_REGISTRY,
        dataset_registry: DatasetRegistry = DEFAULT_DATASET_REGISTRY,
        training_runner: TrainingRunner | None = None,
    ) -> None:
        self.model_registry = model_registry
        self.dataset_registry = dataset_registry
        self.training_runner = training_runner or GenericTrainingRunner(
            model_registry=model_registry, dataset_registry=dataset_registry
        )

    def explore(
        self,
        *,
        dataset_id: str,
        model_id: str,
        hyperparameter: str,
        values: tuple[Any, ...] | list[Any],
        metric: str,
        dataset_parameters: dict[str, Any] | None = None,
        base_parameters: dict[str, Any] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> HyperparameterExploration:
        """Evaluate ordered parameter values with a fixed dataset, split, and random seed."""
        model = self.model_registry.get(model_id)
        dataset = self.dataset_registry.build(dataset_id, dataset_parameters or {})
        if dataset.problem_type is ProblemType.REGRESSION:
            raise ValueError("Hyperparameter Explorer currently supports classification and clustering.")
        if model.problem_type is not ProblemType.CLUSTERING and dataset.problem_type is not model.problem_type:
            raise ValueError(
                f"Dataset '{dataset_id}' is {dataset.problem_type.value}, but model '{model_id}' "
                f"requires {model.problem_type.value}."
            )

        parameter_by_name = {item.name: item for item in model.hyperparameters}
        parameter = parameter_by_name.get(hyperparameter)
        if parameter is None or not is_explorable(parameter):
            raise ValueError(f"Hyperparameter '{hyperparameter}' is not explorable for model '{model_id}'.")
        ordered_values = _ordered_values(parameter, values)
        if len(ordered_values) < 2:
            raise ValueError("Provide at least two distinct values to explore.")

        allowed_metrics = (
            CLUSTERING_METRICS
            if model.problem_type is ProblemType.CLUSTERING
            else CLASSIFICATION_METRICS
        )
        if metric not in allowed_metrics:
            raise ValueError(
                f"Metric '{metric}' is not available for {model.problem_type.value}; "
                f"choose from {allowed_metrics}."
            )

        supplied_base = dict(base_parameters or {})
        if "random_state" in parameter_by_name and "random_state" not in supplied_base:
            supplied_base["random_state"] = random_state
        resolved_base = model.validate_parameters(supplied_base)
        for item in model.hyperparameters:
            if item.name not in resolved_base and item.default is not UNSET:
                resolved_base[item.name] = item.default
        parameter.validate(ordered_values[0])
        requests: list[TrainingRequest] = []
        for value in ordered_values:
            parameter.validate(value)
            configured = dict(resolved_base)
            configured[hyperparameter] = value
            model.validate_parameters(configured)
            requests.append(
                TrainingRequest(
                    dataset_id=dataset_id,
                    model_id=model_id,
                    dataset_parameters=dict(dataset_parameters or {}),
                    model_parameters=configured,
                    test_size=test_size,
                    random_state=random_state,
                    stratify=stratify,
                )
            )

        run_results: list[HyperparameterRun] = []
        for value, request in zip(ordered_values, requests, strict=True):
            output = self.training_runner.run(request)
            evaluated = (
                evaluate_clustering(output)
                if model.problem_type is ProblemType.CLUSTERING
                else evaluate_classification(output)
            )
            run_results.append(
                HyperparameterRun(
                    value=value,
                    metrics=dict(evaluated.metrics),
                    metric_availability=dict(evaluated.metric_availability),
                    training_seconds=output.training_seconds,
                )
            )

        return HyperparameterExploration(
            model_id=model.id,
            model_name=model.display_name,
            dataset_id=dataset_id,
            dataset_name=dataset.dataset_name,
            problem_type=model.problem_type,
            hyperparameter=hyperparameter,
            metric=metric,
            values=ordered_values,
            results=tuple(run_results),
            base_parameters=resolved_base,
            dataset_parameters=dict(dataset.parameters),
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )


def _ordered_values(
    parameter: HyperparameterSpec, values: tuple[Any, ...] | list[Any]
) -> tuple[Any, ...]:
    candidates = tuple(values)
    if not candidates:
        raise ValueError("At least two values are required to explore a hyperparameter.")
    for value in candidates:
        parameter.validate(value)
    if len(set(candidates)) != len(candidates):
        raise ValueError("Exploration values must be distinct.")
    choices = parameter.choices
    if choices is not None:
        order = {value: index for index, value in enumerate(choices)}
        return tuple(sorted(candidates, key=lambda value: order[value]))
    if all(value is None or isinstance(value, (int, float)) for value in candidates):
        if any(isinstance(value, float) and not isfinite(value) for value in candidates):
            raise ValueError("Exploration values must be finite.")
        return tuple(sorted(candidates, key=lambda value: (value is None, value or 0)))
    return candidates


__all__ = [
    "CLASSIFICATION_METRICS",
    "CLUSTERING_METRICS",
    "HyperparameterExploration",
    "HyperparameterExplorer",
    "HyperparameterRun",
    "is_explorable",
]
