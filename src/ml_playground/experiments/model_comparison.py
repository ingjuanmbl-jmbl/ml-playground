"""Comparable supervised runs executed through the generic runner and evaluator."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.data.registry import DatasetRegistry
from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.evaluation.results import MetricAvailability
from ml_playground.models.catalog import DEFAULT_MODEL_REGISTRY
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import ProblemType, UNSET
from ml_playground.training.contracts import TrainingRequest, TrainingRunner, TrainingSplit
from ml_playground.training.runner import GenericTrainingRunner
from ml_playground.training.splitting import create_training_split


@dataclass(frozen=True, slots=True)
class ComparedModelResult:
    """Evaluation metrics and effective configuration for a single compared model."""

    model_id: str
    model_name: str
    configuration: Mapping[str, Any]
    metrics: Mapping[str, float | None]
    metric_availability: Mapping[str, MetricAvailability]
    training_seconds: float


@dataclass(frozen=True, slots=True)
class ModelComparison:
    """Results from registered classifiers evaluated on one shared test split."""

    dataset_id: str
    dataset_name: str
    dataset_parameters: Mapping[str, Any]
    split: TrainingSplit
    test_size: float
    random_state: int
    stratify: bool
    models: tuple[ComparedModelResult, ...]
    timestamp: datetime


class ModelComparisonService:
    """Coordinate identical-data classifier runs without implementing training itself."""

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

    def compare(
        self,
        *,
        dataset_id: str,
        model_ids: tuple[str, ...] | list[str],
        dataset_parameters: Mapping[str, Any] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True,
    ) -> ModelComparison:
        """Train/evaluate selected registered classifiers against one fixed split."""
        selected_ids = tuple(model_ids)
        if len(selected_ids) < 2:
            raise ValueError("Select at least two distinct classification models.")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("Each model can only be selected once.")
        if not isinstance(stratify, bool):
            raise TypeError("stratify must be a bool.")

        dataset_specification = self.dataset_registry.get(dataset_id)
        dataset_parameters_used = dataset_specification.validate_parameters(
            dict(dataset_parameters or {})
        )
        dataset = self.dataset_registry.build(dataset_id, dataset_parameters_used)
        if dataset.problem_type is not ProblemType.CLASSIFICATION or dataset.y is None:
            raise ValueError("Model Comparison requires a classification dataset with a target.")
        specifications = [self.model_registry.get(model_id) for model_id in selected_ids]
        incompatible = [model.display_name for model in specifications if model.problem_type is not ProblemType.CLASSIFICATION]
        if incompatible:
            raise ValueError("Model Comparison only supports classification models.")

        actual_stratify = bool(stratify)
        try:
            split = create_training_split(
                n_samples=dataset.n_observations,
                test_size=test_size,
                random_state=random_state,
                target=dataset.y,
                stratified=actual_stratify,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Unable to create shared train/test split: {error}") from error

        results: list[ComparedModelResult] = []
        for specification in specifications:
            model_parameters = {
                parameter.name: parameter.default
                for parameter in specification.hyperparameters
                if parameter.default is not UNSET
            }
            if any(parameter.name == "random_state" for parameter in specification.hyperparameters):
                model_parameters["random_state"] = random_state
            resolved = specification.validate_parameters(model_parameters)
            request = TrainingRequest(
                dataset_id=dataset_id,
                model_id=specification.id,
                dataset_parameters=dataset_parameters_used,
                model_parameters=resolved,
                test_size=test_size,
                random_state=random_state,
                stratify=actual_stratify,
                split=split,
            )
            output = self.training_runner.run(request)
            evaluated = evaluate_classification(output)
            results.append(
                ComparedModelResult(
                    model_id=specification.id,
                    model_name=specification.display_name,
                    configuration=dict(evaluated.configuration),
                    metrics=dict(evaluated.metrics),
                    metric_availability=dict(evaluated.metric_availability),
                    training_seconds=output.training_seconds,
                )
            )

        return ModelComparison(
            dataset_id=dataset_id,
            dataset_name=dataset.dataset_name,
            dataset_parameters=dataset_parameters_used,
            split=split,
            test_size=test_size,
            random_state=random_state,
            stratify=actual_stratify,
            models=tuple(results),
            timestamp=datetime.now(timezone.utc),
        )


__all__ = ["ComparedModelResult", "ModelComparison", "ModelComparisonService"]
