"""Generic training orchestration driven by problem type and declared capabilities."""

from __future__ import annotations

from datetime import datetime, timezone
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.data.registry import DatasetRegistry
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import ModelCapability, ProblemType
from ml_playground.preprocessing.pipelines import ESTIMATOR_STEP, build_pipeline
from ml_playground.training.contracts import (
    TrainingConfiguration,
    TrainingOutput,
    TrainingRequest,
    TrainingRunner,
)


class GenericTrainingRunner:
    """Train registered estimators without algorithm-specific branches.

    Supervised tasks split features and target before fitting. Clustering fits the full feature
    matrix without reading its optional target and obtains outputs through declared capabilities.
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        dataset_registry: DatasetRegistry = DEFAULT_DATASET_REGISTRY,
    ) -> None:
        self._model_registry = model_registry
        self._dataset_registry = dataset_registry

    def run(self, request: TrainingRequest) -> TrainingOutput:
        """Validate, fit, and return a structured result for the requested problem type."""
        dataset = self._dataset_registry.build(request.dataset_id, request.dataset_parameters)
        specification = self._model_registry.get(request.model_id)
        clustering_uses_features_only = (
            specification.problem_type is ProblemType.CLUSTERING
            and dataset.problem_type is not ProblemType.REGRESSION
        )
        if dataset.problem_type is not specification.problem_type and not clustering_uses_features_only:
            raise ValueError(
                f"Dataset '{request.dataset_id}' is {dataset.problem_type.value}, but model "
                f"'{request.model_id}' requires {specification.problem_type.value}."
            )
        if specification.problem_type is not ProblemType.CLUSTERING and dataset.y is None:
            raise ValueError(f"Dataset '{request.dataset_id}' has no target for supervised training.")

        model_parameters = dict(request.model_parameters)
        declared_parameters = {parameter.name for parameter in specification.hyperparameters}
        # The request seed always controls the split. For an estimator that declares a
        # random_state hyperparameter, reuse that seed only when the caller did not set one.
        if "random_state" in declared_parameters and "random_state" not in model_parameters:
            model_parameters["random_state"] = request.random_state
        resolved_parameters = specification.validate_parameters(model_parameters)
        if (
            specification.problem_type is ProblemType.CLUSTERING
            and "n_clusters" in resolved_parameters
            and resolved_parameters["n_clusters"] > dataset.n_observations
        ):
            raise ValueError(
                "n_clusters cannot exceed the number of observations in the dataset."
            )
        estimator_random_state = (
            resolved_parameters.get("random_state") if "random_state" in declared_parameters else None
        )

        pipeline = build_pipeline(specification, resolved_parameters)
        if specification.problem_type is ProblemType.CLUSTERING:
            started_at = datetime.now(timezone.utc)
            started = perf_counter()
            # Unsupervised fit receives X alone: no split, target access, or target-derived state.
            pipeline.fit(dataset.X)
            training_seconds = perf_counter() - started
            final_estimator = pipeline.named_steps[ESTIMATOR_STEP]
            cluster_labels = self._read_attribute(
                final_estimator,
                specification.capabilities,
                ModelCapability.CLUSTER_LABELS,
                "labels_",
            )
            centroids = self._read_attribute(
                final_estimator,
                specification.capabilities,
                ModelCapability.CENTROIDS,
                "cluster_centers_",
            )
            preprocessor = pipeline.named_steps["preprocessing"]
            inverse_transform = getattr(preprocessor, "inverse_transform", None)
            if cluster_labels is not None:
                cluster_labels = np.asarray(cluster_labels)
            if centroids is not None:
                centroids = np.asarray(centroids)
                if callable(inverse_transform):
                    centroids = np.asarray(inverse_transform(centroids))
            centroids_are_original_scale = (
                isinstance(preprocessor, str) and preprocessor == "passthrough"
                or callable(inverse_transform)
            )
            configuration = TrainingConfiguration(
                dataset_id=request.dataset_id,
                model_id=request.model_id,
                dataset_parameters=dict(dataset.parameters),
                model_parameters=resolved_parameters,
                test_size=request.test_size,
                split_random_state=request.random_state,
                estimator_random_state=estimator_random_state,
                stratified=False,
                split_performed=False,
            )
            metadata: dict[str, object] = {
                "dataset_name": dataset.dataset_name,
                "dataset_metadata": dict(dataset.metadata),
                "dataset_problem_type": dataset.problem_type.value,
                "model_name": specification.display_name,
                "n_observations": dataset.n_observations,
                "n_features": dataset.n_features,
                "target_used": False,
                "train_test_split": False,
                "centroids_space": (
                    "original feature scale"
                    if centroids_are_original_scale
                    else "preprocessed feature space"
                ),
                "started_at": started_at.isoformat(),
            }
            return TrainingOutput(
                trained_model=pipeline,
                predictions=None,
                probabilities=None,
                scores=None,
                training_seconds=training_seconds,
                metadata=metadata,
                configuration=configuration,
                cluster_labels=cluster_labels,
                centroids=centroids,
                X_used=dataset.X.copy(),
            )

        stratify_target = (
            dataset.y
            if specification.problem_type is ProblemType.CLASSIFICATION and request.stratify
            else None
        )
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.X,
                dataset.y,
                test_size=request.test_size,
                random_state=request.random_state,
                stratify=stratify_target,
            )
        except ValueError as error:
            raise ValueError(
                f"Unable to split dataset '{request.dataset_id}' for training: {error}"
            ) from error

        started_at = datetime.now(timezone.utc)
        started = perf_counter()
        pipeline.fit(X_train, y_train)
        training_seconds = perf_counter() - started

        predictions = self._call_capability(
            pipeline,
            specification.capabilities,
            ModelCapability.PREDICT,
            "predict",
            X_test,
        )
        probabilities = self._call_capability(
            pipeline,
            specification.capabilities,
            ModelCapability.PREDICT_PROBA,
            "predict_proba",
            X_test,
        )
        scores = self._call_capability(
            pipeline,
            specification.capabilities,
            ModelCapability.DECISION_FUNCTION,
            "decision_function",
            X_test,
        )
        final_estimator = pipeline.named_steps[ESTIMATOR_STEP]
        feature_importances = self._read_attribute(
            final_estimator,
            specification.capabilities,
            ModelCapability.FEATURE_IMPORTANCES,
            "feature_importances_",
        )
        coefficients = self._read_attribute(
            final_estimator,
            specification.capabilities,
            ModelCapability.COEFFICIENTS,
            "coef_",
        )

        configuration = TrainingConfiguration(
            dataset_id=request.dataset_id,
            model_id=request.model_id,
            dataset_parameters=dict(dataset.parameters),
            model_parameters=resolved_parameters,
            test_size=request.test_size,
            split_random_state=request.random_state,
            estimator_random_state=estimator_random_state,
            stratified=stratify_target is not None,
        )
        metadata: dict[str, object] = {
            "dataset_name": dataset.dataset_name,
            "dataset_metadata": dict(dataset.metadata),
            "model_name": specification.display_name,
            "n_observations": dataset.n_observations,
            "n_features": dataset.n_features,
            "train_observations": len(X_train),
            "test_observations": len(X_test),
            "started_at": started_at.isoformat(),
        }
        return TrainingOutput(
            trained_model=pipeline,
            predictions=predictions,
            probabilities=probabilities,
            scores=scores,
            training_seconds=training_seconds,
            metadata=metadata,
            configuration=configuration,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_importances=feature_importances,
            coefficients=coefficients,
        )

    @staticmethod
    def _call_capability(
        pipeline: Pipeline,
        capabilities: frozenset[ModelCapability],
        capability: ModelCapability,
        method_name: str,
        X: pd.DataFrame,
    ) -> np.ndarray | None:
        if capability not in capabilities:
            return None
        method = getattr(pipeline, method_name, None)
        if not callable(method):
            raise ValueError(
                f"Model declares capability '{capability.value}' but its pipeline does not "
                f"provide '{method_name}'."
            )
        try:
            return np.asarray(method(X))
        except AttributeError as error:
            raise ValueError(
                f"Model declares capability '{capability.value}' but its estimator does not "
                f"implement '{method_name}'."
            ) from error

    @staticmethod
    def _read_attribute(
        estimator: object,
        capabilities: frozenset[ModelCapability],
        capability: ModelCapability,
        attribute_name: str,
    ) -> object | None:
        if capability not in capabilities:
            return None
        try:
            return getattr(estimator, attribute_name)
        except AttributeError as error:
            raise ValueError(
                f"Model declares capability '{capability.value}' but the fitted estimator does "
                f"not expose '{attribute_name}'."
            ) from error


__all__ = [
    "GenericTrainingRunner",
    "TrainingConfiguration",
    "TrainingOutput",
    "TrainingRequest",
    "TrainingRunner",
]
