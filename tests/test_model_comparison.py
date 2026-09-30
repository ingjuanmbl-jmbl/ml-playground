from __future__ import annotations

import importlib
import inspect

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.evaluation.results import MetricAvailability
from ml_playground.experiments.model_comparison import ModelComparisonService
from ml_playground.models.catalog import DEFAULT_MODEL_REGISTRY
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import ModelCapability, ModelSpecification, ProblemType
from ml_playground.training.contracts import TrainingRequest, TrainingRunner, TrainingSplit
from ml_playground.training.runner import GenericTrainingRunner


class RecordingRunner:
    def __init__(self, delegate: TrainingRunner) -> None:
        self.delegate = delegate
        self.requests: list[TrainingRequest] = []
        self.outputs = []

    def run(self, request: TrainingRequest):
        self.requests.append(request)
        output = self.delegate.run(request)
        self.outputs.append(output)
        return output


def _service(runner: RecordingRunner | None = None) -> ModelComparisonService:
    delegate = runner or RecordingRunner(
        GenericTrainingRunner(DEFAULT_MODEL_REGISTRY, DEFAULT_DATASET_REGISTRY)
    )
    return ModelComparisonService(training_runner=delegate)


def test_comparison_requires_two_distinct_registered_models() -> None:
    service = _service()
    with pytest.raises(ValueError, match="at least two"):
        service.compare(dataset_id="iris", model_ids=["logistic_regression"])
    with pytest.raises(ValueError, match="once"):
        service.compare(dataset_id="iris", model_ids=["decision_tree", "decision_tree"])
    with pytest.raises(KeyError, match="Unknown model"):
        service.compare(dataset_id="iris", model_ids=["unknown", "decision_tree"])


def test_five_registered_classifiers_share_exact_same_split_and_random_state() -> None:
    recorder = RecordingRunner(GenericTrainingRunner(DEFAULT_MODEL_REGISTRY, DEFAULT_DATASET_REGISTRY))
    selected = [
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "xgboost_classifier",
        "mlp_classifier",
    ]
    result = ModelComparisonService(training_runner=recorder).compare(
        dataset_id="iris", model_ids=selected, test_size=0.25, random_state=13
    )
    assert tuple(item.model_id for item in result.models) == tuple(selected)
    assert len(recorder.requests) == 5
    assert all(request.split is result.split for request in recorder.requests)
    assert {request.random_state for request in recorder.requests} == {13}
    assert all(output.configuration.split == result.split for output in recorder.outputs)
    first = recorder.outputs[0]
    for output in recorder.outputs[1:]:
        np.testing.assert_array_equal(output.X_train.to_numpy(), first.X_train.to_numpy())
        np.testing.assert_array_equal(output.X_test.to_numpy(), first.X_test.to_numpy())
        np.testing.assert_array_equal(output.y_train.to_numpy(), first.y_train.to_numpy())
        np.testing.assert_array_equal(output.y_test.to_numpy(), first.y_test.to_numpy())


def test_result_has_metrics_availability_training_time_and_shared_configuration() -> None:
    comparison = _service().compare(
        dataset_id="iris", model_ids=["logistic_regression", "decision_tree"]
    )
    assert isinstance(comparison.split, TrainingSplit)
    assert comparison.timestamp.tzinfo is not None
    for model in comparison.models:
        assert model.training_seconds >= 0
        assert set(model.metrics) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert model.metrics["accuracy"] is not None
        assert "roc_auc" in model.metric_availability
        assert isinstance(model.metric_availability["roc_auc"], MetricAvailability)
        assert model.metric_availability["roc_auc"].available is True
        assert model.configuration["split_random_state"] == comparison.random_state


def test_comparison_reproducible_for_fixed_configuration() -> None:
    args = dict(dataset_id="make_moons", model_ids=["decision_tree", "random_forest"],
                dataset_parameters={"n_samples": 120, "random_state": 7}, random_state=19)
    first = _service().compare(**args)
    second = _service().compare(**args)
    assert first.split == second.split
    assert [item.metrics for item in first.models] == [item.metrics for item in second.models]


def test_regression_dataset_and_clustering_model_are_rejected() -> None:
    service = _service()
    with pytest.raises(ValueError, match="classification dataset"):
        service.compare(dataset_id="make_regression", model_ids=["decision_tree", "random_forest"])
    with pytest.raises(ValueError, match="classification models"):
        service.compare(dataset_id="iris", model_ids=["decision_tree", "kmeans"])


class _NoScoresClassifier(ClassifierMixin, BaseEstimator):
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.majority_ = self.classes_[0]
        return self

    def predict(self, X):
        return np.full(len(X), self.majority_)


def test_roc_auc_unavailable_is_none_with_reason() -> None:
    registry = ModelRegistry(
        [
            *DEFAULT_MODEL_REGISTRY.list(),
            ModelSpecification(
            id="no_scores_test_classifier",
            display_name="No scores test model",
            problem_type=ProblemType.CLASSIFICATION,
            estimator_factory=_NoScoresClassifier,
            capabilities=frozenset({ModelCapability.PREDICT}),
            ),
        ]
    )
    service = ModelComparisonService(model_registry=registry)
    result = service.compare(
        dataset_id="iris", model_ids=["no_scores_test_classifier", "decision_tree"]
    )
    unavailable = result.models[0]
    assert unavailable.metrics["roc_auc"] is None
    assert unavailable.metric_availability["roc_auc"].available is False
    assert "No probabilities" in unavailable.metric_availability["roc_auc"].reason


def test_service_uses_generic_runner_and_evaluator_without_streamlit_or_manual_fit() -> None:
    module = importlib.import_module("ml_playground.experiments.model_comparison")
    source = inspect.getsource(module)
    assert "GenericTrainingRunner" in source
    assert "evaluate_classification" in source
    assert ".fit(" not in source
    assert "streamlit" not in source.lower()
    assert "LogisticRegression" not in source
    assert "DecisionTreeClassifier" not in source
    assert "RandomForestClassifier" not in source
    assert "XGBClassifier" not in source
    assert "MLPClassifier" not in source


def test_app_test_runs_model_comparison() -> None:
    from pathlib import Path
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(Path("app.py").resolve()), default_timeout=60).run()
    assert not app.exception
    comparison_button = next(button for button in app.button if button.label == "Run comparison")
    comparison_button.click().run()
    assert not app.exception
    assert any("Model Comparison" in header.value for header in app.header)
    assert app.session_state["model_comparison_result"] is not None
