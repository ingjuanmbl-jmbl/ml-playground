"""Integration tests for generic, specification-driven supervised training."""

import ast
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_playground.data.contracts import Dataset
from ml_playground.data.registry import DatasetRegistry
from ml_playground.data.specifications import DatasetSpecification
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.clustering import kmeans_specification
from ml_playground.models.specifications import (
    HyperparameterSpec,
    ModelCapability,
    ModelSpecification,
    ProblemType,
)
from ml_playground.preprocessing.pipelines import ESTIMATOR_STEP, PREPROCESSING_STEP
from ml_playground.training.contracts import TrainingOutput, TrainingRequest
from ml_playground.training.runner import GenericTrainingRunner


class FakeClassifier(ClassifierMixin, BaseEstimator):
    """Minimal test estimator recording exactly the observations passed to fit."""

    def __init__(self, random_state: int = 0, offset: int = 0) -> None:
        self.random_state = random_state
        self.offset = offset

    def fit(self, X, y):
        self.fit_values_ = np.asarray(X).copy()
        self.fit_targets_ = np.asarray(y).copy()
        self.classes_ = np.unique(y)
        generator = np.random.default_rng(self.random_state)
        self.predicted_class_ = generator.choice(self.classes_)
        self.feature_importances_ = np.ones(self.fit_values_.shape[1])
        self.coef_ = np.zeros((1, self.fit_values_.shape[1]))
        return self

    def predict(self, X):
        return np.full(len(X), self.predicted_class_)

    def predict_proba(self, X):
        return np.tile([0.25, 0.75], (len(X), 1))

    def decision_function(self, X):
        return np.asarray(X)[:, 0]


def make_classification_dataset(
    n_samples: int = 60, random_state: int = 19
) -> Dataset:
    generator = np.random.default_rng(random_state)
    X = pd.DataFrame(
        generator.normal(size=(n_samples, 2)), columns=("feature_a", "feature_b")
    )
    y = pd.Series(np.arange(n_samples) % 2, name="target")
    return Dataset(
        X=X,
        y=y,
        feature_names=("feature_a", "feature_b"),
        target_names=("class_0", "class_1"),
        dataset_name="Tiny test data",
        problem_type=ProblemType.CLASSIFICATION,
        metadata={"source": "test"},
        parameters={"n_samples": n_samples, "random_state": random_state},
    )


def make_regression_dataset(n_samples: int = 60, random_state: int = 19) -> Dataset:
    dataset = make_classification_dataset(n_samples=n_samples, random_state=random_state)
    return Dataset(
        X=dataset.X,
        y=dataset.y.astype(float),
        feature_names=dataset.feature_names,
        target_names=None,
        dataset_name="Tiny test regression data",
        problem_type=ProblemType.REGRESSION,
        metadata=dataset.metadata,
        parameters=dataset.parameters,
    )


def make_clustering_dataset() -> Dataset:
    dataset = make_classification_dataset()
    return Dataset(
        X=dataset.X,
        y=dataset.y,
        feature_names=dataset.feature_names,
        target_names=dataset.target_names,
        dataset_name="Tiny test clustering data",
        problem_type=ProblemType.CLUSTERING,
    )


def make_separated_dataset(n_samples: int = 60, random_state: int = 19) -> Dataset:
    """Place observations selected into train and test at very different locations."""
    y = pd.Series(np.arange(n_samples) % 2, name="target")
    train_indices, test_indices = train_test_split(
        np.arange(n_samples), test_size=0.25, random_state=42, stratify=y
    )
    values = np.zeros(n_samples)
    values[test_indices] = 100.0
    X = pd.DataFrame({"feature": values})
    return Dataset(
        X=X,
        y=y,
        feature_names=("feature",),
        target_names=("class_0", "class_1"),
        dataset_name="Train-test separated data",
        problem_type=ProblemType.CLASSIFICATION,
    )


def make_dataset_registry(*, include_regression: bool = True, include_separated: bool = False):
    specifications = [
        DatasetSpecification(
            id="tiny",
            display_name="Tiny test data",
            problem_type=ProblemType.CLASSIFICATION,
            factory=make_classification_dataset,
            parameters=(
                HyperparameterSpec("n_samples", int, default=60, minimum=10, maximum=1000),
                HyperparameterSpec("random_state", int, default=19, minimum=0),
            ),
        )
    ]
    if include_regression:
        specifications.append(
            DatasetSpecification(
                id="regression",
                display_name="Tiny test regression data",
                problem_type=ProblemType.REGRESSION,
                factory=make_regression_dataset,
                parameters=(
                    HyperparameterSpec("n_samples", int, default=60, minimum=10, maximum=1000),
                    HyperparameterSpec("random_state", int, default=19, minimum=0),
                ),
            )
        )
    if include_separated:
        specifications.append(
            DatasetSpecification(
                id="separated",
                display_name="Train-test separated data",
                problem_type=ProblemType.CLASSIFICATION,
                factory=make_separated_dataset,
            )
        )
    return DatasetRegistry(specifications)


def make_model_specification(
    *,
    problem_type: ProblemType = ProblemType.CLASSIFICATION,
    capabilities: frozenset[ModelCapability] | None = None,
    requires_scaling: bool = True,
    with_random_state: bool = True,
) -> ModelSpecification:
    parameters = (
        (
            HyperparameterSpec("random_state", int, default=5, minimum=0),
            HyperparameterSpec("offset", int, default=0),
        )
        if with_random_state
        else (HyperparameterSpec("offset", int, default=0),)
    )
    return ModelSpecification(
        id="fake-classifier",
        display_name="Fake Classifier",
        problem_type=problem_type,
        estimator_factory=FakeClassifier,
        hyperparameters=parameters,
        capabilities=capabilities
        if capabilities is not None
        else frozenset(
            {
                ModelCapability.PREDICT,
                ModelCapability.PREDICT_PROBA,
                ModelCapability.DECISION_FUNCTION,
                ModelCapability.FEATURE_IMPORTANCES,
                ModelCapability.COEFFICIENTS,
            }
        ),
        requires_scaling=requires_scaling,
    )


def make_request(**overrides) -> TrainingRequest:
    values = {
        "dataset_id": "tiny",
        "model_id": "fake-classifier",
        "dataset_parameters": {"n_samples": 60},
        "model_parameters": {},
        "test_size": 0.25,
        "random_state": 42,
        "stratify": True,
    }
    values.update(overrides)
    return TrainingRequest(**values)


@pytest.fixture()
def runner():
    return GenericTrainingRunner(
        model_registry=ModelRegistry([make_model_specification()]),
        dataset_registry=make_dataset_registry(),
    )


def test_training_request_can_be_created_with_split_configuration():
    request = make_request()

    assert request.dataset_id == "tiny"
    assert request.model_id == "fake-classifier"
    assert request.test_size == 0.25
    assert request.random_state == 42
    assert request.stratify is True


@pytest.mark.parametrize("test_size", [0, 1, -0.1, 1.1, float("nan")])
def test_training_request_rejects_invalid_test_size(test_size):
    with pytest.raises(ValueError, match="test_size"):
        make_request(test_size=test_size)


def test_training_request_rejects_wrong_execution_option_types():
    with pytest.raises(TypeError, match="random_state must be an integer"):
        make_request(random_state=True)
    with pytest.raises(TypeError, match="stratify must be a bool"):
        make_request(stratify=1)


def test_training_request_rejects_random_state_outside_sklearn_seed_range():
    with pytest.raises(ValueError, match=r"0 and 2\*\*32 - 1"):
        make_request(random_state=2**32)


def test_runner_performs_stratified_train_test_split_and_returns_both_sides(runner):
    output = runner.run(make_request())

    assert len(output.X_train) == 45
    assert len(output.X_test) == 15
    assert len(output.y_train) == 45
    assert len(output.y_test) == 15
    assert sorted(output.y_train.value_counts().tolist()) == [22, 23]
    assert sorted(output.y_test.value_counts().tolist()) == [7, 8]
    assert output.configuration.stratified is True


def test_runner_is_reproducible_for_same_dataset_and_execution_seed(runner):
    first = runner.run(make_request(random_state=31))
    second = runner.run(make_request(random_state=31))

    assert first.X_train.index.tolist() == second.X_train.index.tolist()
    assert first.X_test.index.tolist() == second.X_test.index.tolist()
    np.testing.assert_array_equal(first.predictions, second.predictions)


def test_runner_builds_and_fits_pipeline_and_returns_predictions(runner):
    output = runner.run(make_request())

    assert isinstance(output, TrainingOutput)
    assert isinstance(output.trained_model.named_steps[PREPROCESSING_STEP], StandardScaler)
    assert isinstance(output.trained_model.named_steps[ESTIMATOR_STEP], FakeClassifier)
    assert output.predictions.shape == (len(output.X_test),)
    assert output.y_test is not None


def test_runner_captures_fit_time_metadata_and_resolved_configuration(runner):
    output = runner.run(make_request())

    assert output.training_seconds >= 0
    assert output.metadata["dataset_name"] == "Tiny test data"
    assert output.metadata["train_observations"] == 45
    assert output.metadata["test_observations"] == 15
    assert output.configuration.dataset_parameters["random_state"] == 19
    assert output.configuration.model_parameters["random_state"] == 42
    assert output.configuration.split_random_state == 42
    assert output.configuration.estimator_random_state == 42


def test_runner_propagates_model_parameters_and_preserves_explicit_random_state(runner):
    output = runner.run(make_request(model_parameters={"random_state": 77, "offset": 9}))
    estimator = output.trained_model.named_steps[ESTIMATOR_STEP]

    assert estimator.random_state == 77
    assert estimator.offset == 9
    assert output.configuration.model_parameters["offset"] == 9
    assert output.configuration.estimator_random_state == 77


def test_runner_does_not_add_random_state_when_model_spec_does_not_declare_it():
    registry = ModelRegistry(
        [make_model_specification(with_random_state=False, requires_scaling=False)]
    )
    runner = GenericTrainingRunner(registry, make_dataset_registry())

    output = runner.run(make_request())

    assert output.trained_model.named_steps[ESTIMATOR_STEP].random_state == 0
    assert output.configuration.estimator_random_state is None


def test_invalid_model_hyperparameters_fail_before_fitting(runner):
    with pytest.raises(ValueError, match="Unknown parameter"):
        runner.run(make_request(model_parameters={"not_a_parameter": 1}))


def test_invalid_dataset_parameters_fail_through_dataset_specification(runner):
    with pytest.raises(TypeError, match="must be of type int"):
        runner.run(make_request(dataset_parameters={"n_samples": "sixty"}))


def test_unknown_model_id_has_clear_error(runner):
    with pytest.raises(KeyError, match="Unknown model id"):
        runner.run(make_request(model_id="missing"))


def test_unknown_dataset_id_has_clear_error(runner):
    with pytest.raises(KeyError, match="Unknown dataset id"):
        runner.run(make_request(dataset_id="missing"))


def test_incompatible_dataset_and_model_problem_types_are_rejected():
    runner = GenericTrainingRunner(
        ModelRegistry([make_model_specification()]), make_dataset_registry()
    )

    with pytest.raises(ValueError, match="requires classification"):
        runner.run(make_request(dataset_id="regression"))


def test_absent_capabilities_leave_corresponding_outputs_unavailable():
    model = make_model_specification(capabilities=frozenset({ModelCapability.PREDICT}))
    runner = GenericTrainingRunner(ModelRegistry([model]), make_dataset_registry())

    output = runner.run(make_request())

    assert output.predictions is not None
    assert output.probabilities is None
    assert output.scores is None
    assert output.feature_importances is None
    assert output.coefficients is None


def test_predict_is_not_attempted_when_capability_is_not_declared():
    model = make_model_specification(capabilities=frozenset())
    runner = GenericTrainingRunner(ModelRegistry([model]), make_dataset_registry())

    output = runner.run(make_request())

    assert output.predictions is None
    assert output.probabilities is None
    assert output.scores is None


def test_declared_predict_probability_and_decision_capabilities_are_used(runner):
    output = runner.run(make_request())

    assert output.probabilities.shape == (15, 2)
    assert output.scores.shape == (15,)
    assert output.feature_importances.shape == (2,)
    assert output.coefficients.shape == (1, 2)


def test_test_observations_do_not_reach_estimator_fit(runner):
    output = runner.run(make_request())
    estimator = output.trained_model.named_steps[ESTIMATOR_STEP]
    scaler = output.trained_model.named_steps[PREPROCESSING_STEP]

    assert len(estimator.fit_values_) == len(output.X_train)
    assert len(estimator.fit_values_) != len(output.X_train) + len(output.X_test)
    np.testing.assert_allclose(estimator.fit_values_, scaler.transform(output.X_train))


def test_integration_leakage_check_fits_scaler_on_split_train_only():
    model = make_model_specification()
    runner = GenericTrainingRunner(
        ModelRegistry([model]), make_dataset_registry(include_separated=True)
    )

    output = runner.run(make_request(dataset_id="separated", dataset_parameters={}))
    scaler = output.trained_model.named_steps[PREPROCESSING_STEP]
    all_data_mean = pd.concat([output.X_train, output.X_test]).mean().to_numpy()
    test_transformed = scaler.transform(output.X_test)

    np.testing.assert_allclose(scaler.mean_, output.X_train.mean().to_numpy())
    assert not np.allclose(scaler.mean_, all_data_mean)
    assert np.all(np.abs(test_transformed) > 50)
    assert output.predictions.shape == (len(output.X_test),)


def test_clustering_runner_uses_declared_labels_without_assuming_supervised_predict():
    clustering_dataset = DatasetSpecification(
        id="tiny-clustering",
        display_name="Tiny clustering data",
        problem_type=ProblemType.CLUSTERING,
        factory=make_clustering_dataset,
    )
    runner = GenericTrainingRunner(
        ModelRegistry([kmeans_specification()]), DatasetRegistry([clustering_dataset])
    )
    output = runner.run(
        TrainingRequest(
            dataset_id="tiny-clustering", model_id="kmeans", model_parameters={"n_clusters": 2}
        )
    )
    assert output.cluster_labels is not None
    assert output.X_used is not None
    assert output.y_train is None and output.y_test is None


def test_training_package_has_no_streamlit_imports():
    training_dir = Path(__file__).parents[1] / "src" / "ml_playground" / "training"
    imported_modules: list[str] = []
    for path in training_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module is not None:
                imported_modules.append(node.module)

    assert all(module != "streamlit" and not module.startswith("streamlit.") for module in imported_modules)
