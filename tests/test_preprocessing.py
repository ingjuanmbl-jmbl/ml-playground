"""Behavioral tests for model-aware sklearn preprocessing pipelines."""

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_playground.models.specifications import (
    HyperparameterSpec,
    ModelCapability,
    ModelSpecification,
    ProblemType,
)
from ml_playground.preprocessing.pipelines import (
    ESTIMATOR_STEP,
    PREPROCESSING_STEP,
    build_pipeline,
    build_preprocessor,
    get_pipeline_stages,
)


class SeededChoiceEstimator(ClassifierMixin, BaseEstimator):
    """A small deterministic estimator used only to exercise the pipeline contract."""

    def __init__(self, random_state: int = 0) -> None:
        self.random_state = random_state

    def fit(self, X, y):
        values = np.asarray(X)
        self.classes_ = np.unique(y)
        self.chosen_class_ = np.random.default_rng(self.random_state).choice(self.classes_)
        self.fit_mean_ = values.mean(axis=0)
        return self

    def predict(self, X):
        return np.full(len(X), self.chosen_class_)


def make_model_specification(*, requires_scaling: bool) -> ModelSpecification:
    return ModelSpecification(
        id="test-seeded-estimator",
        display_name="Test Seeded Estimator",
        problem_type=ProblemType.CLASSIFICATION,
        estimator_factory=SeededChoiceEstimator,
        hyperparameters=(HyperparameterSpec("random_state", int, default=7, minimum=0),),
        capabilities=frozenset({ModelCapability.PREDICT}),
        requires_scaling=requires_scaling,
    )


@pytest.fixture()
def train_data():
    X = pd.DataFrame({"feature_a": [-2.0, 0.0, 2.0], "feature_b": [10.0, 12.0, 14.0]})
    y = pd.Series([0, 1, 0])
    return X, y


def test_pipeline_without_preprocessing_uses_passthrough_and_trains(train_data):
    specification = make_model_specification(requires_scaling=False)
    pipeline = build_pipeline(specification)
    X_train, y_train = train_data

    fitted = pipeline.fit(X_train, y_train)
    predictions = fitted.predict(X_train)

    assert isinstance(pipeline, Pipeline)
    assert pipeline.named_steps[PREPROCESSING_STEP] == "passthrough"
    assert isinstance(pipeline.named_steps[ESTIMATOR_STEP], SeededChoiceEstimator)
    assert predictions.shape == (len(X_train),)


def test_pipeline_with_scaler_trains_and_predicts(train_data):
    pipeline = build_pipeline(make_model_specification(requires_scaling=True))
    X_train, y_train = train_data

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_train)

    assert isinstance(pipeline.named_steps[PREPROCESSING_STEP], StandardScaler)
    assert pipeline.named_steps[ESTIMATOR_STEP].fit_mean_ == pytest.approx([0.0, 0.0])
    assert predictions.shape == (len(X_train),)


def test_pipeline_stage_order_is_preprocessing_then_estimator():
    pipeline = build_pipeline(make_model_specification(requires_scaling=True))

    assert [name for name, _ in get_pipeline_stages(pipeline)] == [
        PREPROCESSING_STEP,
        ESTIMATOR_STEP,
    ]
    assert get_pipeline_stages(pipeline) == tuple(pipeline.steps)


def test_build_preprocessor_obeys_model_specification():
    assert isinstance(
        build_preprocessor(make_model_specification(requires_scaling=True)), StandardScaler
    )
    assert build_preprocessor(make_model_specification(requires_scaling=False)) == "passthrough"


def test_pipeline_accepts_a_fictitious_estimator_from_model_specification(train_data):
    specification = make_model_specification(requires_scaling=False)
    pipeline = build_pipeline(specification, {"random_state": 11})
    X_train, y_train = train_data

    pipeline.fit(X_train, y_train)

    assert isinstance(pipeline.named_steps[ESTIMATOR_STEP], SeededChoiceEstimator)
    assert pipeline.named_steps[ESTIMATOR_STEP].random_state == 11


def test_same_estimator_random_state_produces_reproducible_predictions(train_data):
    specification = make_model_specification(requires_scaling=True)
    X_train, y_train = train_data
    first = build_pipeline(specification, {"random_state": 29}).fit(X_train, y_train)
    second = build_pipeline(specification, {"random_state": 29}).fit(X_train, y_train)

    np.testing.assert_array_equal(first.predict(X_train), second.predict(X_train))


def test_invalid_estimator_configuration_is_rejected():
    with pytest.raises(ValueError, match="Unknown parameter"):
        build_pipeline(make_model_specification(requires_scaling=False), {"unknown": 1})


def test_scaler_is_fitted_only_on_training_data():
    specification = make_model_specification(requires_scaling=True)
    pipeline = build_pipeline(specification)
    X_train = pd.DataFrame({"feature": [-1.0, 1.0]})
    y_train = pd.Series([0, 1])
    X_test = pd.DataFrame({"feature": [100.0, 102.0]})
    combined_mean = pd.concat([X_train, X_test]).mean().to_numpy()

    pipeline.fit(X_train, y_train)
    scaler = pipeline.named_steps[PREPROCESSING_STEP]
    test_transformed = pipeline.named_steps[PREPROCESSING_STEP].transform(X_test)
    test_predictions = pipeline.predict(X_test)

    np.testing.assert_allclose(scaler.mean_, X_train.mean().to_numpy())
    assert not np.allclose(scaler.mean_, combined_mean)
    assert np.all(np.abs(test_transformed) > 50)
    assert test_predictions.shape == (len(X_test),)


def test_model_specification_scaling_flag_controls_pipeline_without_model_name_checks():
    scale_sensitive = make_model_specification(requires_scaling=True)
    scale_insensitive = ModelSpecification(
        id="different-name",
        display_name="Different Name",
        problem_type=ProblemType.CLASSIFICATION,
        estimator_factory=SeededChoiceEstimator,
        requires_scaling=False,
    )

    assert isinstance(build_pipeline(scale_sensitive).named_steps[PREPROCESSING_STEP], StandardScaler)
    assert build_pipeline(scale_insensitive).named_steps[PREPROCESSING_STEP] == "passthrough"
