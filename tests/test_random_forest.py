"""Registration, training, and output tests for the Random Forest classifier."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.evaluation.results import ClassificationResult
from ml_playground.models.classification import (
    random_forest_specification,
)
from ml_playground.models.catalog import DEFAULT_MODEL_REGISTRY
from ml_playground.models.specifications import ModelCapability, ProblemType
from ml_playground.preprocessing.pipelines import (
    ESTIMATOR_STEP,
    PREPROCESSING_STEP,
    build_pipeline,
)
from ml_playground.training.contracts import TrainingRequest
from ml_playground.training.runner import GenericTrainingRunner
from ml_playground.visualization.classification import (
    decision_boundary_figure,
    feature_importance_figure,
)


def _fit(
    dataset_id: str = "iris",
    dataset_parameters: dict[str, object] | None = None,
    *,
    seed: int = 42,
    model_parameters: dict[str, object] | None = None,
):
    request = TrainingRequest(
        dataset_id=dataset_id,
        model_id="random_forest",
        dataset_parameters=dataset_parameters or {},
        model_parameters={"n_estimators": 12, **(model_parameters or {})},
        test_size=0.25,
        random_state=seed,
    )
    return GenericTrainingRunner(DEFAULT_MODEL_REGISTRY).run(request)


def test_random_forest_is_registered_with_expected_contract():
    specification = DEFAULT_MODEL_REGISTRY.get("random_forest")
    assert specification.display_name == "Random Forest"
    assert specification.problem_type is ProblemType.CLASSIFICATION
    assert specification.requires_scaling is False
    assert specification.capabilities == frozenset(
        {
            ModelCapability.PREDICT,
            ModelCapability.PREDICT_PROBA,
            ModelCapability.FEATURE_IMPORTANCES,
        }
    )
    assert {spec.id for spec in DEFAULT_MODEL_REGISTRY.list()} == {
        "logistic_regression", "decision_tree", "random_forest", "mlp_classifier", "xgboost_classifier", "kmeans"
    }


def test_random_forest_specification_declares_requested_parameters_and_defaults():
    specification = random_forest_specification()
    assert {item.name for item in specification.hyperparameters} == {
        "n_estimators", "max_depth", "min_samples_split", "min_samples_leaf",
        "max_features", "random_state",
    }
    estimator = specification.build_estimator({"max_depth": None, "max_features": "log2"})
    assert isinstance(estimator, RandomForestClassifier)
    assert estimator.max_depth is None and estimator.max_features == "log2"
    assert estimator.n_estimators == 100


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"n_estimators": 0}, ValueError, "at least 1"),
        ({"n_estimators": "100"}, TypeError, "must be of type int"),
        ({"max_depth": 0}, ValueError, "at least 1"),
        ({"min_samples_split": 1}, ValueError, "at least 2"),
        ({"min_samples_leaf": 0}, ValueError, "at least 1"),
        ({"max_features": "invalid"}, ValueError, "must be one of"),
        ({"unknown": 3}, ValueError, "Unknown parameter"),
    ],
)
def test_invalid_random_forest_parameters_are_rejected(parameters, error, message):
    with pytest.raises(error, match=message):
        random_forest_specification().validate_parameters(parameters)


def test_random_forest_pipeline_uses_passthrough_then_estimator():
    pipeline = build_pipeline(random_forest_specification(), {"n_estimators": 5})
    assert list(pipeline.named_steps) == [PREPROCESSING_STEP, ESTIMATOR_STEP]
    assert pipeline.named_steps[PREPROCESSING_STEP] == "passthrough"
    assert isinstance(pipeline.named_steps[ESTIMATOR_STEP], RandomForestClassifier)


def test_generic_runner_trains_random_forest_and_propagates_configuration():
    output = _fit(
        seed=32,
        model_parameters={
            "n_estimators": 15,
            "max_depth": 4,
            "min_samples_split": 3,
            "min_samples_leaf": 2,
            "max_features": "log2",
        },
    )
    estimator = output.trained_model.named_steps[ESTIMATOR_STEP]
    assert estimator.n_estimators == 15
    assert estimator.max_depth == 4
    assert estimator.min_samples_split == 3
    assert estimator.min_samples_leaf == 2
    assert estimator.max_features == "log2"
    assert output.configuration.estimator_random_state == 32
    assert output.predictions is not None
    assert output.probabilities is not None
    assert output.scores is None
    assert output.feature_importances is not None
    assert output.training_seconds >= 0


def test_random_forest_predictions_are_reproducible_for_same_seed():
    first = _fit(seed=71)
    second = _fit(seed=71)
    np.testing.assert_array_equal(first.y_test, second.y_test)
    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_allclose(first.probabilities, second.probabilities)


def test_random_forest_output_is_supported_by_existing_evaluation():
    output = _fit()
    result = evaluate_classification(output)
    assert isinstance(result, ClassificationResult)
    assert result.model_id == "random_forest"
    assert all(result.metrics[name] is not None for name in ("accuracy", "precision", "recall", "f1"))
    assert result.confusion_matrix.shape == (3, 3)


def test_random_forest_feature_importances_map_to_features_and_visualize():
    output = _fit()
    values = np.asarray(output.feature_importances)
    names = list(output.X_train.columns)
    assert values.shape == (len(names),)
    figure = feature_importance_figure(values, names)
    plotted_values = np.asarray(figure.data[0].y)
    assert list(figure.data[0].x) == [names[index] for index in np.argsort(values)[::-1]]
    assert np.all(plotted_values[:-1] >= plotted_values[1:])
    assert figure.layout.title.text == "Estimator feature_importances_"


def test_random_forest_decision_boundary_uses_trained_pipeline_for_two_features():
    output = _fit("make_moons", {"n_samples": 100, "noise": 0.12, "random_state": 5})
    labels = evaluate_classification(output).class_labels
    figure = decision_boundary_figure(output.trained_model, output.X_test, output.y_test, labels)
    assert figure.data[0].type == "contour"


def test_random_forest_rejects_regression_dataset():
    with pytest.raises(ValueError, match="requires classification"):
        _fit("make_regression")


def test_random_forest_end_to_end_registry_runner_and_evaluation_flow():
    dataset = DEFAULT_DATASET_REGISTRY.build("iris")
    specification = DEFAULT_MODEL_REGISTRY.get("random_forest")
    request = TrainingRequest(
        dataset_id="iris",
        model_id=specification.id,
        model_parameters={"n_estimators": 10, "max_depth": None},
        test_size=0.2,
        random_state=13,
    )
    output = GenericTrainingRunner(DEFAULT_MODEL_REGISTRY).run(request)
    result = evaluate_classification(output)
    assert dataset.problem_type is ProblemType.CLASSIFICATION
    assert isinstance(result, ClassificationResult)
    assert result.model_id == "random_forest"
    assert output.metadata["dataset_name"] == "Iris"
    assert output.configuration.model_parameters["n_estimators"] == 10


def test_random_forest_addition_does_not_add_runner_specific_logic():
    path = Path(__file__).parents[1] / "src" / "ml_playground" / "training" / "runner.py"
    source = path.read_text(encoding="utf-8").lower()
    syntax = ast.parse(source)
    imported = [
        name.lower()
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom)
        for name in [node.module or "", *(alias.name for alias in node.names)]
    ]
    assert "random_forest" not in source
    assert "randomforestclassifier" not in source
    assert all("sklearn.ensemble" not in name for name in imported)


def test_streamlit_random_forest_form_runs_on_iris():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py", default_timeout=20).run()
    next(item for item in app.selectbox if item.label == "Model").select("random_forest").run()
    next(item for item in app.number_input if item.label == "N estimators").set_value(10)
    next(item for item in app.button if item.label == "Train and evaluate").click().run()
    assert not app.exception, app.exception
    rendered_parameters = [json.loads(item.value) for item in app.json]
    assert any(values.get("n_estimators") == 10 for values in rendered_parameters)
    assert "Feature importance" in [item.value for item in app.subheader]
