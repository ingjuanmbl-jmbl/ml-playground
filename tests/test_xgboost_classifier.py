"""XGBoost registration, generic training, evaluation, and UI integration."""

import ast
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.evaluation.results import ClassificationResult
from ml_playground.models.classification import (
    xgboost_classifier_specification,
)
from ml_playground.models.catalog import DEFAULT_MODEL_REGISTRY
from ml_playground.models.specifications import ModelCapability, ProblemType
from ml_playground.preprocessing.pipelines import ESTIMATOR_STEP, PREPROCESSING_STEP, build_pipeline
from ml_playground.training.contracts import TrainingRequest
from ml_playground.training.runner import GenericTrainingRunner
from ml_playground.visualization.classification import (
    decision_boundary_figure,
    feature_importance_figure,
)


def _run(dataset_id="iris", **parameters):
    return GenericTrainingRunner(DEFAULT_MODEL_REGISTRY).run(
        TrainingRequest(
            dataset_id=dataset_id,
            model_id="xgboost_classifier",
            dataset_parameters=parameters.pop("dataset_parameters", {}),
            model_parameters=parameters,
            random_state=17,
        )
    )


def test_xgboost_is_registered_with_expected_classification_contract():
    spec = DEFAULT_MODEL_REGISTRY.get("xgboost_classifier")
    assert spec.display_name == "XGBoost"
    assert spec.problem_type is ProblemType.CLASSIFICATION
    assert not spec.requires_scaling
    assert spec.capabilities == frozenset({
        ModelCapability.PREDICT,
        ModelCapability.PREDICT_PROBA,
        ModelCapability.FEATURE_IMPORTANCES,
    })
    assert isinstance(spec.build_estimator(), XGBClassifier)
    assert {item.name for item in spec.hyperparameters} == {
        "n_estimators", "max_depth", "learning_rate", "min_child_weight", "subsample",
        "colsample_bytree", "gamma", "reg_alpha", "reg_lambda", "random_state",
    }


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"n_estimators": 0}, ValueError, "at least 1"),
        ({"max_depth": 0}, ValueError, "at least 1"),
        ({"learning_rate": 0.0}, ValueError, "at least 0.001"),
        ({"learning_rate": "fast"}, TypeError, "must be of type float"),
        ({"subsample": 1.1}, ValueError, "at most 1.0"),
        ({"reg_alpha": -1.0}, ValueError, "at least 0.0"),
        ({"typo": 5}, ValueError, "Unknown parameter"),
    ],
)
def test_xgboost_rejects_invalid_hyperparameters(parameters, error, message):
    with pytest.raises(error, match=message):
        xgboost_classifier_specification().build_estimator(parameters)


def test_xgboost_pipeline_uses_passthrough_and_no_scaler():
    pipeline = build_pipeline(xgboost_classifier_specification(), {"n_estimators": 2})
    assert isinstance(pipeline, Pipeline)
    assert list(pipeline.named_steps) == [PREPROCESSING_STEP, ESTIMATOR_STEP]
    assert pipeline.named_steps[PREPROCESSING_STEP] == "passthrough"
    assert isinstance(pipeline.named_steps[ESTIMATOR_STEP], XGBClassifier)


@pytest.mark.parametrize(
    "parameters",
    [
        {"n_estimators": 10, "max_depth": 2, "learning_rate": 0.30},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.05},
    ],
)
def test_requested_xgboost_configurations_train_using_generic_runner(parameters):
    output = _run(**parameters)
    estimator = output.trained_model.named_steps[ESTIMATOR_STEP]
    assert isinstance(estimator, XGBClassifier)
    assert all(output.configuration.model_parameters[k] == v for k, v in parameters.items())
    assert output.predictions is not None
    assert output.probabilities.shape == (len(output.y_test), 3)
    assert output.feature_importances.shape == (output.X_test.shape[1],)
    assert output.training_seconds >= 0


def test_xgboost_supports_binary_prediction_probabilities_and_evaluation():
    output = _run("make_moons", dataset_parameters={"n_samples": 120, "noise": 0.2})
    assert set(np.unique(output.predictions)) <= {0, 1}
    assert output.probabilities.shape == (len(output.y_test), 2)
    result = evaluate_classification(output)
    assert isinstance(result, ClassificationResult)
    assert result.model_id == "xgboost_classifier"
    assert all(result.metrics[key] is not None for key in ("accuracy", "precision", "recall", "f1", "roc_auc"))
    assert result.confusion_matrix.shape == (2, 2)


def test_xgboost_multiclass_feature_importance_and_visualization():
    output = _run("iris", n_estimators=10, max_depth=2)
    result = evaluate_classification(output)
    assert result.confusion_matrix.shape == (3, 3)
    assert output.probabilities.shape == (len(output.y_test), 3)
    assert len(output.feature_importances) == len(output.X_test.columns)
    figure = feature_importance_figure(output.feature_importances, output.X_test.columns)
    assert set(figure.data[0].x) == set(output.X_test.columns)
    assert figure.layout.yaxis.title.text == "Importancia de variables"


def test_xgboost_pipeline_supports_existing_two_feature_boundary_figure():
    output = _run("make_moons", dataset_parameters={"n_samples": 100, "noise": 0.2}, n_estimators=5)
    result = evaluate_classification(output)
    figure = decision_boundary_figure(
        output.trained_model, output.X_test, output.y_test, result.class_labels
    )
    assert len(figure.data) == 3  # decision surface + both observed classes


def test_xgboost_rejects_regression_dataset():
    with pytest.raises(ValueError, match="requires classification"):
        _run("make_regression")


def test_xgboost_is_reproducible_for_same_random_state():
    options = {"n_estimators": 10, "max_depth": 2}
    first, second = _run(**options), _run(**options)
    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_allclose(first.probabilities, second.probabilities)


def test_xgboost_default_objective_handles_multiclass_without_forcing_binary_objective():
    output = _run("iris", n_estimators=2)
    estimator = output.trained_model.named_steps[ESTIMATOR_STEP]
    assert estimator.get_xgb_params()["objective"] == "multi:softprob"


def test_training_runner_remains_free_of_xgboost_specific_logic():
    path = Path(__file__).parents[1] / "src" / "ml_playground" / "training" / "runner.py"
    source = path.read_text(encoding="utf-8").lower()
    syntax = ast.parse(source)
    imports = [
        (node.module or "") + " " + " ".join(alias.name for alias in node.names)
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom)
    ]
    assert "xgboost" not in source
    assert "xgbclassifier" not in source
    assert all("xgboost" not in name.lower() for name in imports)
    assert "xgboost_classifier" not in source


def test_streamlit_apptest_runs_iris_with_xgboost():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py", default_timeout=60).run()
    next(item for item in app.selectbox if item.label == "Modelo").select("xgboost_classifier").run()
    for label, value in (("n_estimators", 10), ("max_depth", 2), ("learning_rate", 0.3)):
        widget = next(item for item in app.number_input if item.label == label)
        widget.set_value(value)
    next(item for item in app.button if item.label == "🚀 Entrenar modelo").click().run()
    assert not app.exception, app.exception
    parameters = [
        json.loads(item.value).get("parámetros_del_modelo", {}) for item in app.json
    ]
    assert any(
        values.get("n_estimators") == 10
        and values.get("max_depth") == 2
        and values.get("learning_rate") == 0.3
        for values in parameters
    )
    assert "📊 Métricas" in [item.value for item in app.subheader]
    assert "Importancia de variables" in [item.value for item in app.subheader]
