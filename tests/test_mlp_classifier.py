"""Registration, training, and Streamlit integration for MLPClassifier."""

import ast
import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.evaluation.results import ClassificationResult
from ml_playground.models.classification import (
    mlp_classifier_specification,
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
from ml_playground.visualization.classification import loss_curve_figure


def _fit(architecture: tuple[int, ...] = (10,), *, seed: int = 42, max_iter: int = 600):
    request = TrainingRequest(
        dataset_id="iris",
        model_id="mlp_classifier",
        model_parameters={
            "hidden_layer_sizes": architecture,
            "max_iter": max_iter,
            "early_stopping": False,
        },
        test_size=0.25,
        random_state=seed,
    )
    return GenericTrainingRunner(DEFAULT_MODEL_REGISTRY).run(request)


def test_mlp_classifier_is_registered_with_expected_contract():
    specification = DEFAULT_MODEL_REGISTRY.get("mlp_classifier")
    assert specification.display_name == "MLPClassifier"
    assert specification.problem_type is ProblemType.CLASSIFICATION
    assert specification.requires_scaling is True
    assert specification.capabilities == frozenset(
        {ModelCapability.PREDICT, ModelCapability.PREDICT_PROBA}
    )
    assert {spec.id for spec in DEFAULT_MODEL_REGISTRY.list()} == {
        "logistic_regression", "decision_tree", "random_forest", "mlp_classifier", "xgboost_classifier", "kmeans"
    }


def test_mlp_specification_accepts_requested_architecture_presets():
    spec = mlp_classifier_specification()
    parameter = next(item for item in spec.hyperparameters if item.name == "hidden_layer_sizes")
    assert parameter.choices == ((10,), (20,), (20, 10), (50, 25, 10))
    for architecture in parameter.choices:
        estimator = spec.build_estimator({"hidden_layer_sizes": architecture})
        assert estimator.hidden_layer_sizes == architecture


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"hidden_layer_sizes": (0,)}, ValueError, "must be one of"),
        ({"hidden_layer_sizes": ()}, ValueError, "must be one of"),
        ({"hidden_layer_sizes": "20,10"}, TypeError, "must be of type tuple"),
        ({"activation": "swish"}, ValueError, "must be one of"),
        ({"solver": "rmsprop"}, ValueError, "must be one of"),
        ({"alpha": -0.1}, ValueError, "at least 0.0"),
        ({"learning_rate_init": 0.0}, ValueError, "at least 1e-05"),
        ({"max_iter": 0}, ValueError, "at least 50"),
        ({"batch_size": 12}, ValueError, "must be one of"),
        ({"early_stopping": "yes"}, TypeError, "must be of type bool"),
    ],
)
def test_mlp_invalid_hyperparameters_are_rejected(parameters, error, message):
    with pytest.raises(error, match=message):
        mlp_classifier_specification().validate_parameters(parameters)


def test_mlp_pipeline_scales_then_fits_the_estimator():
    pipeline = build_pipeline(
        mlp_classifier_specification(),
        {"hidden_layer_sizes": (10,), "max_iter": 100},
    )
    assert list(pipeline.named_steps) == [PREPROCESSING_STEP, ESTIMATOR_STEP]
    assert isinstance(pipeline.named_steps[PREPROCESSING_STEP], StandardScaler)
    assert isinstance(pipeline.named_steps[ESTIMATOR_STEP], MLPClassifier)


@pytest.mark.parametrize("architecture", [(10,), (20, 10)])
def test_generic_runner_trains_mlp_with_outputs_and_scaling(architecture):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        output = _fit(architecture)
    for warning in caught:
        warnings.warn(warning.message, warning.category)
    estimator = output.trained_model.named_steps[ESTIMATOR_STEP]
    assert isinstance(estimator, MLPClassifier)
    assert estimator.hidden_layer_sizes == architecture
    assert output.predictions is not None
    assert output.probabilities is not None
    assert output.scores is None
    assert output.feature_importances is None
    assert output.coefficients is None
    assert output.configuration.model_parameters["hidden_layer_sizes"] == architecture
    assert output.trained_model.named_steps[PREPROCESSING_STEP].mean_.shape[0] == 4
    assert output.training_seconds >= 0
    # Re-emit any captured warning so pytest and CI still report it.


def test_mlp_predictions_are_reproducible_with_same_random_state():
    first = _fit((10,), seed=93)
    second = _fit((10,), seed=93)
    np.testing.assert_array_equal(first.y_test, second.y_test)
    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_allclose(first.probabilities, second.probabilities)


def test_mlp_training_output_uses_existing_evaluation():
    result = evaluate_classification(_fit((10,)))
    assert isinstance(result, ClassificationResult)
    assert result.model_id == "mlp_classifier"
    assert all(result.metrics[name] is not None for name in ("accuracy", "precision", "recall", "f1"))
    assert result.confusion_matrix.shape == (3, 3)


def test_mlp_loss_curve_uses_values_recorded_by_fitted_estimator():
    estimator = _fit((10,)).trained_model.named_steps[ESTIMATOR_STEP]
    assert hasattr(estimator, "loss_curve_") and len(estimator.loss_curve_) > 0
    figure = loss_curve_figure(estimator.loss_curve_)
    assert figure.data[0].type == "scatter"
    assert list(figure.data[0].x) == list(range(1, len(estimator.loss_curve_) + 1))
    np.testing.assert_allclose(figure.data[0].y, estimator.loss_curve_)


def test_mlp_exposes_network_summary_without_invented_feature_importance():
    estimator = _fit((20, 10)).trained_model.named_steps[ESTIMATOR_STEP]
    parameter_count = sum(
        np.asarray(values).size for values in (*estimator.coefs_, *estimator.intercepts_)
    )
    assert estimator.n_layers_ == 4  # input + two hidden layers + output
    assert estimator.n_iter_ > 0
    assert parameter_count > 0


def test_mlp_rejects_regression_dataset():
    request = TrainingRequest(dataset_id="make_regression", model_id="mlp_classifier")
    with pytest.raises(ValueError, match="requires classification"):
        GenericTrainingRunner(DEFAULT_MODEL_REGISTRY).run(request)


def test_runner_has_no_mlp_import_or_model_specific_branch():
    path = Path(__file__).parents[1] / "src" / "ml_playground" / "training" / "runner.py"
    source = path.read_text(encoding="utf-8").lower()
    syntax = ast.parse(source)
    imports = [
        name.lower()
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom)
        for name in [node.module or "", *(alias.name for alias in node.names)]
    ]
    assert "mlp_classifier" not in source and "mlpclassifier" not in source
    assert all("sklearn.neural_network" not in name for name in imports)


@pytest.mark.parametrize("architecture", [(10,), (20, 10)])
def test_streamlit_apptest_runs_iris_with_selected_mlp_architecture(architecture):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py", default_timeout=30).run()
    next(item for item in app.selectbox if item.label == "Model").select("mlp_classifier").run()
    next(item for item in app.selectbox if item.label == "Hidden layer sizes").select(architecture)
    next(item for item in app.button if item.label == "Train and evaluate").click().run()
    assert not app.exception, app.exception
    parameter_objects = [json.loads(item.value) for item in app.json]
    assert any(
        values.get("hidden_layer_sizes") == list(architecture)
        for values in parameter_objects
    )
    assert "Neural network details" in [item.value for item in app.subheader]
    assert len(app.get("plotly_chart")) >= 2  # confusion matrix and recorded loss curve
