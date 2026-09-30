"""Registration-to-evaluation integration tests for Logistic Regression."""

import ast
import warnings
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.evaluation.results import ClassificationResult
from ml_playground.models.classification import (
    DEFAULT_MODEL_REGISTRY,
    logistic_regression_specification,
)
from ml_playground.models.specifications import ModelCapability, ProblemType
from ml_playground.preprocessing.pipelines import ESTIMATOR_STEP, PREPROCESSING_STEP, build_pipeline
from ml_playground.training.contracts import TrainingRequest
from ml_playground.training.runner import GenericTrainingRunner
from ml_playground.ui.training import build_training_request
from ml_playground.visualization.classification import (
    coefficient_figure,
    decision_boundary_figure,
)


def test_logistic_regression_is_registered_with_expected_contract():
    specification = DEFAULT_MODEL_REGISTRY.get("logistic_regression")

    assert specification.problem_type is ProblemType.CLASSIFICATION
    assert specification.requires_scaling is True
    assert callable(specification.estimator_factory)
    assert specification.capabilities == frozenset(
        {
            ModelCapability.PREDICT,
            ModelCapability.PREDICT_PROBA,
            ModelCapability.DECISION_FUNCTION,
            ModelCapability.COEFFICIENTS,
        }
    )
    assert {item.id for item in DEFAULT_MODEL_REGISTRY.list()} == {
        "logistic_regression", "decision_tree", "random_forest", "mlp_classifier"
    }


def test_logistic_regression_hyperparameters_have_valid_defaults_and_ranges():
    specification = logistic_regression_specification()

    assert {parameter.name for parameter in specification.hyperparameters} == {
        "C", "regularization", "solver", "max_iter", "random_state"
    }
    assert specification.validate_parameters() == {
        "C": 1.0,
        "regularization": "L2",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42,
    }
    estimator = specification.build_estimator(
        {"C": 2.0, "regularization": "L1", "solver": "saga", "max_iter": 500}
    )
    assert estimator.C == 2.0 and estimator.l1_ratio == 1.0 and estimator.solver == "saga"
    with pytest.raises(ValueError, match="at least 0.001"):
        specification.validate_parameters({"C": 0.0})
    with pytest.raises(ValueError, match="must be one of"):
        specification.validate_parameters({"solver": "invalid"})
    with pytest.raises(TypeError, match="must be of type float"):
        specification.validate_parameters({"C": "1.0"})


@pytest.mark.parametrize(
    ("regularization", "solver"),
    [("L1", "lbfgs"), ("L1", "newton-cg"), ("L1", "newton-cholesky"), ("L1", "sag")],
)
def test_incompatible_regularization_solver_pairs_are_rejected_by_spec(regularization, solver):
    with pytest.raises(ValueError, match="incompatible"):
        logistic_regression_specification().validate_parameters(
            {"regularization": regularization, "solver": solver}
        )


@pytest.mark.parametrize(
    ("regularization", "solver"),
    [("L1", "liblinear"), ("L1", "saga"), ("L2", "lbfgs"),
     ("L2", "liblinear"), ("L2", "newton-cg"), ("L2", "newton-cholesky"),
     ("L2", "sag"), ("L2", "saga")],
)
def test_compatible_regularization_solver_pairs_construct_estimators(regularization, solver):
    parameters = {"regularization": regularization, "solver": solver}
    assert logistic_regression_specification().validate_parameters(parameters)["solver"] == solver


@pytest.mark.parametrize(("regularization", "expected_ratio"), [("L2", 0.0), ("L1", 1.0)])
def test_l1_l2_map_to_modern_l1_ratio_without_penalty_future_warning(regularization, expected_ratio):
    specification = logistic_regression_specification()
    X, y = make_classification(
        n_samples=240, n_features=12, n_informative=3, n_redundant=0,
        n_repeated=0, n_classes=2, random_state=17,
    )
    estimator = specification.build_estimator(
        {"regularization": regularization, "solver": "liblinear", "C": 0.1, "max_iter": 2000}
    )
    assert estimator.l1_ratio == expected_ratio
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        estimator.fit(StandardScaler().fit_transform(X), y)

    penalty_warnings = [
        warning for warning in captured
        if issubclass(warning.category, FutureWarning) and "penalty" in str(warning.message)
    ]
    assert penalty_warnings == []
    if regularization == "L1":
        assert np.count_nonzero(estimator.coef_) < estimator.coef_.size
    else:
        assert np.count_nonzero(estimator.coef_) == estimator.coef_.size


def test_streamlit_form_values_build_training_request():
    request = build_training_request(
        dataset_id="iris",
        model_id="logistic_regression",
        dataset_parameters={},
        model_parameters={"C": 1.0, "regularization": "L1", "solver": "saga", "max_iter": 500},
        test_size=0.3,
        random_state=123,
    )

    assert isinstance(request, TrainingRequest)
    assert request.model_parameters["regularization"] == "L1"
    assert request.test_size == 0.3 and request.random_state == 123


def _fit(dataset_id="make_classification", dataset_parameters=None, *, seed=42):
    request = TrainingRequest(
        dataset_id=dataset_id,
        model_id="logistic_regression",
        dataset_parameters=dataset_parameters or {},
        model_parameters={"max_iter": 500},
        test_size=0.25,
        random_state=seed,
    )
    return GenericTrainingRunner(DEFAULT_MODEL_REGISTRY).run(request)


def test_generic_runner_passes_execution_seed_into_logistic_estimator():
    output = _fit("iris", seed=29)

    assert output.trained_model.named_steps[ESTIMATOR_STEP].random_state == 29
    assert output.configuration.model_parameters["random_state"] == 29


@pytest.mark.parametrize("regularization,solver", [("L1", "liblinear"), ("L2", "lbfgs")])
def test_runner_trains_with_l1_and_l2_without_logistic_warnings(regularization, solver):
    request = TrainingRequest(
        dataset_id="make_classification",
        model_id="logistic_regression",
        model_parameters={"regularization": regularization, "solver": solver, "max_iter": 1000},
        test_size=0.25,
        random_state=31,
    )
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        output = GenericTrainingRunner(DEFAULT_MODEL_REGISTRY).run(request)

    estimator = output.trained_model.named_steps[ESTIMATOR_STEP]
    assert estimator.l1_ratio == (1.0 if regularization == "L1" else 0.0)
    assert output.predictions is not None and output.probabilities is not None
    assert not any(
        issubclass(item.category, FutureWarning) and "penalty" in str(item.message)
        for item in captured
    )
    assert not any("converg" in str(item.message).lower() for item in captured)


def test_pipeline_is_scaler_then_logistic_regression():
    specification = logistic_regression_specification()
    pipeline = build_pipeline(specification, {"max_iter": 250})

    assert list(pipeline.named_steps) == [PREPROCESSING_STEP, ESTIMATOR_STEP]
    assert isinstance(pipeline.named_steps[PREPROCESSING_STEP], StandardScaler)
    assert isinstance(pipeline.named_steps[ESTIMATOR_STEP], LogisticRegression)


@pytest.mark.parametrize(
    ("dataset_id", "parameters"),
    [
        ("iris", {}),
        ("wine", {}),
        ("breast_cancer", {}),
        ("digits", {}),
        ("make_classification", {"n_samples": 100, "n_features": 2, "n_informative": 2,
                                 "n_redundant": 0, "n_repeated": 0, "n_classes": 2,
                                 "n_clusters_per_class": 1, "class_sep": 1.0, "flip_y": 0.01,
                                 "random_state": 9}),
        ("make_moons", {"n_samples": 100, "noise": 0.15, "random_state": 9}),
        ("make_circles", {"n_samples": 100, "noise": 0.05, "factor": 0.5, "random_state": 9}),
    ],
)
def test_registered_logistic_regression_trains_on_compatible_classification_datasets(
    dataset_id, parameters
):
    output = _fit(dataset_id, parameters)

    assert output.predictions is not None
    assert output.probabilities is not None
    assert output.scores is not None
    assert output.coefficients is not None
    assert output.trained_model.named_steps[ESTIMATOR_STEP].random_state == 42
    assert output.trained_model.named_steps[PREPROCESSING_STEP].mean_.shape[0] == output.X_train.shape[1]


def test_end_to_end_dataset_registry_runner_and_evaluation_contract():
    output = _fit(
        "make_classification",
        {"n_samples": 160, "n_features": 2, "n_informative": 2, "n_redundant": 0,
         "n_repeated": 0, "n_classes": 2, "n_clusters_per_class": 1, "class_sep": 1.0,
         "flip_y": 0.01, "random_state": 11},
    )
    result = evaluate_classification(output)

    assert isinstance(result, ClassificationResult)
    assert result.model_id == "logistic_regression"
    assert result.metrics["accuracy"] is not None
    assert result.metrics["precision"] is not None
    assert result.metrics["recall"] is not None
    assert result.metrics["f1"] is not None
    assert result.metrics["roc_auc"] is not None
    assert result.confusion_matrix.shape == (2, 2)


def test_training_predictions_probabilities_and_scores_are_reproducible():
    parameters = {"n_samples": 120, "n_features": 2, "n_informative": 2, "n_redundant": 0,
                  "n_repeated": 0, "n_classes": 2, "n_clusters_per_class": 1,
                  "class_sep": 1.0, "flip_y": 0.01, "random_state": 7}
    first = _fit("make_classification", parameters, seed=13)
    second = _fit("make_classification", parameters, seed=13)

    np.testing.assert_array_equal(first.y_test, second.y_test)
    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_allclose(first.probabilities, second.probabilities)
    np.testing.assert_allclose(first.scores, second.scores)


def test_regression_dataset_is_rejected_with_clear_problem_type_error():
    with pytest.raises(ValueError, match="requires classification"):
        _fit("make_regression")


def test_decision_boundary_uses_fitted_pipeline_for_two_feature_dataset():
    output = _fit("make_moons", {"n_samples": 100, "noise": 0.12, "random_state": 4})
    result = evaluate_classification(output)
    figure = decision_boundary_figure(
        output.trained_model, output.X_test, output.y_test, result.class_labels
    )

    assert figure.data[0].type == "contour"
    assert len(figure.data) == len(result.class_labels) + 1


def test_direct_decision_boundary_rejects_more_than_two_features():
    output = _fit("iris")
    result = evaluate_classification(output)

    with pytest.raises(ValueError, match="exactly two"):
        decision_boundary_figure(output.trained_model, output.X_test, output.y_test, result.class_labels)


def test_binary_coefficients_render_one_signed_value_per_feature():
    output = _fit("breast_cancer")
    result = evaluate_classification(output)
    figure = coefficient_figure(output.coefficients, output.X_train.columns, result.class_labels)

    assert figure.data[0].type == "bar"
    assert len(figure.data[0].y) == output.X_train.shape[1]


def test_multiclass_coefficients_render_class_by_feature_heatmap():
    output = _fit("iris")
    result = evaluate_classification(output)
    figure = coefficient_figure(output.coefficients, output.X_train.columns, result.class_labels)

    assert figure.data[0].type == "heatmap"
    assert np.asarray(figure.data[0].z).shape == (len(result.class_labels), output.X_train.shape[1])


def test_logistic_regression_addition_did_not_add_runner_algorithm_branches():
    runner_path = Path(__file__).parents[1] / "src" / "ml_playground" / "training" / "runner.py"
    source = runner_path.read_text(encoding="utf-8").lower()
    tree = ast.parse(source)
    imported = [
        alias.name.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    imported += [
        node.module.lower()
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    ]

    assert "logistic_regression" not in source
    assert all("linear_model" not in name for name in imported)
    assert _fit("iris").trained_model.named_steps[ESTIMATOR_STEP].__class__ is LogisticRegression


def test_supported_dataset_ids_are_present_in_dataset_registry():
    assert {"iris", "wine", "breast_cancer", "digits", "make_classification", "make_moons", "make_circles"} <= {
        specification.id for specification in DEFAULT_DATASET_REGISTRY.list()
    }
