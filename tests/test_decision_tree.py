"""Contract and integration coverage for the registered Decision Tree classifier."""

import ast
from pathlib import Path

import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.evaluation.results import ClassificationResult
from ml_playground.models.classification import (
    decision_tree_specification,
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


def _fit(dataset_id="iris", dataset_parameters=None, *, seed=42, model_parameters=None):
    request = TrainingRequest(
        dataset_id=dataset_id,
        model_id="decision_tree",
        dataset_parameters=dataset_parameters or {},
        model_parameters=model_parameters or {},
        test_size=0.25,
        random_state=seed,
    )
    return GenericTrainingRunner(DEFAULT_MODEL_REGISTRY).run(request)


def test_decision_tree_registered_with_classification_capabilities():
    specification = DEFAULT_MODEL_REGISTRY.get("decision_tree")
    assert specification.display_name == "Decision Tree"
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


def test_decision_tree_hyperparameter_defaults_and_unlimited_depth():
    spec = decision_tree_specification()
    assert {item.name for item in spec.hyperparameters} == {
        "criterion", "splitter", "max_depth", "min_samples_split",
        "min_samples_leaf", "max_features", "random_state",
    }
    estimator = spec.build_estimator({"max_depth": None})
    assert isinstance(estimator, DecisionTreeClassifier)
    assert estimator.max_depth is None
    assert spec.validate_parameters({"criterion": "entropy", "splitter": "random"})[
        "criterion"
    ] == "entropy"


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"criterion": "invalid"}, ValueError, "must be one of"),
        ({"splitter": "invalid"}, ValueError, "must be one of"),
        ({"max_depth": 0}, ValueError, "at least 1"),
        ({"min_samples_split": 1}, ValueError, "at least 2"),
        ({"min_samples_leaf": 0}, ValueError, "at least 1"),
        ({"max_features": "unsupported"}, ValueError, "must be one of"),
        ({"max_depth": "deep"}, TypeError, "must be of type int or NoneType"),
        ({"min_samples_split": 2.5}, TypeError, "must be of type int"),
        ({"unknown": 3}, ValueError, "Unknown parameter"),
    ],
)
def test_invalid_decision_tree_parameters_are_rejected(parameters, error, message):
    with pytest.raises(error, match=message):
        decision_tree_specification().validate_parameters(parameters)


def test_decision_tree_pipeline_uses_passthrough_then_estimator():
    pipeline = build_pipeline(decision_tree_specification(), {"max_depth": 3})
    assert list(pipeline.named_steps) == [PREPROCESSING_STEP, ESTIMATOR_STEP]
    assert pipeline.named_steps[PREPROCESSING_STEP] == "passthrough"
    assert isinstance(pipeline.named_steps[ESTIMATOR_STEP], DecisionTreeClassifier)


@pytest.mark.parametrize("depth", [1, 10])
def test_runner_trains_decision_tree_with_parameters_and_output_capabilities(depth):
    output = _fit(model_parameters={"max_depth": depth, "criterion": "entropy"}, seed=19)
    estimator = output.trained_model.named_steps[ESTIMATOR_STEP]
    assert estimator.max_depth == depth
    assert estimator.criterion == "entropy"
    assert output.configuration.model_parameters["max_depth"] == depth
    assert output.configuration.estimator_random_state == 19
    assert output.predictions is not None
    assert output.probabilities is not None
    assert output.scores is None
    assert output.feature_importances is not None
    assert output.coefficients is None
    assert output.training_seconds >= 0


def test_decision_tree_runner_is_reproducible_with_same_seed():
    first = _fit(seed=27, model_parameters={"splitter": "random"})
    second = _fit(seed=27, model_parameters={"splitter": "random"})
    np.testing.assert_array_equal(first.y_test, second.y_test)
    np.testing.assert_array_equal(first.predictions, second.predictions)
    np.testing.assert_allclose(first.probabilities, second.probabilities)


def test_decision_tree_training_output_works_with_existing_evaluator():
    result = evaluate_classification(_fit())
    assert isinstance(result, ClassificationResult)
    assert result.model_id == "decision_tree"
    assert all(result.metrics[name] is not None for name in ("accuracy", "precision", "recall", "f1"))
    assert result.confusion_matrix.shape == (3, 3)


def test_tree_importances_match_features_and_plot_in_descending_order():
    output = _fit()
    importances = np.asarray(output.feature_importances)
    assert importances.shape == (len(output.X_train.columns),)
    figure = feature_importance_figure(importances, output.X_train.columns)
    plotted = np.asarray(figure.data[0].y)
    assert list(figure.data[0].x) == [output.X_train.columns[index] for index in np.argsort(importances)[::-1]]
    assert np.all(plotted[:-1] >= plotted[1:])
    assert figure.layout.yaxis.title.text == "feature_importances_"


def test_decision_boundary_is_available_for_two_features_via_pipeline():
    output = _fit("make_moons", {"n_samples": 100, "noise": 0.15, "random_state": 9})
    result = evaluate_classification(output)
    figure = decision_boundary_figure(
        output.trained_model, output.X_test, output.y_test, result.class_labels
    )
    assert figure.data[0].type == "contour"


def test_direct_boundary_rejects_dataset_with_more_than_two_features():
    output = _fit("iris")
    with pytest.raises(ValueError, match="exactly two"):
        decision_boundary_figure(
            output.trained_model,
            output.X_test,
            output.y_test,
            evaluate_classification(output).class_labels,
        )


def test_regression_dataset_is_rejected_by_classification_model():
    with pytest.raises(ValueError, match="requires classification"):
        _fit("make_regression")


def test_end_to_end_registry_runner_and_evaluation_flow():
    dataset = DEFAULT_DATASET_REGISTRY.build("iris")
    spec = DEFAULT_MODEL_REGISTRY.get("decision_tree")
    request = TrainingRequest(
        dataset_id=dataset.dataset_name.lower(),
        model_id=spec.id,
        model_parameters={"max_depth": 4},
        test_size=0.2,
        random_state=17,
    )
    output = GenericTrainingRunner(DEFAULT_MODEL_REGISTRY).run(request)
    result = evaluate_classification(output)
    assert isinstance(result, ClassificationResult)
    assert output.metadata["dataset_name"] == "Iris"
    assert output.configuration.model_id == "decision_tree"
    assert output.configuration.model_parameters["max_depth"] == 4


def test_training_runner_stays_free_of_decision_tree_specific_logic():
    path = Path(__file__).parents[1] / "src" / "ml_playground" / "training" / "runner.py"
    source = path.read_text(encoding="utf-8").lower()
    syntax = ast.parse(source)
    imported = [
        name.lower()
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom)
        for name in [node.module or "", *(alias.name for alias in node.names)]
    ]
    assert "decision_tree" not in source
    assert "decisiontreeclassifier" not in source
    assert all("sklearn.tree" not in name for name in imported)


def test_streamlit_builds_optional_depth_control_from_hyperparameter_spec():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py", default_timeout=20).run()
    next(item for item in app.selectbox if item.label == "Model").select("decision_tree").run()
    assert not app.exception
    depth_toggle = next(
        item for item in app.checkbox if item.label == "Set a maximum for max depth"
    )
    assert not depth_toggle.value
    depth_toggle.check().run()
    assert not app.exception
    assert any(item.label == "Max depth" for item in app.number_input)


def test_default_dataset_catalog_contains_supported_classification_datasets():
    assert {
        "iris", "wine", "breast_cancer", "digits", "make_classification",
        "make_moons", "make_circles",
    } <= {item.id for item in DEFAULT_DATASET_REGISTRY.list()}
