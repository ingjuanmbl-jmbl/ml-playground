"""Independent tests for held-out classification evaluation."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.evaluation.results import ClassificationResult, MetricAvailability
from ml_playground.models.specifications import ProblemType
from ml_playground.training.contracts import TrainingConfiguration, TrainingOutput


class NoFitEstimator:
    def fit(self, *args, **kwargs):
        raise AssertionError("Evaluation must not train an estimator.")


def make_output(
    y_true=(0, 1, 0, 1),
    y_pred=(0, 1, 1, 1),
    *,
    probabilities=None,
    scores=None,
):
    return TrainingOutput(
        trained_model=NoFitEstimator(),
        predictions=None if y_pred is None else np.asarray(y_pred),
        probabilities=None if probabilities is None else np.asarray(probabilities),
        scores=None if scores is None else np.asarray(scores),
        training_seconds=0.125,
        metadata={"model_name": "Test model", "started_at": "2026-01-02T03:04:05+00:00"},
        configuration=TrainingConfiguration(
            dataset_id="test-data",
            model_id="test-model",
            dataset_parameters={},
            model_parameters={},
            test_size=0.25,
            split_random_state=4,
            estimator_random_state=None,
            stratified=True,
        ),
        y_test=None if y_true is None else pd.Series(y_true),
    )


def test_binary_metrics_and_confusion_matrix_match_sklearn():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    result = evaluate_classification(make_output(y_true, y_pred))

    assert result.metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert result.metrics["precision"] == precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    assert result.metrics["recall"] == recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    assert result.metrics["f1"] == f1_score(y_true, y_pred, average="macro", zero_division=0)
    np.testing.assert_array_equal(
        result.confusion_matrix,
        confusion_matrix(y_true, y_pred, labels=result.class_labels),
    )
    assert result.averaging_strategy == "macro"
    assert result.class_labels == (0, 1)


def test_multiclass_metrics_and_confusion_matrix_match_sklearn():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 2, 2, 0, 1, 1])
    result = evaluate_classification(make_output(y_true, y_pred))

    assert result.metrics["accuracy"] == accuracy_score(y_true, y_pred)
    assert result.metrics["precision"] == precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    assert result.metrics["recall"] == recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    assert result.metrics["f1"] == f1_score(y_true, y_pred, average="macro", zero_division=0)
    np.testing.assert_array_equal(
        result.confusion_matrix,
        confusion_matrix(y_true, y_pred, labels=(0, 1, 2)),
    )


def test_binary_roc_auc_uses_predict_proba_and_matches_sklearn():
    y_true = np.array([0, 1, 0, 1])
    probabilities = np.array([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.1, 0.9]])
    result = evaluate_classification(
        make_output(y_true, (0, 1, 1, 1), probabilities=probabilities)
    )

    assert result.metrics["roc_auc"] == roc_auc_score(y_true, probabilities[:, 1])
    assert result.metric_availability["roc_auc"].available


def test_binary_roc_auc_uses_decision_function_and_matches_sklearn():
    y_true = np.array([0, 1, 0, 1])
    scores = np.array([-2.0, 1.0, -0.5, 3.0])
    result = evaluate_classification(make_output(y_true, (0, 1, 1, 1), scores=scores))

    assert result.metrics["roc_auc"] == roc_auc_score(y_true, scores)


def test_multiclass_roc_auc_uses_ovr_macro_and_matches_sklearn():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    probabilities = np.array(
        [[.8, .1, .1], [.1, .8, .1], [.1, .2, .7], [.6, .2, .2], [.2, .7, .1], [.1, .1, .8]]
    )
    result = evaluate_classification(
        make_output(y_true, (0, 1, 2, 0, 1, 2), probabilities=probabilities)
    )

    assert result.metrics["roc_auc"] == roc_auc_score(
        y_true, probabilities, labels=(0, 1, 2), multi_class="ovr", average="macro"
    )


def test_roc_auc_unavailable_without_scores_has_explanation():
    result = evaluate_classification(make_output())

    assert result.metrics["roc_auc"] is None
    availability = result.metric_availability["roc_auc"]
    assert not availability.available
    assert "No probabilities" in availability.reason


@pytest.mark.parametrize(
    "probabilities",
    [np.ones((4, 3)), np.ones((3, 2)), np.array([[.1, .9], [np.nan, .2], [.4, .6], [.3, .7]])],
)
def test_incompatible_probability_scores_mark_auc_unavailable(probabilities):
    result = evaluate_classification(make_output(probabilities=probabilities))

    assert result.metrics["roc_auc"] is None
    assert not result.metric_availability["roc_auc"].available
    assert result.metric_availability["roc_auc"].reason


def test_roc_auc_unavailable_when_y_true_has_only_one_class():
    result = evaluate_classification(make_output((1, 1, 1), (1, 1, 1), scores=(.1, .2, .3)))

    assert result.metrics["roc_auc"] is None
    assert "at least two classes" in result.metric_availability["roc_auc"].reason


def test_missing_predicted_class_and_zero_division_are_handled_explicitly():
    result = evaluate_classification(make_output((0, 1, 0, 1), (0, 0, 0, 0)))

    assert result.metrics["precision"] == pytest.approx(0.25)
    assert result.metrics["recall"] == pytest.approx(0.5)
    assert result.metrics["f1"] == pytest.approx(1 / 3)
    assert result.class_labels == (0, 1)
    assert result.confusion_matrix.tolist() == [[2, 0], [2, 0]]


@pytest.mark.parametrize(
    ("output", "message"),
    [
        (make_output(y_true=(), y_pred=()), "cannot be empty"),
        (make_output(y_true=(0, 1), y_pred=(0,)), "same length"),
        (make_output(y_true=None), "requires y_test"),
        (make_output(y_pred=None), "requires predictions"),
        (make_output(y_pred=((0, 1), (1, 0))), "one-dimensional"),
        (make_output(y_true=(0, None), y_pred=(0, 1)), "valid, non-missing"),
    ],
)
def test_invalid_evaluation_inputs_raise_clear_errors(output, message):
    with pytest.raises(ValueError, match=message):
        evaluate_classification(output)


def test_result_contract_carries_metrics_classes_and_availability():
    result = evaluate_classification(make_output())

    assert isinstance(result, ClassificationResult)
    assert result.problem_type is ProblemType.CLASSIFICATION
    assert set(result.metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert isinstance(result.metric_availability["roc_auc"], MetricAvailability)
    assert result.metric_availability["accuracy"].available


def test_evaluation_uses_only_test_outputs_and_does_not_mutate_training_output():
    output = make_output(probabilities=((.8, .2), (.1, .9), (.6, .4), (.2, .8)))
    before_predictions = output.predictions.copy()
    before_targets = output.y_test.copy()
    result = evaluate_classification(output)

    np.testing.assert_array_equal(output.predictions, before_predictions)
    pd.testing.assert_series_equal(output.y_test, before_targets)
    assert output.X_train is None and output.X_test is None and output.y_train is None
    assert result is not output


def test_evaluation_package_has_no_streamlit_imports_or_training_calls():
    evaluation_dir = Path(__file__).parents[1] / "src" / "ml_playground" / "evaluation"
    for path in evaluation_dir.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = []
        called_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_modules.append(node.module)
            elif isinstance(node, ast.Call):
                called_names.append(
                    node.func.id if isinstance(node.func, ast.Name) else
                    node.func.attr if isinstance(node.func, ast.Attribute) else ""
                )
        assert all(name != "streamlit" and not name.startswith("streamlit.") for name in imported_modules)
        assert "fit" not in called_names and "fit_predict" not in called_names
