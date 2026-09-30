"""Core sensitivity-explorer tests independent from Streamlit."""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.data.contracts import Dataset
from ml_playground.data.registry import DatasetRegistry
from ml_playground.data.specifications import DatasetSpecification
from ml_playground.evaluation.results import MetricAvailability
from ml_playground.experiments.hyperparameter_explorer import (
    HyperparameterExploration,
    HyperparameterExplorer,
    is_explorable,
)
from ml_playground.models.catalog import DEFAULT_MODEL_REGISTRY
from ml_playground.models.classification import decision_tree_specification, mlp_classifier_specification
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import ProblemType
from ml_playground.training.runner import GenericTrainingRunner
from ml_playground.visualization.hyperparameter_explorer import hyperparameter_metric_figure


class RecordingRunner:
    def __init__(self, model_registry, dataset_registry):
        self.delegate = GenericTrainingRunner(model_registry, dataset_registry)
        self.requests = []
        self.outputs = []

    def run(self, request):
        self.requests.append(request)
        output = self.delegate.run(request)
        self.outputs.append(output)
        return output


def _explorer(*, dataset_registry=DEFAULT_DATASET_REGISTRY):
    runner = RecordingRunner(DEFAULT_MODEL_REGISTRY, dataset_registry)
    return HyperparameterExplorer(DEFAULT_MODEL_REGISTRY, dataset_registry, runner), runner


def test_exploration_result_is_structured_and_decision_tree_depth_is_comparable():
    explorer, runner = _explorer()
    result = explorer.explore(
        dataset_id="iris",
        model_id="decision_tree",
        hyperparameter="max_depth",
        values=[10, 1],
        metric="accuracy",
        test_size=0.25,
        random_state=27,
    )
    assert isinstance(result, HyperparameterExploration)
    assert (result.model_id, result.dataset_id, result.hyperparameter) == (
        "decision_tree", "iris", "max_depth"
    )
    assert result.values == (1, 10)
    assert result.base_parameters["max_depth"] is None
    assert len(result.results) == 2
    assert all(run.metrics["accuracy"] is not None for run in result.results)
    assert all(run.training_seconds >= 0 for run in result.results)
    assert all(request.random_state == 27 and request.test_size == 0.25 for request in runner.requests)
    assert all(request.stratify for request in runner.requests)
    first, second = runner.requests
    changed = {
        name
        for name in first.model_parameters
        if first.model_parameters[name] != second.model_parameters[name]
    }
    assert changed == {"max_depth"}
    np.testing.assert_array_equal(runner.outputs[0].y_test, runner.outputs[1].y_test)
    np.testing.assert_array_equal(runner.outputs[0].X_test, runner.outputs[1].X_test)
    assert all(request.dataset_parameters == second.dataset_parameters for request in runner.requests)


def test_random_forest_n_estimators_exploration_runs_multiple_configurations():
    explorer, _ = _explorer()
    result = explorer.explore(
        dataset_id="iris",
        model_id="random_forest",
        hyperparameter="n_estimators",
        values=[4, 2],
        metric="f1",
        random_state=13,
    )
    assert result.values == (2, 4)
    assert [item.metrics["f1"] for item in result.results]
    assert all(isinstance(item.metrics["f1"], float) for item in result.results)


def test_kmeans_n_clusters_exploration_uses_clustering_evaluation():
    explorer, runner = _explorer()
    result = explorer.explore(
        dataset_id="make_blobs",
        dataset_parameters={"n_samples": 60, "centers": 3},
        model_id="kmeans",
        hyperparameter="n_clusters",
        values=[3, 2],
        metric="silhouette",
        random_state=8,
    )
    assert result.problem_type is ProblemType.CLUSTERING
    assert result.values == (2, 3)
    assert all(run.metrics["silhouette"] is not None for run in result.results)
    assert all(run.metric_availability["silhouette"].available for run in result.results)
    assert all(output.configuration.split_performed is False for output in runner.outputs)
    assert all(output.metadata["target_used"] is False for output in runner.outputs)


def test_categorical_values_follow_hyperparameter_choice_order():
    explorer, _ = _explorer()
    result = explorer.explore(
        dataset_id="iris",
        model_id="decision_tree",
        hyperparameter="criterion",
        values=["log_loss", "gini"],
        metric="accuracy",
    )
    assert result.values == ("gini", "log_loss")


def test_exploration_metrics_are_reproducible_with_same_seed():
    explorer = HyperparameterExplorer(DEFAULT_MODEL_REGISTRY, DEFAULT_DATASET_REGISTRY)
    options = {
        "dataset_id": "iris",
        "model_id": "decision_tree",
        "hyperparameter": "max_depth",
        "values": [1, 5, 10],
        "metric": "f1",
        "random_state": 61,
    }
    first = explorer.explore(**options)
    second = explorer.explore(**options)
    assert [item.metrics for item in first.results] == [item.metrics for item in second.results]


def test_metric_unavailability_is_preserved_with_reason():
    dataset = Dataset(
        X=pd.DataFrame({"a": np.arange(10), "b": np.arange(10) ** 2}),
        y=pd.Series([0] * 10, name="target"),
        feature_names=("a", "b"),
        target_names=("only",),
        dataset_name="Single class test data",
        problem_type=ProblemType.CLASSIFICATION,
    )
    registry = DatasetRegistry(
        [DatasetSpecification("single", "Single", ProblemType.CLASSIFICATION, lambda: dataset)]
    )
    explorer, _ = _explorer(dataset_registry=registry)
    result = explorer.explore(
        dataset_id="single",
        model_id="decision_tree",
        hyperparameter="max_depth",
        values=[1, 2],
        metric="roc_auc",
        stratify=False,
    )
    assert all(run.metrics["roc_auc"] is None for run in result.results)
    assert all(
        isinstance(run.metric_availability["roc_auc"], MetricAvailability)
        and not run.metric_availability["roc_auc"].available
        and run.metric_availability["roc_auc"].reason
        for run in result.results
    )


@pytest.mark.parametrize(
    ("kwargs", "error", "message"),
    [
        ({"model_id": "missing", "hyperparameter": "max_depth"}, KeyError, "missing"),
        ({"model_id": "decision_tree", "hyperparameter": "unknown"}, ValueError, "not explorable"),
        ({"model_id": "decision_tree", "hyperparameter": "max_depth", "values": [0, 2]}, ValueError, "at least 1"),
        ({"model_id": "decision_tree", "hyperparameter": "random_state"}, ValueError, "not explorable"),
        ({"model_id": "mlp_classifier", "hyperparameter": "hidden_layer_sizes"}, ValueError, "not explorable"),
    ],
)
def test_invalid_model_or_exploration_parameter_is_rejected(kwargs, error, message):
    explorer = HyperparameterExplorer(DEFAULT_MODEL_REGISTRY, DEFAULT_DATASET_REGISTRY)
    values = kwargs.pop("values", [1, 2])
    with pytest.raises(error, match=message):
        explorer.explore(
            dataset_id="iris", values=values, metric="accuracy", **kwargs
        )


def test_exploration_rejects_metric_for_another_problem_type():
    explorer = HyperparameterExplorer(DEFAULT_MODEL_REGISTRY, DEFAULT_DATASET_REGISTRY)
    with pytest.raises(ValueError, match="not available for clustering"):
        explorer.explore(
            dataset_id="make_blobs",
            model_id="kmeans",
            hyperparameter="n_clusters",
            values=[2, 3],
            metric="accuracy",
        )


def test_explorer_service_and_chart_preserve_value_order_and_have_no_streamlit_dependency():
    module_path = Path(__file__).parents[1] / "src" / "ml_playground" / "experiments" / "hyperparameter_explorer.py"
    source = module_path.read_text(encoding="utf-8").lower()
    tree = ast.parse(source)
    modules = [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ]
    modules.extend(
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert all(module != "streamlit" and not module.startswith("streamlit.") for module in modules)
    figure = hyperparameter_metric_figure(
        model_name="Decision Tree",
        hyperparameter="max_depth",
        metric="accuracy",
        values=(1, 10),
        scores=(0.8, 0.9),
    )
    assert list(figure.data[0].x) == [1, 10]
    assert list(figure.data[0].y) == [0.8, 0.9]
    assert figure.layout.title.text == "Decision Tree: accuracy vs max_depth"


def test_structural_parameters_are_not_explorable():
    parameter = next(
        item for item in mlp_classifier_specification().hyperparameters
        if item.name == "hidden_layer_sizes"
    )
    assert not is_explorable(parameter)


@pytest.mark.parametrize(
    ("model_id", "parameter_name", "values", "metric"),
    [
        ("decision_tree", "max_depth", "1,10", "accuracy"),
        ("kmeans", "n_clusters", "2,3", "silhouette"),
    ],
)
def test_streamlit_apptest_runs_hyperparameter_explorations(
    model_id, parameter_name, values, metric
):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py", default_timeout=90).run()
    next(item for item in app.selectbox if item.label == "Explorer model").select(model_id).run()
    next(item for item in app.selectbox if item.label == "Hyperparameter").select(parameter_name).run()
    next(item for item in app.text_input if item.label == "Values (comma separated)").set_value(values)
    next(item for item in app.selectbox if item.label == "Metric").select(metric)
    next(item for item in app.button if item.label == "Run exploration").click().run()
    assert not app.exception, app.exception
    assert "Exploration results" in [item.value for item in app.subheader]
    assert len(app.get("plotly_chart")) >= 2
    result = app.session_state["hyperparameter_explorer_result"]
    assert result.values == tuple(int(value) for value in values.split(","))
