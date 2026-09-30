"""K-Means contracts, generic execution, evaluation, and Streamlit integration."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.evaluation.clustering import evaluate_clustering
from ml_playground.models.catalog import DEFAULT_MODEL_REGISTRY
from ml_playground.models.clustering import kmeans_specification
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import ModelCapability, ProblemType
from ml_playground.preprocessing.pipelines import ESTIMATOR_STEP, PREPROCESSING_STEP, build_pipeline
from ml_playground.training.contracts import TrainingRequest
from ml_playground.training.runner import GenericTrainingRunner
from ml_playground.visualization.clustering import clustering_scatter_figure


def _run(dataset_id: str = "make_blobs", *, dataset_parameters=None, **parameters):
    return GenericTrainingRunner(DEFAULT_MODEL_REGISTRY).run(
        TrainingRequest(
            dataset_id=dataset_id,
            model_id="kmeans",
            dataset_parameters=dataset_parameters or {},
            model_parameters=parameters,
            random_state=31,
        )
    )


def test_kmeans_is_registered_with_clustering_contract():
    spec = DEFAULT_MODEL_REGISTRY.get("kmeans")
    assert spec.problem_type is ProblemType.CLUSTERING
    assert spec.requires_scaling is True
    assert spec.capabilities == frozenset(
        {ModelCapability.CLUSTER_LABELS, ModelCapability.CENTROIDS}
    )
    assert isinstance(spec.build_estimator(), KMeans)
    assert {item.id for item in DEFAULT_MODEL_REGISTRY.list()} == {
        "logistic_regression", "decision_tree", "random_forest", "mlp_classifier",
        "xgboost_classifier", "kmeans",
    }


def test_kmeans_hyperparameters_have_declared_defaults_and_accept_valid_values():
    spec = kmeans_specification()
    assert {parameter.name for parameter in spec.hyperparameters} == {
        "n_clusters", "init", "n_init", "max_iter", "random_state"
    }
    estimator = spec.build_estimator(
        {"n_clusters": 4, "init": "random", "n_init": 5, "max_iter": 120, "random_state": 7}
    )
    assert estimator.get_params()["n_clusters"] == 4
    assert estimator.get_params()["random_state"] == 7


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"n_clusters": 1}, ValueError, "at least 2"),
        ({"n_clusters": "3"}, TypeError, "must be of type int"),
        ({"init": "forgy"}, ValueError, "must be one of"),
        ({"n_init": 0}, ValueError, "at least 1"),
        ({"max_iter": -1}, ValueError, "at least 1"),
        ({"random_state": -1}, ValueError, "at least 0"),
        ({"unknown": 1}, ValueError, "Unknown parameter"),
    ],
)
def test_kmeans_rejects_invalid_parameters(parameters, error, message):
    with pytest.raises(error, match=message):
        kmeans_specification().build_estimator(parameters)


def test_kmeans_pipeline_scales_then_fits_estimator():
    pipeline = build_pipeline(kmeans_specification(), {"n_clusters": 3})
    assert list(pipeline.named_steps) == [PREPROCESSING_STEP, ESTIMATOR_STEP]
    assert isinstance(pipeline.named_steps[PREPROCESSING_STEP], StandardScaler)
    assert isinstance(pipeline.named_steps[ESTIMATOR_STEP], KMeans)


@pytest.mark.parametrize("dataset_id", ["make_blobs", "make_moons"])
def test_kmeans_trains_reproducibly_on_unsupervised_datasets(dataset_id):
    output = _run(dataset_id, n_clusters=2, n_init=5)
    assert output.cluster_labels.shape == (output.metadata["n_observations"],)
    assert output.X_used.shape[0] == output.cluster_labels.size
    assert output.y_train is None and output.y_test is None
    assert output.predictions is None and output.probabilities is None
    assert output.centroids.shape == (2, output.X_used.shape[1])
    assert output.metadata["target_used"] is False
    assert output.configuration.split_performed is False
    assert output.configuration.model_parameters["n_clusters"] == 2

    repeated = _run(dataset_id, n_clusters=2, n_init=5)
    np.testing.assert_array_equal(output.cluster_labels, repeated.cluster_labels)
    np.testing.assert_allclose(output.centroids, repeated.centroids)


def test_kmeans_fits_all_X_and_does_not_use_original_target(monkeypatch):
    original = DEFAULT_DATASET_REGISTRY.build("iris")
    flipped = replace(original, y=original.y.iloc[::-1].reset_index(drop=True))

    class StaticDatasetRegistry:
        def __init__(self, dataset):
            self.dataset = dataset

        def build(self, dataset_id, parameters=None):
            return self.dataset

    runner_original = GenericTrainingRunner(
        ModelRegistry([kmeans_specification()]), StaticDatasetRegistry(original)
    )
    runner_flipped = GenericTrainingRunner(
        ModelRegistry([kmeans_specification()]), StaticDatasetRegistry(flipped)
    )
    import ml_playground.training.runner as runner_module

    monkeypatch.setattr(
        runner_module,
        "create_training_split",
        lambda *args, **kwargs: pytest.fail("clustering must not split or pass the target"),
    )
    request = TrainingRequest(
        dataset_id="iris", model_id="kmeans", model_parameters={"n_clusters": 3}, random_state=23
    )
    first, second = runner_original.run(request), runner_flipped.run(request)
    assert first.y_train is None and first.y_test is None
    np.testing.assert_array_equal(first.cluster_labels, second.cluster_labels)
    np.testing.assert_allclose(first.centroids, second.centroids)


@pytest.mark.parametrize("dataset_id", ["iris", "wine", "make_moons", "make_circles"])
def test_kmeans_accepts_classification_datasets_as_unlabeled_feature_sources(dataset_id):
    output = _run(dataset_id, n_clusters=2, n_init=3)
    assert output.cluster_labels is not None
    assert output.metadata["target_used"] is False


def test_kmeans_rejects_more_clusters_than_observations():
    with pytest.raises(ValueError, match="cannot exceed the number of observations"):
        _run(
            "make_blobs",
            dataset_parameters={"n_samples": 8, "n_features": 2, "centers": 2},
            n_clusters=9,
        )


def test_kmeans_rejects_regression_dataset():
    with pytest.raises(ValueError, match="requires clustering"):
        _run("make_regression")


def test_clustering_evaluation_returns_metrics_and_original_scale_centroids():
    output = _run("make_blobs", n_clusters=3, n_init=5)
    result = evaluate_clustering(output)
    assert result.problem_type is ProblemType.CLUSTERING
    assert result.model_id == "kmeans"
    assert result.n_clusters == 3
    assert result.cluster_labels.shape == (len(output.X_used),)
    assert result.centroids.shape == (3, output.X_used.shape[1])
    assert result.metadata["centroids_space"] == "original feature scale"
    assert all(value is not None for value in result.metrics.values())
    assert all(item.available for item in result.metric_availability.values())


def test_clustering_evaluation_marks_single_cluster_metrics_unavailable():
    output = _run("make_blobs", n_clusters=2, n_init=3)
    output = replace(output, cluster_labels=np.zeros(len(output.X_used), dtype=int))
    result = evaluate_clustering(output)
    assert result.n_clusters == 1
    assert all(value is None for value in result.metrics.values())
    assert all(not item.available and item.reason for item in result.metric_availability.values())


def test_clustering_visualization_shows_observations_and_centroids_in_two_features():
    output = _run("make_blobs", n_clusters=3)
    result = evaluate_clustering(output)
    figure = clustering_scatter_figure(
        output.X_used, result.cluster_labels, "feature_0", "feature_1", centroids=result.centroids
    )
    assert len(figure.data) == 4  # three clusters and one centroid trace
    assert figure.data[-1].name == "Centroids"
    assert list(figure.data[-1].x) == pytest.approx(list(result.centroids[:, 0]))


def test_clustering_visualization_supports_selected_features_in_higher_dimensions():
    output = _run(
        "make_blobs",
        dataset_parameters={"n_samples": 90, "n_features": 4, "centers": 3},
        n_clusters=3,
    )
    result = evaluate_clustering(output)
    figure = clustering_scatter_figure(
        output.X_used, result.cluster_labels, "feature_1", "feature_3", centroids=result.centroids
    )
    assert figure.layout.xaxis.title.text == "feature_1"
    assert figure.layout.yaxis.title.text == "feature_3"
    assert len(figure.data) == result.n_clusters + 1


def test_streamlit_apptest_runs_kmeans_on_blobs_and_displays_centroids():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app.py", default_timeout=60).run()
    next(item for item in app.selectbox if item.label == "Dataset").select("make_blobs").run()
    next(item for item in app.selectbox if item.label == "Model").select("kmeans").run()
    assert not any(item.label == "Test set size" for item in app.slider)
    next(item for item in app.button if item.label == "Train and evaluate").click().run()
    assert not app.exception, app.exception
    assert "Clustering evaluation" in [item.value for item in app.subheader]
    assert "Cluster centroids" in [item.value for item in app.subheader]
    assert len(app.get("plotly_chart")) >= 2  # dataset preview and the cluster/centroid chart


def test_runner_supports_clustering_generically_without_kmeans_specific_branch():
    path = Path(__file__).parents[1] / "src" / "ml_playground" / "training" / "runner.py"
    source = path.read_text(encoding="utf-8").lower()
    assert "kmeans" not in source
    assert "model_id == \"kmeans\"" not in source
    assert "from sklearn.cluster import" not in source
