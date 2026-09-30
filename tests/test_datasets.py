"""Dataset catalog, validation, metadata, and reproducibility tests."""

import pandas as pd
import pytest

from ml_playground.data.catalog import create_default_dataset_registry
from ml_playground.data.contracts import Dataset
from ml_playground.models.specifications import ProblemType


@pytest.fixture()
def registry():
    return create_default_dataset_registry()


@pytest.mark.parametrize(
    ("dataset_id", "display_name"),
    [
        ("iris", "Iris"),
        ("wine", "Wine"),
        ("breast_cancer", "Breast Cancer"),
        ("digits", "Digits"),
    ],
)
def test_sklearn_classification_datasets_have_separate_target_and_names(
    registry, dataset_id, display_name
):
    dataset = registry.build(dataset_id)

    assert isinstance(dataset, Dataset)
    assert dataset.dataset_name == display_name
    assert dataset.problem_type is ProblemType.CLASSIFICATION
    assert dataset.X.shape[0] == len(dataset.y)
    assert dataset.feature_names == tuple(dataset.X.columns)
    assert dataset.target_names
    assert "target" not in dataset.X.columns
    assert dataset.y.name == "target"
    assert dataset.n_observations == dataset.X.shape[0]
    assert dataset.n_features == dataset.X.shape[1]
    assert dataset.metadata["source"] == "scikit-learn"
    assert dataset.parameters == {"as_frame": True}


@pytest.mark.parametrize(
    ("dataset_id", "parameters", "problem_type", "expected_features"),
    [
        (
            "make_classification",
            {
                "n_samples": 40,
                "n_features": 4,
                "n_informative": 2,
                "n_redundant": 1,
                "n_repeated": 0,
                "random_state": 17,
            },
            ProblemType.CLASSIFICATION,
            4,
        ),
        ("make_moons", {"n_samples": 40, "noise": 0.2, "random_state": 17}, ProblemType.CLASSIFICATION, 2),
        (
            "make_circles",
            {"n_samples": 40, "factor": 0.4, "noise": 0.05, "random_state": 17},
            ProblemType.CLASSIFICATION,
            2,
        ),
        (
            "make_blobs",
            {"n_samples": 40, "n_features": 3, "centers": 4, "random_state": 17},
            ProblemType.CLUSTERING,
            3,
        ),
        (
            "make_regression",
            {"n_samples": 40, "n_features": 5, "n_informative": 3, "random_state": 17},
            ProblemType.REGRESSION,
            5,
        ),
    ],
)
def test_synthetic_datasets_preserve_shape_configuration_and_problem_type(
    registry, dataset_id, parameters, problem_type, expected_features
):
    dataset = registry.build(dataset_id, parameters)

    assert dataset.problem_type is problem_type
    assert dataset.n_observations == parameters["n_samples"]
    assert dataset.n_features == expected_features
    assert len(dataset.y) == dataset.n_observations
    assert dataset.feature_names == tuple(dataset.X.columns)
    assert dataset.parameters["random_state"] == 17
    assert dataset.parameters["n_samples"] == parameters["n_samples"]
    assert dataset.metadata["source"] == "scikit-learn synthetic generator"


@pytest.mark.parametrize(
    "dataset_id",
    [
        "make_classification",
        "make_moons",
        "make_circles",
        "make_blobs",
        "make_regression",
    ],
)
def test_synthetic_datasets_are_reproducible(registry, dataset_id):
    first = registry.build(dataset_id, {"n_samples": 32, "random_state": 23})
    second = registry.build(dataset_id, {"n_samples": 32, "random_state": 23})

    pd.testing.assert_frame_equal(first.X, second.X)
    pd.testing.assert_series_equal(first.y, second.y)
    assert first.parameters == second.parameters


def test_regression_contract_has_continuous_target_and_regression_problem_type(registry):
    dataset = registry.build(
        "make_regression", {"n_samples": 24, "n_features": 3, "n_informative": 2}
    )

    assert dataset.problem_type is ProblemType.REGRESSION
    assert dataset.y is not None
    assert dataset.y.dtype.kind == "f"
    assert dataset.feature_names == ("feature_0", "feature_1", "feature_2")
    assert dataset.target_names is None


def test_registry_lists_and_retrieves_datasets(registry):
    dataset_ids = {specification.id for specification in registry.list()}

    assert dataset_ids == {
        "iris",
        "wine",
        "breast_cancer",
        "digits",
        "make_classification",
        "make_moons",
        "make_circles",
        "make_blobs",
        "make_regression",
    }
    assert registry.get("iris").display_name == "Iris"


def test_unknown_dataset_is_rejected(registry):
    with pytest.raises(KeyError, match="Unknown dataset id"):
        registry.build("not-a-dataset")


@pytest.mark.parametrize(
    ("dataset_id", "parameters", "error", "message"),
    [
        ("make_moons", {"made_up": 1}, ValueError, "Unknown parameter"),
        ("make_moons", {"n_samples": "40"}, TypeError, "must be of type int"),
        ("make_moons", {"n_samples": 1}, ValueError, "at least 2"),
        ("make_circles", {"factor": 1.0}, ValueError, "at most 0.99"),
        (
            "make_classification",
            {"n_features": 2, "n_informative": 2, "n_redundant": 1},
            ValueError,
            "cannot exceed n_features",
        ),
        (
            "make_regression",
            {"n_features": 2, "n_informative": 3},
            ValueError,
            "cannot exceed n_features",
        ),
        (
            "make_blobs",
            {"n_samples": 3, "centers": 4},
            ValueError,
            "centers cannot exceed n_samples",
        ),
    ],
)
def test_invalid_dataset_parameters_are_rejected(registry, dataset_id, parameters, error, message):
    with pytest.raises(error, match=message):
        registry.build(dataset_id, parameters)


def test_boolean_is_not_accepted_as_an_integer_parameter(registry):
    with pytest.raises(TypeError, match="must be of type int"):
        registry.build("make_moons", {"n_samples": True})
