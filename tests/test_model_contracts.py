"""Tests for model, parameter, registry, runner, and result contracts."""

from datetime import datetime, timezone

import pytest
from sklearn.pipeline import Pipeline

from ml_playground.evaluation.results import ClassificationResult, ClusteringResult
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import (
    HyperparameterSpec,
    ModelCapability,
    ModelSpecification,
    ParameterRule,
    ProblemType,
)
from ml_playground.training.contracts import (
    TrainingConfiguration,
    TrainingOutput,
    TrainingRequest,
    TrainingRunner,
)


class ExampleEstimator:
    """Small test double used to verify factory invocation, not a project model."""

    def __init__(self, *, strength: float, mode: str = "simple") -> None:
        self.strength = strength
        self.mode = mode


def make_specification() -> ModelSpecification:
    return ModelSpecification(
        id="test-example",
        display_name="Test Example",
        problem_type=ProblemType.CLASSIFICATION,
        estimator_factory=ExampleEstimator,
        hyperparameters=(
            HyperparameterSpec(
                "strength", float, default=1.0, minimum=0.0, maximum=2.0, step=0.5
            ),
            HyperparameterSpec("mode", str, default="simple", choices=("simple", "complex")),
            HyperparameterSpec("label", str, optional=True),
            HyperparameterSpec("detail", str, optional=True),
        ),
        capabilities=frozenset({ModelCapability.PREDICT, ModelCapability.PREDICT_PROBA}),
        requires_scaling=True,
        description="A test-only factory specification.",
        parameter_rules=(
            ParameterRule(
                when={"mode": "complex"},
                require=frozenset({"detail"}),
                message="Complex mode requires detail.",
            ),
            ParameterRule(
                when={"mode": "simple"},
                forbid=frozenset({"label"}),
                message="Simple mode does not use label.",
            ),
        ),
    )


def test_model_specification_carries_contract_fields() -> None:
    specification = make_specification()

    assert specification.id == "test-example"
    assert specification.display_name == "Test Example"
    assert specification.problem_type is ProblemType.CLASSIFICATION
    assert specification.requires_scaling is True
    assert ModelCapability.PREDICT_PROBA in specification.capabilities
    assert specification.description


def test_hyperparameter_spec_validates_types_ranges_choices_and_step() -> None:
    strength = make_specification().hyperparameters[0]

    strength.validate(1.5)
    with pytest.raises(TypeError, match="type float"):
        strength.validate("1.0")
    with pytest.raises(ValueError, match="at most"):
        strength.validate(2.5)
    with pytest.raises(ValueError, match="follow step"):
        strength.validate(1.25)


def test_model_parameter_validation_inserts_defaults_and_omits_optional_values() -> None:
    specification = make_specification()

    assert specification.validate_parameters({}) == {"strength": 1.0, "mode": "simple"}
    estimator = specification.build_estimator({"strength": 1.5})
    assert isinstance(estimator, ExampleEstimator)
    assert estimator.strength == 1.5


@pytest.mark.parametrize(
    ("parameters", "error", "message"),
    [
        ({"unknown": 1}, ValueError, "Unknown parameter"),
        ({"strength": "high"}, TypeError, "type float"),
        ({"strength": 3.0}, ValueError, "at most"),
        ({"mode": "unsupported"}, ValueError, "must be one of"),
        ({"mode": "simple", "label": "custom"}, ValueError, "does not use label"),
        ({"mode": "complex"}, ValueError, "Complex mode requires detail"),
    ],
)
def test_invalid_parameter_configurations_fail(parameters, error, message) -> None:
    with pytest.raises(error, match=message):
        make_specification().validate_parameters(parameters)


def test_parameter_rule_accepts_valid_conditional_combination() -> None:
    parameters = make_specification().validate_parameters(
        {"mode": "complex", "detail": "full"}
    )

    assert parameters["detail"] == "full"


def test_required_parameter_without_default_must_be_supplied() -> None:
    specification = ModelSpecification(
        id="required-test",
        display_name="Required Test",
        problem_type=ProblemType.CLASSIFICATION,
        estimator_factory=ExampleEstimator,
        hyperparameters=(HyperparameterSpec("strength", float),),
    )

    with pytest.raises(ValueError, match="Missing required parameter 'strength'"):
        specification.validate_parameters({})


def test_model_registry_builds_from_registered_specification() -> None:
    specification = make_specification()
    registry = ModelRegistry([specification])

    estimator = registry.build("test-example", {"strength": 0.5})

    assert isinstance(registry.get("test-example"), ModelSpecification)
    assert isinstance(estimator, ExampleEstimator)
    assert estimator.strength == 0.5
    assert registry.list() == (specification,)


def test_registry_rejects_duplicate_or_unknown_ids() -> None:
    specification = make_specification()
    registry = ModelRegistry([specification])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(specification)
    with pytest.raises(KeyError, match="Unknown model id"):
        registry.get("missing")


def test_training_request_and_output_are_framework_independent_contracts() -> None:
    request = TrainingRequest(
        dataset_id="test-data",
        model_id="test-example",
        model_parameters={"strength": 1.0},
    )
    output = TrainingOutput(
        trained_model=Pipeline([("estimator", ExampleEstimator(strength=1.0))]),
        predictions=[1],
        probabilities=[[0.2, 0.8]],
        scores=[0.8],
        training_seconds=0.01,
        metadata={"seed": 7},
        configuration=TrainingConfiguration(
            dataset_id="test-data",
            model_id="test-example",
            dataset_parameters={},
            model_parameters={"strength": 1.0},
            test_size=0.2,
            split_random_state=42,
            estimator_random_state=None,
            stratified=True,
        ),
    )

    assert request.dataset_id == "test-data"
    assert request.model_id == "test-example"
    assert output.predictions == [1]
    assert output.probabilities == [[0.2, 0.8]]
    assert output.scores == [0.8]
    assert output.configuration.model_parameters == {"strength": 1.0}

    class ExampleRunner:
        def run(self, training_request: TrainingRequest) -> TrainingOutput:
            return output

    assert isinstance(ExampleRunner(), TrainingRunner)


def test_classification_and_clustering_results_share_only_common_fields() -> None:
    common = {
        "model_id": "test-example",
        "model_name": "Test Example",
        "configuration": {},
        "training_seconds": 0.01,
        "timestamp": datetime.now(timezone.utc),
    }
    classification = ClassificationResult(
        **common,
        problem_type=ProblemType.CLASSIFICATION,
        predictions=[0, 1],
        probabilities=[[0.9, 0.1], [0.2, 0.8]],
        metrics={"accuracy": 1.0},
    )
    clustering = ClusteringResult(
        **common,
        problem_type=ProblemType.CLUSTERING,
        cluster_labels=[0, 0, 1],
        noise_mask=[False, False, True],
        centroids=[[1.0, 2.0]],
        metrics={"silhouette": 0.5},
    )

    assert classification.predictions == [0, 1]
    assert classification.probabilities is not None
    assert clustering.cluster_labels == [0, 0, 1]
    assert clustering.noise_mask == [False, False, True]
    assert clustering.centroids == [[1.0, 2.0]]


def test_result_contract_rejects_wrong_problem_type() -> None:
    with pytest.raises(ValueError, match="requires classification"):
        ClassificationResult(
            model_id="test-example",
            model_name="Test Example",
            problem_type=ProblemType.CLUSTERING,
            configuration={},
            training_seconds=0.01,
            timestamp=datetime.now(timezone.utc),
            predictions=[],
        )
