"""Declarative contracts for model metadata and hyperparameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from math import isclose
from typing import Any, Callable, Mapping, TypeAlias

EstimatorFactory: TypeAlias = Callable[..., Any]
UNSET = object()


class ProblemType(StrEnum):
    """Supported kinds of machine learning problem."""

    CLASSIFICATION = "classification"
    CLUSTERING = "clustering"


class ModelCapability(StrEnum):
    """Operations or outputs an estimator explicitly supports."""

    PREDICT = "predict"
    PREDICT_PROBA = "predict_proba"
    DECISION_FUNCTION = "decision_function"
    FEATURE_IMPORTANCES = "feature_importances"
    COEFFICIENTS = "coefficients"
    CLUSTER_LABELS = "cluster_labels"
    NOISE_LABELS = "noise_labels"
    CENTROIDS = "centroids"


@dataclass(frozen=True, slots=True)
class ParameterRule:
    """A conditional requirement or prohibition among parameter values.

    The rule applies when every ``when`` entry matches the supplied parameter mapping.
    """

    when: Mapping[str, Any]
    require: frozenset[str] = frozenset()
    forbid: frozenset[str] = frozenset()
    message: str = "Parameter combination is not supported."

    def __post_init__(self) -> None:
        if not self.when:
            raise ValueError("A parameter rule must have at least one condition.")
        if self.require & self.forbid:
            raise ValueError("A parameter cannot be both required and forbidden by one rule.")


@dataclass(frozen=True, slots=True)
class HyperparameterSpec:
    """Description and validation rules for one estimator parameter."""

    name: str
    value_type: type[Any] | tuple[type[Any], ...]
    default: Any = UNSET
    minimum: int | float | None = None
    maximum: int | float | None = None
    choices: tuple[Any, ...] | None = None
    step: int | float | None = None
    description: str = ""
    optional: bool = False
    item_type: type[Any] | tuple[type[Any], ...] | None = None
    item_minimum: int | float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Hyperparameter name cannot be empty.")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError(f"Minimum exceeds maximum for parameter '{self.name}'.")
        if self.step is not None and self.step <= 0:
            raise ValueError(f"Step must be positive for parameter '{self.name}'.")
        if self.choices is not None and not self.choices:
            raise ValueError(f"Choices cannot be empty for parameter '{self.name}'.")
        if self.default is not UNSET:
            self.validate(self.default)

    def validate(self, value: Any) -> None:
        """Raise ``ValueError`` when a value does not satisfy this specification."""
        if value is None:
            if self.optional:
                return
            raise ValueError(f"Parameter '{self.name}' is required and cannot be None.")
        if not isinstance(value, self.value_type) or isinstance(value, bool) and self.value_type != bool:
            expected = _type_name(self.value_type)
            raise TypeError(f"Parameter '{self.name}' must be of type {expected}.")
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"Parameter '{self.name}' must be one of {self.choices!r}.")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.minimum is not None and value < self.minimum:
                raise ValueError(f"Parameter '{self.name}' must be at least {self.minimum}.")
            if self.maximum is not None and value > self.maximum:
                raise ValueError(f"Parameter '{self.name}' must be at most {self.maximum}.")
            if self.step is not None:
                origin = self.minimum or 0
                steps = (value - origin) / self.step
                if not isclose(steps, round(steps), abs_tol=1e-9):
                    raise ValueError(
                        f"Parameter '{self.name}' must follow step {self.step} from {origin}."
                    )
        if isinstance(value, (tuple, list)) and self.item_type is not None:
            for index, item in enumerate(value):
                if not isinstance(item, self.item_type) or isinstance(item, bool):
                    raise TypeError(
                        f"Items in parameter '{self.name}' must be of type "
                        f"{_type_name(self.item_type)}."
                    )
                if self.item_minimum is not None and item < self.item_minimum:
                    raise ValueError(
                        f"Items in parameter '{self.name}' must be at least {self.item_minimum} "
                        f"(item {index})."
                    )


@dataclass(frozen=True, slots=True)
class ModelSpecification:
    """Common, immutable metadata and factory contract for an estimator."""

    id: str
    display_name: str
    problem_type: ProblemType
    estimator_factory: EstimatorFactory
    hyperparameters: tuple[HyperparameterSpec, ...] = ()
    capabilities: frozenset[ModelCapability] = frozenset()
    requires_scaling: bool = False
    description: str = ""
    parameter_rules: tuple[ParameterRule, ...] = ()

    def __post_init__(self) -> None:
        if not self.id.strip():
            raise ValueError("Model id cannot be empty.")
        if not self.display_name.strip():
            raise ValueError("Model display name cannot be empty.")
        if not callable(self.estimator_factory):
            raise TypeError("Estimator factory must be callable.")
        names = [parameter.name for parameter in self.hyperparameters]
        if len(names) != len(set(names)):
            raise ValueError("Hyperparameter names must be unique within a model specification.")
        known = set(names)
        for rule in self.parameter_rules:
            referenced = set(rule.when) | set(rule.require) | set(rule.forbid)
            unknown = referenced - known
            if unknown:
                raise ValueError(f"Parameter rule references unknown parameters: {sorted(unknown)}")

    def validate_parameters(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Validate supplied values, insert defaults, and return constructor kwargs."""
        supplied = dict(values or {})
        specs = {parameter.name: parameter for parameter in self.hyperparameters}
        unknown = supplied.keys() - specs.keys()
        if unknown:
            raise ValueError(f"Unknown parameter(s) for model '{self.id}': {sorted(unknown)}")

        validated: dict[str, Any] = {}
        for name, spec in specs.items():
            if name in supplied:
                value = supplied[name]
            elif spec.default is not UNSET:
                value = spec.default
            elif spec.optional:
                continue
            else:
                raise ValueError(f"Missing required parameter '{name}' for model '{self.id}'.")
            spec.validate(value)
            if value is not None:
                validated[name] = value

        for rule in self.parameter_rules:
            if all(validated.get(name) == expected for name, expected in rule.when.items()):
                missing = rule.require - validated.keys()
                forbidden = rule.forbid & validated.keys()
                if missing or forbidden:
                    raise ValueError(
                        f"{rule.message} Missing: {sorted(missing)}; "
                        f"not allowed: {sorted(forbidden)}."
                    )
        return validated

    def build_estimator(self, values: Mapping[str, Any] | None = None) -> Any:
        """Validate parameters and construct the estimator declared by this specification."""
        return self.estimator_factory(**self.validate_parameters(values))


def _type_name(value_type: type[Any] | tuple[type[Any], ...]) -> str:
    types = value_type if isinstance(value_type, tuple) else (value_type,)
    return " or ".join(item.__name__ for item in types)
