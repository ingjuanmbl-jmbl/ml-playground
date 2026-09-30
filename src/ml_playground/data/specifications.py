"""Declarative dataset specifications and configuration validation."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from ml_playground.data.contracts import Dataset
from ml_playground.models.specifications import (
    HyperparameterSpec,
    ParameterRule,
    ProblemType,
    validate_parameter_values,
)

DatasetFactory: TypeAlias = Callable[..., Dataset]
ConfigurationValidator: TypeAlias = Callable[[Mapping[str, Any]], None]


@dataclass(frozen=True, slots=True)
class DatasetSpecification:
    """Declarative metadata, parameters, and factory for one dataset."""

    id: str
    display_name: str
    problem_type: ProblemType
    factory: DatasetFactory
    parameters: tuple[HyperparameterSpec, ...] = ()
    description: str = ""
    parameter_rules: tuple[ParameterRule, ...] = ()
    configuration_validator: ConfigurationValidator | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.display_name.strip():
            raise ValueError("Dataset id and display name cannot be empty.")
        if not callable(self.factory):
            raise TypeError("Dataset factory must be callable.")
        parameter_names = [parameter.name for parameter in self.parameters]
        if len(parameter_names) != len(set(parameter_names)):
            raise ValueError("Dataset parameter names must be unique.")
        known = set(parameter_names)
        for rule in self.parameter_rules:
            referenced = set(rule.when) | set(rule.require) | set(rule.forbid)
            if unknown := referenced - known:
                raise ValueError(f"Parameter rule references unknown parameters: {sorted(unknown)}")

    def validate_parameters(self, values: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Validate known parameters, apply defaults, and check cross-parameter constraints."""
        validated = validate_parameter_values(
            self.parameters,
            values,
            entity_type="dataset",
            entity_id=self.id,
            rules=self.parameter_rules,
        )
        if self.configuration_validator is not None:
            self.configuration_validator(validated)
        return validated

    def build(self, values: Mapping[str, Any] | None = None) -> Dataset:
        """Validate a configuration and call this dataset's factory."""
        parameters = self.validate_parameters(values)
        dataset = self.factory(**parameters)
        if not isinstance(dataset, Dataset):
            raise TypeError(f"Factory for dataset '{self.id}' must return a Dataset.")
        if dataset.problem_type is not self.problem_type:
            raise ValueError(f"Factory for dataset '{self.id}' returned the wrong problem type.")
        return dataset
