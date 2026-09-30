"""Declarative model metadata and registration contracts."""

from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import (
    HyperparameterSpec,
    ModelCapability,
    ModelSpecification,
    ParameterRule,
    ProblemType,
)

__all__ = [
    "HyperparameterSpec",
    "ModelCapability",
    "ModelRegistry",
    "ModelSpecification",
    "ParameterRule",
    "ProblemType",
]

