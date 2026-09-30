"""In-memory registry for declarative model specifications."""

from __future__ import annotations

from collections.abc import Iterable

from ml_playground.models.specifications import ModelSpecification


class ModelRegistry:
    """Index model specifications by stable id and construct their estimators."""

    def __init__(self, specifications: Iterable[ModelSpecification] = ()) -> None:
        self._specifications: dict[str, ModelSpecification] = {}
        for specification in specifications:
            self.register(specification)

    def register(self, specification: ModelSpecification) -> None:
        """Add a specification; duplicate ids are rejected to avoid silent replacement."""
        if specification.id in self._specifications:
            raise ValueError(f"Model id '{specification.id}' is already registered.")
        self._specifications[specification.id] = specification

    def get(self, model_id: str) -> ModelSpecification:
        """Return a specification by id or raise a helpful error."""
        try:
            return self._specifications[model_id]
        except KeyError as error:
            raise KeyError(f"Unknown model id '{model_id}'.") from error

    def build(self, model_id: str, parameters: dict[str, object] | None = None) -> object:
        """Construct an estimator through its specification."""
        return self.get(model_id).build_estimator(parameters)

    def list(self) -> tuple[ModelSpecification, ...]:
        """Return registered specifications in registration order."""
        return tuple(self._specifications.values())
