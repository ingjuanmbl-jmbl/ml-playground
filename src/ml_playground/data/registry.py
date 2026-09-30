"""Central registry for dataset specifications."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ml_playground.data.contracts import Dataset
from ml_playground.data.specifications import DatasetSpecification


class DatasetRegistry:
    """List, look up, validate, and build datasets through registered specifications."""

    def __init__(self, specifications: Iterable[DatasetSpecification] = ()) -> None:
        self._specifications: dict[str, DatasetSpecification] = {}
        for specification in specifications:
            self.register(specification)

    def register(self, specification: DatasetSpecification) -> None:
        """Register a specification, rejecting duplicate IDs."""
        if specification.id in self._specifications:
            raise ValueError(f"Dataset id '{specification.id}' is already registered.")
        self._specifications[specification.id] = specification

    def get(self, dataset_id: str) -> DatasetSpecification:
        """Return a dataset specification by ID."""
        try:
            return self._specifications[dataset_id]
        except KeyError as error:
            raise KeyError(f"Unknown dataset id '{dataset_id}'.") from error

    def build(self, dataset_id: str, parameters: Mapping[str, Any] | None = None) -> Dataset:
        """Validate configuration and construct the selected dataset."""
        return self.get(dataset_id).build(parameters)

    def list(self) -> tuple[DatasetSpecification, ...]:
        """Return specifications in registration order."""
        return tuple(self._specifications.values())
