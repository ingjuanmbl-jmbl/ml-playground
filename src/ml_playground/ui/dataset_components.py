"""Shared Streamlit components for choosing and summarizing registered datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import streamlit as st

from ml_playground.data.contracts import Dataset
from ml_playground.data.registry import DatasetRegistry
from ml_playground.data.specifications import DatasetSpecification
from ml_playground.models.specifications import ProblemType
from ml_playground.ui.education import PROBLEM_TYPE_LABELS, dataset_name
from ml_playground.ui.widgets import parameter_widget


@dataclass(frozen=True, slots=True)
class DatasetSelection:
    """A registered dataset and the validated settings used to build it."""

    dataset_id: str
    specification: DatasetSpecification
    parameters: dict[str, Any]
    dataset: Dataset


def is_model_compatible(problem_type: ProblemType, model_problem_type: ProblemType) -> bool:
    """Mirror the runner's generic compatibility contract for selector filtering."""
    if model_problem_type is ProblemType.CLUSTERING:
        return problem_type is not ProblemType.REGRESSION
    return model_problem_type is problem_type


def render_dataset_parameters(
    specification: DatasetSpecification, *, key_prefix: str
) -> dict[str, Any]:
    """Render only parameters declared by the selected dataset specification."""
    if not specification.parameters:
        return {}
    st.subheader("Parámetros de generación")
    return {
        parameter.name: parameter_widget(parameter, key_prefix=key_prefix)
        for parameter in specification.parameters
    }


def build_registered_dataset(
    registry: DatasetRegistry,
    dataset_id: str,
    parameters: dict[str, Any],
) -> Dataset:
    """Build through the shared registry, keeping dataset construction out of widgets."""
    return registry.build(dataset_id, parameters)


def render_dataset_summary(
    dataset: Dataset,
    specification: DatasetSpecification,
    parameters: dict[str, Any],
) -> None:
    """Show concise dataset context, including generation settings when present."""
    st.caption(
        f"**{dataset_name(dataset.dataset_name, specification.display_name)}** · "
        f"{PROBLEM_TYPE_LABELS[dataset.problem_type]} · "
        f"{dataset.n_observations} observaciones · {dataset.n_features} variables"
    )
    target_name = dataset.y.name if dataset.y is not None else None
    st.write(f"**Objetivo:** {target_name or 'sin variable objetivo'}")
    if dataset.y is not None:
        labels = dataset.target_names or tuple(str(value) for value in dataset.y.dropna().unique())
        if dataset.problem_type is ProblemType.CLUSTERING:
            st.caption(
                f"Etiquetas de referencia disponibles: {len(labels)}; "
                "el entrenamiento de clustering no las utiliza."
            )
        elif dataset.problem_type is ProblemType.CLASSIFICATION:
            st.caption(f"Clases: {len(labels)} · " + ", ".join(map(str, labels)))
    if parameters:
        st.caption(
            "Parámetros de generación: "
            + ", ".join(f"{name}={value}" for name, value in parameters.items())
        )
        with st.expander("Parámetros de generación del dataset"):
            st.json(parameters)


__all__ = [
    "DatasetSelection",
    "build_registered_dataset",
    "is_model_compatible",
    "render_dataset_parameters",
    "render_dataset_summary",
]
