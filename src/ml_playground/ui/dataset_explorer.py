"""Minimal Streamlit view for selecting and exploring available datasets."""

from __future__ import annotations

from typing import Any

import streamlit as st

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.data.contracts import Dataset
from ml_playground.data.registry import DatasetRegistry
from ml_playground.data.specifications import DatasetSpecification
from ml_playground.models.specifications import ProblemType
from ml_playground.ui.training import render_training_panel
from ml_playground.ui.hyperparameter_explorer import render_hyperparameter_explorer
from ml_playground.ui.widgets import parameter_widget
from ml_playground.visualization.datasets import dataset_scatter

MAX_PLOTTED_FEATURES = 20


def _configure_dataset(specification: DatasetSpecification) -> dict[str, Any]:
    if not specification.parameters:
        return {}
    st.subheader("Synthetic dataset settings")
    return {
        parameter.name: parameter_widget(parameter, key_prefix=f"dataset_{specification.id}")
        for parameter in specification.parameters
    }


def _render_target_summary(dataset: Dataset) -> None:
    st.subheader("Target")
    if dataset.y is None:
        st.write("This dataset has no target column.")
        return
    st.write(f"Target name: `{dataset.y.name or 'target'}`")
    if dataset.target_names:
        st.write("Target labels: " + ", ".join(dataset.target_names))
    if dataset.problem_type is ProblemType.REGRESSION:
        st.dataframe(dataset.y.describe().to_frame(name="target"))
    else:
        st.write(f"Target classes: {dataset.y.nunique()}")
        counts = dataset.y.value_counts().sort_index().rename_axis("target").reset_index(name="count")
        st.dataframe(counts, hide_index=True)


def _render_scatter(dataset: Dataset) -> None:
    if dataset.n_features < 2:
        st.info("At least two features are needed for a 2D scatter plot.")
        return
    if dataset.n_features > MAX_PLOTTED_FEATURES:
        st.info(
            f"This dataset has {dataset.n_features} features. Showing its metadata and preview "
            "without projecting the high-dimensional feature space."
        )
        return

    feature_names = list(dataset.feature_names)
    if dataset.n_features == 2:
        x_feature, y_feature = feature_names
    else:
        x_column, y_column = st.columns(2)
        x_feature = x_column.selectbox("X axis", feature_names, key="dataset_x_feature")
        y_choices = [name for name in feature_names if name != x_feature]
        y_feature = y_column.selectbox("Y axis", y_choices, key="dataset_y_feature")
    st.plotly_chart(dataset_scatter(dataset, x_feature, y_feature), width="stretch")


def render_dataset_explorer(registry: DatasetRegistry = DEFAULT_DATASET_REGISTRY) -> None:
    """Render dataset selection, parameter controls, tabular details, and a simple plot."""
    st.title("ML Playground")
    st.header("Dataset Explorer")
    specifications = registry.list()
    specification_by_id = {specification.id: specification for specification in specifications}
    dataset_id = st.selectbox(
        "Dataset",
        options=list(specification_by_id),
        format_func=lambda value: specification_by_id[value].display_name,
    )
    specification = specification_by_id[dataset_id]
    parameters = _configure_dataset(specification)
    try:
        dataset = registry.build(dataset_id, parameters)
    except (TypeError, ValueError) as error:
        st.error(str(error))
        return

    st.caption(specification.description)
    st.write(f"Problem type: `{dataset.problem_type.value}`")
    st.metric("Rows", dataset.n_observations)
    st.metric("Columns", dataset.n_features)
    st.write("Feature names: " + ", ".join(dataset.feature_names))
    st.subheader("First observations")
    st.dataframe(dataset.X.head())
    _render_target_summary(dataset)
    st.subheader("Dataset metadata")
    st.json({"metadata": dataset.metadata, "parameters": dataset.parameters})
    st.subheader("Feature-space view")
    _render_scatter(dataset)
    st.divider()
    render_training_panel(dataset, dataset_id, parameters, registry)
    render_hyperparameter_explorer(
        current_dataset_id=dataset_id,
        dataset_registry=registry,
    )
