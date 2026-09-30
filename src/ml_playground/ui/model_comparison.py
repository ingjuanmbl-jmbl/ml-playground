"""Streamlit controls and display for supervised model comparison."""

from __future__ import annotations

import warnings
from typing import Any

import streamlit as st

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.data.registry import DatasetRegistry
from ml_playground.experiments.model_comparison import ModelComparison, ModelComparisonService
from ml_playground.models.catalog import DEFAULT_MODEL_REGISTRY
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import ProblemType, UNSET
from ml_playground.ui.widgets import parameter_widget
from ml_playground.visualization.model_comparison import comparison_metric_figure


def render_model_comparison(
    *,
    model_registry: ModelRegistry = DEFAULT_MODEL_REGISTRY,
    dataset_registry: DatasetRegistry = DEFAULT_DATASET_REGISTRY,
) -> None:
    """Render model selection, shared split settings, and comparison outputs."""
    st.header("Model Comparison")
    st.caption(
        "All models use the same dataset and train/test split. A single split is not an exhaustive "
        "assessment of generalization; the test set should not support a definitive performance claim."
    )
    datasets = [item for item in dataset_registry.list() if item.problem_type is ProblemType.CLASSIFICATION]
    models = [item for item in model_registry.list() if item.problem_type is ProblemType.CLASSIFICATION]
    dataset_by_id = {item.id: item for item in datasets}
    model_by_id = {item.id: item for item in models}
    dataset_id = st.selectbox(
        "Comparison dataset",
        options=list(dataset_by_id),
        format_func=lambda value: dataset_by_id[value].display_name,
        key="comparison_dataset_id",
    )
    dataset_spec = dataset_by_id[dataset_id]
    dataset_parameters = {
        parameter.name: parameter_widget(parameter, key_prefix=f"comparison_dataset_{dataset_id}")
        for parameter in dataset_spec.parameters
    }
    selected_ids = st.multiselect(
        "Models",
        options=list(model_by_id),
        default=[item.id for item in models[:2]],
        format_func=lambda value: model_by_id[value].display_name,
        key="comparison_model_ids",
    )
    with st.expander("Current model defaults", expanded=False):
        for model_id in selected_ids:
            spec = model_by_id[model_id]
            st.write(spec.display_name)
            st.json(
                {item.name: item.default for item in spec.hyperparameters if item.default is not UNSET}
            )
    test_size = st.slider("Test size", 0.1, 0.5, 0.2, 0.05, key="comparison_test_size")
    random_state = st.number_input(
        "Random state", min_value=0, max_value=2**32 - 1, value=42, step=1,
        key="comparison_random_state",
    )
    stratify = st.checkbox("Stratify split", value=True, key="comparison_stratify")
    if st.button("Run comparison", key="run_model_comparison"):
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                comparison = ModelComparisonService(model_registry, dataset_registry).compare(
                    dataset_id=dataset_id,
                    model_ids=selected_ids,
                    dataset_parameters=dataset_parameters,
                    test_size=test_size,
                    random_state=int(random_state),
                    stratify=stratify,
                )
            st.session_state["model_comparison_result"] = comparison
            for warning in caught:
                st.warning(f"{warning.category.__name__}: {warning.message}")
        except (KeyError, TypeError, ValueError) as error:
            st.error(str(error))

    comparison: ModelComparison | None = st.session_state.get("model_comparison_result")
    if comparison is None or comparison.dataset_id != dataset_id:
        return
    rows: list[dict[str, Any]] = []
    for result in comparison.models:
        rows.append(
            {
                "Model": result.model_name,
                "Accuracy": _display_metric(result.metrics.get("accuracy")),
                "Precision": _display_metric(result.metrics.get("precision")),
                "Recall": _display_metric(result.metrics.get("recall")),
                "F1": _display_metric(result.metrics.get("f1")),
                "ROC-AUC": _display_metric(result.metrics.get("roc_auc")),
                "Training time (s)": result.training_seconds,
            }
        )
    st.dataframe(rows, hide_index=True, width="stretch")
    unavailable = [
        f"{result.model_name}: {result.metric_availability['roc_auc'].reason}"
        for result in comparison.models
        if "roc_auc" in result.metric_availability
        and not result.metric_availability["roc_auc"].available
    ]
    for reason in unavailable:
        st.caption(f"ROC-AUC unavailable — {reason}")
    for metric in ("accuracy", "f1", "roc_auc", "training_seconds"):
        if metric != "roc_auc" or any(result.metrics.get(metric) is not None for result in comparison.models):
            st.plotly_chart(comparison_metric_figure(comparison, metric), width="stretch")


def _display_metric(value: float | None) -> float | str:
    return "N/A" if value is None else value


__all__ = ["render_model_comparison"]
