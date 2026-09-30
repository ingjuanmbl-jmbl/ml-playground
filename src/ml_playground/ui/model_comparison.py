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
from ml_playground.ui.education import MODEL_GUIDES, dataset_name
from ml_playground.ui.widgets import parameter_widget
from ml_playground.visualization.model_comparison import comparison_metric_figure


def render_model_comparison(
    *,
    model_registry: ModelRegistry = DEFAULT_MODEL_REGISTRY,
    dataset_registry: DatasetRegistry = DEFAULT_DATASET_REGISTRY,
) -> None:
    """Render model selection, shared split settings, and comparison outputs."""
    st.header("Comparación de modelos")
    st.caption(
        "ℹ️ Los modelos usan la misma partición para facilitar la comparación. Esto no sustituye "
        "una validación cruzada ni una evaluación sobre datos independientes."
    )
    datasets = [item for item in dataset_registry.list() if item.problem_type is ProblemType.CLASSIFICATION]
    models = [item for item in model_registry.list() if item.problem_type is ProblemType.CLASSIFICATION]
    dataset_by_id = {item.id: item for item in datasets}
    model_by_id = {item.id: item for item in models}
    dataset_id = st.selectbox(
        "Conjunto de datos para comparar",
        options=list(dataset_by_id),
        format_func=lambda value: dataset_name(value, dataset_by_id[value].display_name),
        key="comparison_dataset_id",
    )
    dataset_spec = dataset_by_id[dataset_id]
    dataset_parameters = {
        parameter.name: parameter_widget(parameter, key_prefix=f"comparison_dataset_{dataset_id}")
        for parameter in dataset_spec.parameters
    }
    selected_ids = st.multiselect(
        "Modelos (selecciona dos o más)",
        options=list(model_by_id),
        default=[item.id for item in models[:2]],
        format_func=lambda value: model_by_id[value].display_name,
        key="comparison_model_ids",
    )
    with st.expander("📚 Qué hace cada modelo y sus valores predeterminados", expanded=False):
        for model_id in selected_ids:
            spec = model_by_id[model_id]
            explanation, usefulness = MODEL_GUIDES.get(
                model_id,
                ("Este estimador realiza la tarea indicada por su tipo de problema.", "Explóralo con distintos datasets y configuraciones."),
            )
            st.markdown(f"**{spec.display_name}**")
            st.caption("Qué hace")
            st.write(explanation)
            st.caption("Cuándo puede resultar útil")
            st.write(usefulness)
            st.caption("Hiperparámetros utilizados")
            st.json(
                {item.name: item.default for item in spec.hyperparameters if item.default is not UNSET}
            )
    test_size = st.slider("Proporción reservada para prueba", 0.1, 0.5, 0.2, 0.05, key="comparison_test_size")
    random_state = st.number_input(
        "Semilla aleatoria", min_value=0, max_value=2**32 - 1, value=42, step=1,
        key="comparison_random_state",
    )
    stratify = st.checkbox("Mantener proporción de clases (estratificar)", value=True, key="comparison_stratify")
    if st.button("Ejecutar comparación", key="run_model_comparison"):
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
                if warning.category.__name__ == "ConvergenceWarning":
                    st.warning("El optimizador de un modelo alcanzó su límite de iteraciones antes de converger.")
                else:
                    st.warning(f"Aviso del modelo ({warning.category.__name__}).")
                with st.expander("Detalle técnico del aviso"):
                    st.code(str(warning.message))
        except (KeyError, TypeError, ValueError) as error:
            st.error("No se pudo completar la comparación. Verifica la selección y la configuración.")
            with st.expander("Detalle técnico del error"):
                st.code(str(error))

    comparison: ModelComparison | None = st.session_state.get("model_comparison_result")
    if comparison is None or comparison.dataset_id != dataset_id:
        return
    rows: list[dict[str, Any]] = []
    for result in comparison.models:
        rows.append(
            {
                "Modelo": result.model_name,
                "Exactitud (Accuracy)": _display_metric(result.metrics.get("accuracy")),
                "Precisión (Precision)": _display_metric(result.metrics.get("precision")),
                "Sensibilidad (Recall)": _display_metric(result.metrics.get("recall")),
                "Puntuación F1": _display_metric(result.metrics.get("f1")),
                "ROC-AUC": _display_metric(result.metrics.get("roc_auc")),
                "Tiempo de entrenamiento (s)": result.training_seconds,
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
        st.caption(f"ROC-AUC no disponible — {reason}")
    for metric in ("accuracy", "f1", "roc_auc", "training_seconds"):
        if metric != "roc_auc" or any(result.metrics.get(metric) is not None for result in comparison.models):
            st.plotly_chart(comparison_metric_figure(comparison, metric), width="stretch")


def _display_metric(value: float | None) -> float | str:
    return "No disponible" if value is None else value


__all__ = ["render_model_comparison"]
