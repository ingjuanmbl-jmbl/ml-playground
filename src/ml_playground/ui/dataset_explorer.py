"""Spanish educational Streamlit experience for datasets, models, and experiments."""

from __future__ import annotations

import streamlit as st

from ml_playground.data.catalog import DEFAULT_DATASET_REGISTRY
from ml_playground.data.contracts import Dataset
from ml_playground.data.registry import DatasetRegistry
from ml_playground.models.specifications import ProblemType
from ml_playground.ui.education import (
    DATASET_GUIDES,
    PROBLEM_TYPE_LABELS,
    dataset_name,
)
from ml_playground.ui.hyperparameter_explorer import render_hyperparameter_explorer
from ml_playground.ui.model_comparison import render_model_comparison
from ml_playground.ui.training import render_training_panel
from ml_playground.ui.dataset_components import (
    build_registered_dataset,
    render_dataset_parameters,
)
from ml_playground.visualization.datasets import dataset_scatter

MAX_PLOTTED_FEATURES = 20


def _render_target_summary(dataset: Dataset) -> None:
    st.subheader("Variable objetivo")
    if dataset.y is None:
        st.write("Este dataset no contiene una variable objetivo.")
        return
    st.write(f"Nombre de la variable objetivo: `{dataset.y.name or 'objetivo'}`")
    if dataset.target_names:
        st.write("Etiquetas de clase: " + ", ".join(dataset.target_names))
    if dataset.problem_type is ProblemType.REGRESSION:
        summary = dataset.y.describe().to_frame(name="Objetivo")
        summary.index.name = "Estadístico"
        st.dataframe(summary, width="stretch")
    else:
        st.write(f"Cantidad de clases observadas: {dataset.y.nunique()}")
        counts = (
            dataset.y.value_counts()
            .sort_index()
            .rename_axis("Clase objetivo")
            .reset_index(name="Observaciones")
        )
        st.dataframe(counts, hide_index=True, width="stretch")


def _render_scatter(dataset: Dataset) -> None:
    if dataset.n_features < 2:
        st.info("Se necesitan al menos dos variables predictoras para un gráfico 2D.")
        return
    if dataset.n_features > MAX_PLOTTED_FEATURES:
        st.info(
            f"El dataset tiene {dataset.n_features} variables. Se muestra la vista tabular y sus "
            "metadatos, sin proyectar las dimensiones altas a un gráfico 2D."
        )
        return
    feature_names = list(dataset.feature_names)
    if dataset.n_features == 2:
        x_feature, y_feature = feature_names
    else:
        x_column, y_column = st.columns(2)
        x_feature = x_column.selectbox("Variable del eje X", feature_names, key="dataset_x_feature")
        y_choices = [name for name in feature_names if name != x_feature]
        y_feature = y_column.selectbox("Variable del eje Y", y_choices, key="dataset_y_feature")
    st.plotly_chart(dataset_scatter(dataset, x_feature, y_feature), width="stretch")


def render_dataset_explorer(registry: DatasetRegistry = DEFAULT_DATASET_REGISTRY) -> None:
    """Render an educational, four-section Spanish experience using existing workflows."""
    st.title("🧪 ML Playground")
    st.subheader("Laboratorio interactivo de Machine Learning")
    st.write(
        "Experimenta con datasets, modelos e hiperparámetros y observa visualmente cómo cambian "
        "sus resultados. La interfaz conecta exploración, entrenamiento y evaluación para apoyar "
        "el aprendizaje práctico."
    )
    with st.expander("¿Cómo usar este laboratorio?", expanded=True):
        st.markdown(
            "1. Selecciona un dataset.\n"
            "2. Selecciona un modelo.\n"
            "3. Ajusta sus hiperparámetros.\n"
            "4. Ejecuta el entrenamiento.\n"
            "5. Analiza métricas y visualizaciones.\n"
            "6. Explora cómo cambia el modelo al modificar sus parámetros."
        )

    data_tab, model_tab, explorer_tab, comparison_tab = st.tabs(
        ["Explorador de datos", "Modelos", "Explorador de hiperparámetros", "Comparación de modelos"]
    )
    with data_tab:
        st.header("Explorador de datos")
        st.caption("Examina las variables, el objetivo y la estructura antes de entrenar un modelo.")
        specifications = registry.list()
        specification_by_id = {item.id: item for item in specifications}
        dataset_id = st.selectbox(
            "Conjunto de datos",
            options=list(specification_by_id),
            format_func=lambda value: dataset_name(value, specification_by_id[value].display_name),
            key="dataset_explorer_id",
        )
        specification = specification_by_id[dataset_id]
        parameters = render_dataset_parameters(
            specification, key_prefix=f"dataset_{specification.id}"
        )
        try:
            dataset = build_registered_dataset(registry, dataset_id, parameters)
        except (TypeError, ValueError) as error:
            st.error(f"No se pudo cargar el conjunto de datos. Revisa su configuración: {error}")
            return

        st.write(f"**Tipo de problema:** {PROBLEM_TYPE_LABELS[dataset.problem_type]}")
        row_column, feature_column = st.columns(2)
        row_column.metric("Observaciones", dataset.n_observations)
        feature_column.metric("Variables predictoras", dataset.n_features)
        st.write("**Variables predictoras:** " + ", ".join(dataset.feature_names))
        st.subheader("Primeras observaciones")
        st.dataframe(dataset.X.head(), width="stretch")
        _render_target_summary(dataset)
        with st.expander("💡 ¿Qué estamos viendo?", expanded=True):
            st.write(DATASET_GUIDES.get(dataset_id, specification.description))
            if specification.parameters:
                st.caption(
                    "En datasets sintéticos, n_samples controla el tamaño, noise agrega dispersión "
                    "cuando aplica y random_state permite reproducir la generación."
                )
        with st.expander("Metadatos y parámetros registrados"):
            st.json({"metadatos": dataset.metadata, "parámetros": dataset.parameters})
        st.subheader("Vista de las variables")
        _render_scatter(dataset)

    with model_tab:
        st.header("Modelos")
        st.caption(
            "Elige un estimador registrado, revisa sus controles y ejecuta el flujo de entrenamiento y evaluación."
        )
        render_training_panel(registry)

    with explorer_tab:
        render_hyperparameter_explorer(
            current_dataset_id=dataset_id if "dataset" in locals() else registry.list()[0].id,
            dataset_registry=registry,
        )
    with comparison_tab:
        render_model_comparison(dataset_registry=registry)


__all__ = ["render_dataset_explorer"]
