"""Streamlit presentation for model training and evaluation workflows."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from ml_playground.data.contracts import Dataset
from ml_playground.data.registry import DatasetRegistry
from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.evaluation.clustering import evaluate_clustering
from ml_playground.evaluation.results import ClassificationResult, ClusteringResult
from ml_playground.models.catalog import DEFAULT_MODEL_REGISTRY
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import ProblemType
from ml_playground.training.contracts import TrainingOutput, TrainingRequest
from ml_playground.training.runner import GenericTrainingRunner
from ml_playground.ui.education import (
    METRIC_HELP,
    MODEL_GUIDES,
    PROBLEM_TYPE_LABELS,
    dataset_name,
    metric_label,
)
from ml_playground.ui.dataset_components import (
    build_registered_dataset,
    is_model_compatible,
    render_dataset_parameters,
    render_dataset_summary,
)
from ml_playground.ui.widgets import parameter_widget
from ml_playground.visualization.classification import (
    coefficient_figure,
    decision_boundary_figure,
    feature_importance_figure,
    loss_curve_figure,
)
from ml_playground.visualization.clustering import clustering_scatter_figure
from ml_playground.visualization.datasets import dataset_scatter


def _class_names(dataset: Dataset, labels: tuple[Any, ...]) -> list[str]:
    names = dataset.target_names or ()
    return [
        names[int(label)]
        if isinstance(label, (int, np.integer)) and 0 <= label < len(names)
        else str(label)
        for label in labels
    ]


def build_training_request(
    *,
    dataset_id: str,
    model_id: str,
    dataset_parameters: dict[str, object],
    model_parameters: dict[str, object],
    test_size: float,
    random_state: int,
) -> TrainingRequest:
    """Translate the form values into the framework-independent training contract."""
    return TrainingRequest(
        dataset_id=dataset_id,
        model_id=model_id,
        dataset_parameters=dataset_parameters,
        model_parameters=model_parameters,
        test_size=test_size,
        random_state=random_state,
    )


def _render_classification_results(
    dataset: Dataset, output: TrainingOutput, result: ClassificationResult
) -> None:
    st.subheader("📊 Métricas")
    metric_columns = st.columns(5)
    for column, metric in zip(
        metric_columns, ("accuracy", "precision", "recall", "f1", "roc_auc"), strict=True
    ):
        value = result.metrics[metric]
        column.metric(metric_label(metric), "No disponible" if value is None else f"{value:.4f}")
        column.caption(METRIC_HELP[metric])
        if metric == "roc_auc" and value is not None:
            column.caption("Promedio macro uno contra el resto" if len(result.class_labels) > 2 else "Clasificación binaria")
        if value is None and metric == "roc_auc":
            column.caption(f"Motivo: {result.metric_availability[metric].reason}")

    with st.expander("💡 ¿Cómo interpretar estas métricas?", expanded=True):
        st.write(
            "Cada valor describe esta configuración y esta partición concreta. F1 resume el equilibrio "
            "entre precisión y sensibilidad; ROC-AUC solo aparece cuando hay scores compatibles. "
            "La matriz conserva el orden de clases mostrado en sus ejes."
        )
    st.subheader("⏱️ Tiempo de entrenamiento")
    st.metric("Duración del ajuste (segundos)", f"{result.training_seconds:.4f}")

    estimator = output.trained_model.named_steps["estimator"]
    if hasattr(estimator, "hidden_layer_sizes") and hasattr(estimator, "coefs_"):
        hidden_layers = tuple(estimator.hidden_layer_sizes)
        approximate_parameters = sum(
            np.asarray(values).size for values in (*estimator.coefs_, *estimator.intercepts_)
        )
        with st.expander("🧠 Detalles de la red neuronal", expanded=True):
            st.write("Arquitectura configurada:", hidden_layers)
            detail_columns = st.columns(4)
            detail_columns[0].metric("Capas ocultas", len(hidden_layers))
            detail_columns[1].metric("Parámetros aproximados", f"{approximate_parameters:,}")
            detail_columns[2].metric("Activación", str(estimator.activation))
            detail_columns[3].metric("Iteraciones realizadas", str(getattr(estimator, "n_iter_", "No disponible")))
            validation_score = getattr(estimator, "best_validation_score_", None)
            if validation_score is not None:
                st.caption(f"Mejor puntuación de validación para parada temprana: {validation_score:.4f}")
            loss_curve = getattr(estimator, "loss_curve_", None)
            if loss_curve is not None and len(loss_curve) > 0:
                st.plotly_chart(loss_curve_figure(loss_curve), width="stretch", key="mlp_loss_curve")

    with st.expander("🔎 Visualizaciones", expanded=True):
        names = _class_names(dataset, result.class_labels)
        st.subheader("Matriz de confusión")
        st.plotly_chart(
            px.imshow(
                result.confusion_matrix,
                x=names,
                y=names,
                text_auto=True,
                labels={"x": "Clase predicha", "y": "Clase real", "color": "Observaciones"},
                title="Predicciones sobre el conjunto de prueba",
            ),
            width="stretch",
            key="evaluation_confusion_matrix",
        )

        if output.coefficients is not None:
            st.subheader("Coeficientes del modelo")
            st.caption(
                "Representan efectos con signo dentro del modelo lineal. Su interpretación depende "
                "de la escala y de la formulación; no son una medida de importancia de variables."
            )
            st.plotly_chart(
                coefficient_figure(
                    output.coefficients,
                    dataset.feature_names,
                    result.class_labels,
                    class_names=names,
                ),
                width="stretch",
                key="logistic_coefficients",
            )

        if output.feature_importances is not None:
            st.subheader("Importancia de variables")
            st.caption(
                "Medida nativa de feature_importances_ proporcionada por el estimador. No expresa "
                "causalidad y no equivale a los coeficientes de Logistic Regression."
            )
            st.plotly_chart(
                feature_importance_figure(output.feature_importances, dataset.feature_names),
                width="stretch",
                key="decision_tree_feature_importances",
            )

        st.subheader("Frontera de decisión")
        if output.X_test is not None and output.y_test is not None and dataset.n_features == 2:
            st.plotly_chart(
                decision_boundary_figure(
                    output.trained_model,
                    output.X_test,
                    output.y_test,
                    result.class_labels,
                    class_names=names,
                ),
                width="stretch",
                key="logistic_decision_boundary",
            )
        else:
            st.info("La frontera directa está disponible cuando el dataset tiene exactamente dos variables.")
            if output.X_test is not None and output.y_test is not None and dataset.n_features > 2:
                feature_names = list(dataset.feature_names)
                x_col, y_col = st.columns(2)
                x_feature = x_col.selectbox("Variable del eje X", feature_names, key="result_x_feature")
                y_feature = y_col.selectbox(
                    "Variable del eje Y",
                    [feature for feature in feature_names if feature != x_feature],
                    key="result_y_feature",
                )
                st.plotly_chart(
                    dataset_scatter(dataset, x_feature, y_feature),
                    width="stretch",
                    key="held_out_feature_preview",
                )

    with st.expander("⚙️ Configuración utilizada"):
        st.json(
            {
                "conjunto_de_datos": output.configuration.dataset_id,
                "parámetros_del_dataset": dict(output.configuration.dataset_parameters),
                "modelo": output.configuration.model_id,
                "parámetros_del_modelo": dict(output.configuration.model_parameters),
                "tamaño_del_conjunto_de_prueba": output.configuration.test_size,
                "random_state": output.configuration.split_random_state,
                "semilla_de_la_partición": output.configuration.split_random_state,
                "escalamiento": "StandardScaler" if output.trained_model.named_steps["preprocessing"] != "passthrough" else "Sin escalamiento",
            }
        )


def _render_clustering_results(
    dataset: Dataset, output: TrainingOutput, result: ClusteringResult
) -> None:
    st.subheader("🎯 Resultado de la agrupación")
    st.caption(
        "K-Means asigna cada observación al centroide más cercano. El resultado depende de la "
        "cantidad de grupos y de la escala de las variables."
    )
    st.info(
        "Este algoritmo realiza agrupamiento no supervisado. Las etiquetas originales del dataset "
        "no se utilizan durante el entrenamiento. Las métricas son medidas internas de estructura "
        "y ninguna determina por sí sola la calidad de una segmentación."
    )
    st.subheader("📊 Métricas")
    columns = st.columns(3)
    for column, name in zip(columns, result.metrics, strict=True):
        value = result.metrics[name]
        column.metric(metric_label(name), "No disponible" if value is None else f"{value:.4f}")
        column.caption(METRIC_HELP.get(name, "Medida interna de estructura del agrupamiento."))
        if value is None:
            column.caption(f"Motivo: {result.metric_availability[name].reason}")
    group_column, time_column = st.columns(2)
    group_column.metric("Grupos encontrados", result.n_clusters)
    time_column.metric("Tiempo de entrenamiento (s)", f"{result.training_seconds:.4f}")

    with st.expander("⚙️ Configuración utilizada"):
        st.json(
            {
                "conjunto_de_datos": output.configuration.dataset_id,
                "modelo": output.configuration.model_id,
                "parámetros_del_modelo": dict(output.configuration.model_parameters),
                "escalamiento": "StandardScaler dentro del pipeline",
            }
        )
    with st.expander("🔎 Visualización y centroides", expanded=True):
        if result.centroids is not None:
            centers = np.asarray(result.centroids)
            st.subheader("Centroides")
            st.caption("Sus coordenadas se expresan en las unidades originales de las variables.")
            st.dataframe(
                pd.DataFrame(
                    centers,
                    columns=dataset.feature_names,
                    index=[f"Grupo {i + 1}" for i in range(len(centers))],
                ),
                width="stretch",
            )
        if dataset.n_features < 2:
            st.info("Se necesitan al menos dos variables para el gráfico de agrupamiento 2D.")
            return
        feature_names = list(dataset.feature_names)
        if len(feature_names) == 2:
            x_feature, y_feature = feature_names
        else:
            x_column, y_column = st.columns(2)
            x_feature = x_column.selectbox("Variable del eje X", feature_names, key="cluster_x_feature")
            y_feature = y_column.selectbox(
                "Variable del eje Y",
                [feature for feature in feature_names if feature != x_feature],
                key="cluster_y_feature",
            )
        if output.cluster_labels is None:
            st.info("El estimador no devolvió etiquetas de grupo.")
            return
        st.plotly_chart(
            clustering_scatter_figure(
                dataset.X, output.cluster_labels, x_feature, y_feature, centroids=result.centroids
            ),
            width="stretch",
            key="clustering_scatter",
        )


def _show_warnings(captured: list[warnings.WarningMessage]) -> None:
    for warning in captured:
        if warning.category.__name__ == "ConvergenceWarning":
            st.warning(
                "El optimizador alcanzó max_iter antes de converger. Revisa la curva de pérdida "
                "y considera ajustar los parámetros."
            )
        else:
            st.warning(f"Aviso del modelo ({warning.category.__name__}).")
        with st.expander("Detalle técnico del aviso"):
            st.code(str(warning.message))


def render_training_panel(
    dataset_registry: DatasetRegistry,
    model_registry: ModelRegistry = DEFAULT_MODEL_REGISTRY,
) -> None:
    """Render a self-contained registered-dataset and model training workflow."""
    st.subheader("1. Selecciona un dataset")
    dataset_specs = dataset_registry.list()
    model_specs = model_registry.list()
    compatible_dataset_specs = [
        dataset_spec
        for dataset_spec in dataset_specs
        if any(
            is_model_compatible(dataset_spec.problem_type, model_spec.problem_type)
            for model_spec in model_specs
        )
    ]
    if not compatible_dataset_specs:
        st.info("No hay datasets compatibles con los modelos registrados.")
        return
    dataset_by_id = {item.id: item for item in compatible_dataset_specs}
    dataset_id = st.selectbox(
        "Conjunto de datos para entrenar",
        options=list(dataset_by_id),
        format_func=lambda value: dataset_name(value, dataset_by_id[value].display_name),
        key="model_training_dataset_id",
    )
    dataset_specification = dataset_by_id[dataset_id]
    dataset_parameters = render_dataset_parameters(
        dataset_specification, key_prefix=f"model_dataset_{dataset_id}"
    )
    try:
        dataset = build_registered_dataset(dataset_registry, dataset_id, dataset_parameters)
    except (KeyError, TypeError, ValueError) as error:
        st.error("No se pudo cargar el dataset seleccionado.")
        with st.expander("Detalle técnico del error"):
            st.code(str(error))
        return
    render_dataset_summary(dataset, dataset_specification, dataset_parameters)

    st.subheader("2. Selecciona un modelo")
    compatible_model_specs = [
        item
        for item in model_specs
        if is_model_compatible(dataset.problem_type, item.problem_type)
    ]
    if not compatible_model_specs:
        st.info("Todavía no hay modelos compatibles con este dataset.")
        return
    spec_by_id = {spec.id: spec for spec in compatible_model_specs}
    model_id = st.selectbox(
        "Modelo",
        list(spec_by_id),
        format_func=lambda identifier: spec_by_id[identifier].display_name,
        key="model_training_model_id",
    )
    specification = spec_by_id[model_id]
    explanation, usefulness = MODEL_GUIDES.get(
        model_id,
        ("Este estimador realiza la tarea indicada por su tipo de problema.", "Explóralo con distintos datos y configuraciones."),
    )
    with st.expander("📚 ¿Qué hace este modelo?", expanded=True):
        st.write(explanation)
    with st.expander("¿Cuándo puede resultar útil?"):
        st.write(usefulness)
    if specification.problem_type is ProblemType.CLUSTERING:
        st.info(
            "El agrupamiento utiliza solo las variables predictoras. Las etiquetas objetivo del "
            "dataset, si existen, no se usan durante el entrenamiento."
        )
    with st.form("training_request_form"):
        st.subheader("Hiperparámetros")
        model_parameters = {
            parameter.name: parameter_widget(parameter, key_prefix=f"{model_id}_{dataset_id}")
            for parameter in specification.hyperparameters
            if parameter.name != "random_state"
        }
        if specification.problem_type is ProblemType.CLUSTERING:
            test_size = 0.2
        else:
            test_size = st.slider(
                "Proporción reservada para prueba",
                min_value=0.1,
                max_value=0.5,
                value=0.2,
                step=0.05,
            )
        random_state = st.number_input(
            "Semilla aleatoria", min_value=0, max_value=2**32 - 1, value=42, step=1
        )
        submitted = st.form_submit_button("🚀 Entrenar modelo", type="primary")

    if not submitted:
        return
    try:
        request = build_training_request(
            dataset_id=dataset_id,
            model_id=model_id,
            dataset_parameters=dataset_parameters,
            model_parameters=model_parameters,
            test_size=float(test_size),
            random_state=int(random_state),
        )
        runner = GenericTrainingRunner(model_registry=model_registry, dataset_registry=dataset_registry)
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            output = runner.run(request)
        _show_warnings(captured)
        if specification.problem_type is ProblemType.CLUSTERING:
            result = evaluate_clustering(output)
            _render_clustering_results(dataset, output, result)
        else:
            result = evaluate_classification(output)
            _render_classification_results(dataset, output, result)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        st.error("No se pudo ejecutar el entrenamiento. Revisa la compatibilidad y los valores indicados.")
        with st.expander("Detalle técnico del error"):
            st.code(str(error))


__all__ = ["build_training_request", "render_training_panel"]
