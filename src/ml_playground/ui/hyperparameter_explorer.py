"""Streamlit controls and rendering for the independent sensitivity Explorer service."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from ml_playground.data.registry import DatasetRegistry
from ml_playground.experiments.hyperparameter_explorer import (
    CLASSIFICATION_METRICS,
    CLUSTERING_METRICS,
    HyperparameterExploration,
    HyperparameterExplorer,
    is_explorable,
)
from ml_playground.models.catalog import DEFAULT_MODEL_REGISTRY
from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import ProblemType, UNSET
from ml_playground.ui.education import dataset_name, metric_label
from ml_playground.visualization.hyperparameter_explorer import hyperparameter_metric_figure

_RESULT_KEY = "hyperparameter_explorer_result"


def _parse_values(raw: str, parameter: Any) -> list[Any]:
    values: list[Any] = []
    types = parameter.value_type if isinstance(parameter.value_type, tuple) else (parameter.value_type,)
    numeric_type = int if int in types else float if float in types else None
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if token.lower() in {"none", "null"}:
            values.append(None)
        elif numeric_type is int:
            values.append(int(token))
        elif numeric_type is float:
            values.append(float(token))
        else:
            raise ValueError(
                "Ingresa valores numéricos separados por comas; usa None para indicar un "
                "valor opcional sin límite."
            )
    return values


def _default_numeric_values(parameter: Any) -> str:
    types = parameter.value_type if isinstance(parameter.value_type, tuple) else (parameter.value_type,)
    if parameter.name == "max_depth" and int in types:
        return "1,5,10"
    if parameter.name == "learning_rate":
        return "0.01,0.1,0.3"
    default = parameter.default if parameter.default is not UNSET else parameter.minimum
    if int in types:
        base = int(default or 1)
        second = min(base * 2, int(parameter.maximum)) if parameter.maximum is not None else base + 1
        first = max(int(parameter.minimum or 1), base // 2)
        return ",".join(map(str, dict.fromkeys((first, base, second))))
    base = float(default or 0.1)
    first = max(float(parameter.minimum or 0.0), base / 2)
    second = min(float(parameter.maximum), base * 2) if parameter.maximum is not None else base * 2
    return ",".join(map(str, dict.fromkeys((first, base, second))))


def _compatible_models(model_registry: ModelRegistry, dataset_problem: ProblemType):
    return [
        model
        for model in model_registry.list()
        if (
            model.problem_type is ProblemType.CLUSTERING
            and dataset_problem is not ProblemType.REGRESSION
        )
        or model.problem_type is dataset_problem
    ]


def _render_result(result: HyperparameterExploration) -> None:
    st.subheader("Resultados de la exploración")
    rows = []
    for run in result.results:
        rows.append(
            {
                "Hiperparámetro": result.hyperparameter,
                "Valor": run.value,
                **{metric_label(name): value for name, value in run.metrics.items()},
                "Tiempo de entrenamiento (s)": run.training_seconds,
            }
        )
        availability = run.metric_availability[result.metric]
        if not availability.available:
            st.caption(f"{metric_label(result.metric)} no disponible para {result.hyperparameter}={run.value}.")
            with st.expander("Motivo técnico de disponibilidad"):
                st.code(availability.reason or "Sin detalle adicional.")
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    scores = [run.metrics[result.metric] for run in result.results]
    st.plotly_chart(
        hyperparameter_metric_figure(
            model_name=result.model_name,
            hyperparameter=result.hyperparameter,
            metric=result.metric,
            values=result.values,
            scores=scores,
        ),
        width="stretch",
        key="hyperparameter_explorer_chart",
    )
    with st.expander("⚙️ Configuración de la exploración"):
        st.json(
            {
                "modelo": result.model_name,
                "conjunto_de_datos": result.dataset_name,
                "hiperparámetro": result.hyperparameter,
                "valores": result.values,
                "parámetros_base": result.base_parameters,
                "parámetros_del_dataset": result.dataset_parameters,
                "proporción_de_prueba": result.test_size,
                "semilla_aleatoria": result.random_state,
                "estratificación": result.stratify,
            }
        )


def render_hyperparameter_explorer(
    *,
    current_dataset_id: str,
    dataset_registry: DatasetRegistry,
    model_registry: ModelRegistry = DEFAULT_MODEL_REGISTRY,
) -> None:
    """Render the form, invoke the non-UI service, and display saved session results."""
    st.header("Explorador de hiperparámetros")
    st.caption(
        "⚠️ Esta herramienta explora la sensibilidad a hiperparámetros. Los resultados dependen del "
        "dataset y de la partición utilizada; por sí solos no constituyen una evaluación definitiva "
        "de generalización."
    )
    dataset_specs = dataset_registry.list()
    dataset_spec_by_id = {item.id: item for item in dataset_specs}
    with st.form("hyperparameter_explorer_form"):
        dataset_id = st.selectbox(
            "Conjunto de datos para explorar",
            options=list(dataset_spec_by_id),
            index=list(dataset_spec_by_id).index(current_dataset_id),
            format_func=lambda value: dataset_name(value, dataset_spec_by_id[value].display_name),
        )
        dataset_spec = dataset_spec_by_id[dataset_id]
        dataset_parameters = {
            parameter.name: _dataset_parameter_widget(parameter, dataset_id)
            for parameter in dataset_spec.parameters
        }
        try:
            dataset = dataset_registry.build(dataset_id, dataset_parameters)
            model_options = _compatible_models(model_registry, dataset.problem_type)
            model_by_id = {item.id: item for item in model_options}
        except (TypeError, ValueError) as error:
            st.error("No se pudo preparar el dataset o encontrar modelos compatibles.")
            with st.expander("Detalle técnico del error"):
                st.code(str(error))
            model_options = []
            model_by_id = {}

        if not model_options:
            st.info("No hay modelos registrados compatibles con este tipo de dataset.")
            submitted = st.form_submit_button("Ejecutar exploración", disabled=True)
            parameter = None
            parameter_options = []
            model_id = ""
            values: list[Any] = []
            metric = ""
            test_size = 0.2
            random_state = 42
            stratify = True
        else:
            model_id = st.selectbox(
                "Modelo para explorar",
                options=list(model_by_id),
                format_func=lambda value: model_by_id[value].display_name,
            )
            model = model_by_id[model_id]
            parameter_options = [item for item in model.hyperparameters if is_explorable(item)]
            parameter_by_name = {item.name: item for item in parameter_options}
            if not parameter_options:
                st.info("Este modelo no tiene hiperparámetros escalares disponibles para explorar.")
                parameter = None
                values = []
            else:
                hyperparameter = st.selectbox(
                    "Hiperparámetro",
                    options=list(parameter_by_name),
                    format_func=lambda value: value.replace("_", " ").title(),
                )
                parameter = parameter_by_name[hyperparameter]
                if parameter.choices is not None:
                    values = st.multiselect(
                        "Valores",
                        options=list(parameter.choices),
                        default=list(parameter.choices[: min(3, len(parameter.choices))]),
                        format_func=lambda value: "Sin límite (None)" if value is None else str(value),
                    )
                elif bool in (
                    parameter.value_type
                    if isinstance(parameter.value_type, tuple)
                    else (parameter.value_type,)
                ):
                    values = st.multiselect("Valores", options=[False, True], default=[False, True])
                else:
                    raw_values = st.text_input(
                        "Valores (separados por coma)", value=_default_numeric_values(parameter),
                        help="Escribe al menos dos valores distintos, separados por comas.",
                    )
                    try:
                        values = _parse_values(raw_values, parameter)
                    except ValueError as error:
                        st.error(f"Revisa los valores ingresados: {error}")
                        values = []

            metrics = (
                CLUSTERING_METRICS
                if model.problem_type is ProblemType.CLUSTERING
                else CLASSIFICATION_METRICS
            )
            metric = st.selectbox(
                "Métrica", options=list(metrics), format_func=metric_label
            )
            test_size = (
                st.slider("Proporción reservada para prueba", 0.1, 0.5, 0.2, 0.05)
                if model.problem_type is ProblemType.CLASSIFICATION
                else 0.2
            )
            random_state = int(st.number_input("Semilla aleatoria", 0, 2**32 - 1, 42, 1))
            stratify = st.checkbox(
                "Mantener proporción de clases (estratificar)", value=True,
                disabled=model.problem_type is ProblemType.CLUSTERING,
            )
            submitted = st.form_submit_button("Ejecutar exploración", type="primary")

    if submitted and model_options and parameter is not None:
        try:
            exploration = HyperparameterExplorer(model_registry, dataset_registry).explore(
                dataset_id=dataset_id,
                dataset_parameters=dataset_parameters,
                model_id=model_id,
                hyperparameter=parameter.name,
                values=values,
                metric=metric,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )
        except (KeyError, TypeError, ValueError, RuntimeError) as error:
            st.error("No se pudo ejecutar la exploración. Revisa los valores y la compatibilidad.")
            with st.expander("Detalle técnico del error"):
                st.code(str(error))
        else:
            st.session_state[_RESULT_KEY] = exploration

    previous = st.session_state.get(_RESULT_KEY)
    if previous is not None:
        _render_result(previous)


def _dataset_parameter_widget(parameter: Any, dataset_id: str) -> Any:
    """Render a dataset's declared controls without importing UI into the Explorer service."""
    from ml_playground.ui.widgets import parameter_widget

    return parameter_widget(parameter, key_prefix=f"explorer_dataset_{dataset_id}")


__all__ = ["render_hyperparameter_explorer"]
