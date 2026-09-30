"""End-to-end AppTests for the self-contained Models section dataset flow."""

import json

import pytest
from streamlit.testing.v1 import AppTest


@pytest.mark.parametrize(
    ("dataset_id", "model_id", "parameter_label", "parameter_value", "result_label"),
    [
        ("iris", "logistic_regression", "C", 0.5, "📊 Métricas"),
        ("make_moons", "decision_tree", "criterion", "entropy", "📊 Métricas"),
        ("make_blobs", "kmeans", "n_clusters", 4, "🎯 Resultado de la agrupación"),
    ],
)
def test_models_section_runs_selected_dataset_and_compatible_model(
    dataset_id: str,
    model_id: str,
    parameter_label: str,
    parameter_value: object,
    result_label: str,
) -> None:
    app = AppTest.from_file("app.py", default_timeout=60).run()

    training_dataset = next(
        item for item in app.selectbox if item.label == "Conjunto de datos para entrenar"
    )
    training_dataset.select(dataset_id).run()
    model_selector = next(item for item in app.selectbox if item.label == "Modelo")

    labels = [item.label for item in app.selectbox]
    assert labels.index("Conjunto de datos para entrenar") < labels.index("Modelo")
    model_selector.select(model_id).run()

    parameter = next(
        item
        for item in (*app.number_input, *app.selectbox)
        if item.label == parameter_label
    )
    if hasattr(parameter, "set_value"):
        parameter.set_value(parameter_value)
    else:
        parameter.select(parameter_value)
    next(item for item in app.button if item.label == "🚀 Entrenar modelo").click().run()

    assert not app.exception, app.exception
    assert result_label in [item.value for item in app.subheader]
    rendered_configs = [json.loads(item.value) for item in app.json]
    assert any(
        configuration.get("conjunto_de_datos") == dataset_id
        and configuration.get("modelo") == model_id
        and configuration.get("parámetros_del_modelo", {}).get(parameter_label)
        == parameter_value
        for configuration in rendered_configs
    )


def test_models_section_filters_datasets_and_models_using_registry_compatibility() -> None:
    app = AppTest.from_file("app.py", default_timeout=60).run()

    dataset_selector = next(
        item for item in app.selectbox if item.label == "Conjunto de datos para entrenar"
    )
    assert "Regresión sintética" not in dataset_selector.options
    dataset_selector.select("make_blobs").run()

    model_selector = next(item for item in app.selectbox if item.label == "Modelo")
    assert list(model_selector.options) == ["K-Means"]
