"""Smoke test that executes the Streamlit entry point."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_exceptions() -> None:
    """The app script runs and renders its startup confirmation."""
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path), default_timeout=30).run()

    assert not app.exception
    assert any(element.value == "🧪 ML Playground" for element in app.title)
    assert {element.label for element in app.tabs} == {
        "Explorador de datos",
        "Modelos",
        "Explorador de hiperparámetros",
        "Comparación de modelos",
    }
    model_selector = next(element for element in app.selectbox if element.label == "Modelo")
    assert set(model_selector.options) == {
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "MLPClassifier",
        "XGBoost",
        "K-Means",
    }
