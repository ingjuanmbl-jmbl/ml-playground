"""Smoke test that executes the Streamlit entry point."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_exceptions() -> None:
    """The app script runs and renders its startup confirmation."""
    app_path = Path(__file__).resolve().parents[1] / "app.py"

    app = AppTest.from_file(str(app_path)).run()

    assert not app.exception
    assert any(element.value == "ML Playground" for element in app.title)
