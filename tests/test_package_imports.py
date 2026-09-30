"""Smoke tests for the initial package structure."""

from importlib import import_module


MODULES = (
    "ml_playground",
    "ml_playground.data",
    "ml_playground.data.loaders",
    "ml_playground.data.generators",
    "ml_playground.preprocessing",
    "ml_playground.preprocessing.pipelines",
    "ml_playground.models",
    "ml_playground.models.registry",
    "ml_playground.models.classification",
    "ml_playground.models.clustering",
    "ml_playground.training.runner",
    "ml_playground.evaluation.classification",
    "ml_playground.evaluation.clustering",
    "ml_playground.visualization.data",
    "ml_playground.visualization.classification",
    "ml_playground.visualization.clustering",
    "ml_playground.experiments.session",
)


def test_project_modules_import() -> None:
    """Every planned module boundary is importable."""
    for module_name in MODULES:
        assert import_module(module_name) is not None
