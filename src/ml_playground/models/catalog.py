"""Default registry catalog for all available model specifications."""

from ml_playground.models.classification import (
    decision_tree_specification,
    logistic_regression_specification,
    mlp_classifier_specification,
    random_forest_specification,
    xgboost_classifier_specification,
)
from ml_playground.models.clustering import kmeans_specification
from ml_playground.models.registry import ModelRegistry


def create_default_model_registry() -> ModelRegistry:
    """Build the shared registry used by model selection and the generic runner."""
    return ModelRegistry(
        [
            logistic_regression_specification(),
            decision_tree_specification(),
            random_forest_specification(),
            mlp_classifier_specification(),
            xgboost_classifier_specification(),
            kmeans_specification(),
        ]
    )


DEFAULT_MODEL_REGISTRY = create_default_model_registry()

__all__ = ["DEFAULT_MODEL_REGISTRY", "create_default_model_registry"]
