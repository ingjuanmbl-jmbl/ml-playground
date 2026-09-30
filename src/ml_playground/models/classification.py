"""Registered classification model specifications."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression

from ml_playground.models.registry import ModelRegistry
from ml_playground.models.specifications import (
    HyperparameterSpec,
    ModelCapability,
    ModelSpecification,
    ParameterRule,
    ProblemType,
)


def logistic_regression_specification() -> ModelSpecification:
    """Return the declarative specification for scikit-learn Logistic Regression."""
    incompatible_solvers = {"L1": ("lbfgs", "newton-cg", "newton-cholesky", "sag")}
    parameter_rules = tuple(
        ParameterRule(
            when={"regularization": regularization, "solver": solver},
            forbid=frozenset({"solver"}),
            message=(
                f"Logistic Regression regularization='{regularization}' is incompatible "
                f"with solver='{solver}'."
            ),
        )
        for regularization, solvers in incompatible_solvers.items()
        for solver in solvers
    )
    return ModelSpecification(
        id="logistic_regression",
        display_name="Logistic Regression",
        problem_type=ProblemType.CLASSIFICATION,
        estimator_factory=_make_logistic_regression,
        hyperparameters=(
            HyperparameterSpec(
                "C", float, default=1.0, minimum=0.001, maximum=100.0,
                description="Inverse regularization strength; smaller values mean stronger regularization.",
            ),
            HyperparameterSpec(
                "regularization", str, default="L2",
                choices=("L1", "L2"),
                description="L1 encourages sparse coefficients; L2 shrinks coefficients smoothly.",
            ),
            HyperparameterSpec(
                "solver", str, default="lbfgs",
                choices=("lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"),
                description="Optimization algorithm.",
            ),
            HyperparameterSpec(
                "max_iter", int, default=1000, minimum=50, maximum=10000, step=50,
                description="Maximum number of optimization iterations.",
            ),
            HyperparameterSpec(
                "random_state", int, default=42, minimum=0, maximum=2**32 - 1,
                description="Execution seed propagated by the generic training runner.",
            ),
        ),
        capabilities=frozenset(
            {
                ModelCapability.PREDICT,
                ModelCapability.PREDICT_PROBA,
                ModelCapability.DECISION_FUNCTION,
                ModelCapability.COEFFICIENTS,
            }
        ),
        requires_scaling=True,
        description=(
            "Linear classifier with regularization. Inputs are standardized by the shared "
            "training pipeline. L1 and L2 are represented by l1_ratio=1 and l1_ratio=0. "
            "Coefficients describe signed linear effects in scaled feature space."
        ),
        parameter_rules=parameter_rules,
    )


def create_default_model_registry() -> ModelRegistry:
    """Create the application's initial model catalog."""
    return ModelRegistry([logistic_regression_specification()])


def _make_logistic_regression(
    *,
    C: float,
    regularization: str,
    solver: str,
    max_iter: int,
    random_state: int,
) -> LogisticRegression:
    """Map the teaching-friendly L1/L2 choice to the scikit-learn 1.8 API."""
    ratio_by_regularization = {"L1": 1.0, "L2": 0.0}
    try:
        l1_ratio = ratio_by_regularization[regularization]
    except KeyError as error:
        raise ValueError(f"Unsupported regularization '{regularization}'.") from error
    return LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )


DEFAULT_MODEL_REGISTRY = create_default_model_registry()

__all__ = [
    "DEFAULT_MODEL_REGISTRY",
    "create_default_model_registry",
    "logistic_regression_specification",
]
