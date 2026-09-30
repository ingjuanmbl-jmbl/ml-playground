"""Registered classification model specifications."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

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
    return ModelRegistry(
        [
            logistic_regression_specification(),
            decision_tree_specification(),
            random_forest_specification(),
            mlp_classifier_specification(),
        ]
    )


def decision_tree_specification() -> ModelSpecification:
    """Return the declarative specification for a scikit-learn decision tree."""
    return ModelSpecification(
        id="decision_tree",
        display_name="Decision Tree",
        problem_type=ProblemType.CLASSIFICATION,
        estimator_factory=DecisionTreeClassifier,
        hyperparameters=(
            HyperparameterSpec(
                "criterion", str, default="gini",
                choices=("gini", "entropy", "log_loss"),
                description="Impurity measure used to choose splits.",
            ),
            HyperparameterSpec(
                "splitter", str, default="best", choices=("best", "random"),
                description="Strategy used to select each split.",
            ),
            HyperparameterSpec(
                "max_depth", (int, type(None)), default=None, minimum=1, optional=True,
                description="Maximum tree depth; None allows the tree to grow until other stopping rules apply.",
            ),
            HyperparameterSpec(
                "min_samples_split", int, default=2, minimum=2,
                description="Minimum number of samples required to split an internal node.",
            ),
            HyperparameterSpec(
                "min_samples_leaf", int, default=1, minimum=1,
                description="Minimum number of samples required at a leaf.",
            ),
            HyperparameterSpec(
                "max_features", (str, type(None)), default=None,
                choices=(None, "sqrt", "log2"), optional=True,
                description="Number of features considered at each split: all, sqrt, or log2.",
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
                ModelCapability.FEATURE_IMPORTANCES,
            }
        ),
        requires_scaling=False,
        description=(
            "A tree classifier that partitions feature space with sequential splits. "
            "max_depth limits tree growth and can help control model complexity. "
            "Feature importances are the tree's feature_importances_ values."
        ),
    )


def random_forest_specification() -> ModelSpecification:
    """Return the declarative specification for a scikit-learn random forest."""
    return ModelSpecification(
        id="random_forest",
        display_name="Random Forest",
        problem_type=ProblemType.CLASSIFICATION,
        estimator_factory=RandomForestClassifier,
        hyperparameters=(
            HyperparameterSpec(
                "n_estimators", int, default=100, minimum=1, maximum=1000,
                description="Number of decision trees in the forest.",
            ),
            HyperparameterSpec(
                "max_depth", (int, type(None)), default=None, minimum=1, optional=True,
                description="Maximum depth of each tree; None allows growth until other stopping rules apply.",
            ),
            HyperparameterSpec(
                "min_samples_split", int, default=2, minimum=2,
                description="Minimum samples required to split an internal node.",
            ),
            HyperparameterSpec(
                "min_samples_leaf", int, default=1, minimum=1,
                description="Minimum samples required at a leaf.",
            ),
            HyperparameterSpec(
                "max_features", str, default="sqrt",
                choices=("sqrt", "log2"),
                description="Feature selection strategy used at each split.",
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
                ModelCapability.FEATURE_IMPORTANCES,
            }
        ),
        requires_scaling=False,
        description=(
            "An ensemble that aggregates predictions from multiple decision trees. "
            "The forest exposes native feature_importances_ values and does not require scaling."
        ),
    )


def mlp_classifier_specification() -> ModelSpecification:
    """Return the declarative specification for scikit-learn's MLP classifier."""
    return ModelSpecification(
        id="mlp_classifier",
        display_name="MLPClassifier",
        problem_type=ProblemType.CLASSIFICATION,
        estimator_factory=MLPClassifier,
        hyperparameters=(
            HyperparameterSpec(
                "hidden_layer_sizes", tuple, default=(20, 10),
                choices=((10,), (20,), (20, 10), (50, 25, 10)),
                item_type=int,
                item_minimum=1,
                description=(
                    "Neurons per hidden layer: for example (20, 10) means two hidden layers "
                    "with 20 and 10 neurons."
                ),
            ),
            HyperparameterSpec(
                "activation", str, default="relu",
                choices=("identity", "logistic", "tanh", "relu"),
                description="Activation function for hidden layers.",
            ),
            HyperparameterSpec(
                "solver", str, default="adam", choices=("lbfgs", "sgd", "adam"),
                description="Optimizer used to train the network weights.",
            ),
            HyperparameterSpec(
                "alpha", float, default=0.0001, minimum=0.0, maximum=1.0,
                description="L2 regularization strength; larger values penalize large weights more.",
            ),
            HyperparameterSpec(
                "learning_rate", str, default="constant",
                choices=("constant", "invscaling", "adaptive"),
                description="Learning-rate schedule for the SGD solver.",
            ),
            HyperparameterSpec(
                "learning_rate_init", float, default=0.001, minimum=0.00001, maximum=1.0,
                description="Initial step size used by the optimizer.",
            ),
            HyperparameterSpec(
                "max_iter", int, default=500, minimum=50, maximum=5000, step=50,
                description="Maximum number of training iterations (epochs for stochastic solvers).",
            ),
            HyperparameterSpec(
                "batch_size", (int, str), default="auto",
                choices=("auto", 32, 64, 128, 256),
                description="Samples per gradient update for stochastic solvers.",
            ),
            HyperparameterSpec(
                "early_stopping", bool, default=False,
                description="Reserve part of training data to stop when validation score stops improving.",
            ),
            HyperparameterSpec(
                "random_state", int, default=42, minimum=0, maximum=2**32 - 1,
                description="Execution seed propagated by the generic training runner.",
            ),
        ),
        capabilities=frozenset(
            {ModelCapability.PREDICT, ModelCapability.PREDICT_PROBA}
        ),
        requires_scaling=True,
        description=(
            "A feed-forward multilayer perceptron classifier. Inputs are standardized inside the "
            "training pipeline. Hidden layer sizes, regularization, and optimizer settings are "
            "exposed for experimentation."
        ),
    )


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
    "decision_tree_specification",
    "logistic_regression_specification",
    "mlp_classifier_specification",
    "random_forest_specification",
]
