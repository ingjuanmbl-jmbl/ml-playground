"""Minimal Streamlit controls and results for generic model training."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import plotly.express as px
import streamlit as st
from sklearn.exceptions import ConvergenceWarning

from ml_playground.data.contracts import Dataset
from ml_playground.data.registry import DatasetRegistry
from ml_playground.evaluation.classification import evaluate_classification
from ml_playground.models.classification import DEFAULT_MODEL_REGISTRY
from ml_playground.models.registry import ModelRegistry
from ml_playground.evaluation.results import ClassificationResult
from ml_playground.training.contracts import TrainingOutput, TrainingRequest
from ml_playground.training.runner import GenericTrainingRunner
from ml_playground.ui.widgets import parameter_widget
from ml_playground.visualization.classification import (
    coefficient_figure,
    decision_boundary_figure,
    feature_importance_figure,
)
from ml_playground.visualization.datasets import dataset_scatter


def _class_names(dataset: Dataset, labels: tuple[Any, ...]) -> list[str]:
    names = dataset.target_names or ()
    return [names[int(label)] if isinstance(label, (int, np.integer)) and 0 <= label < len(names) else str(label) for label in labels]


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


def _render_training_results(
    dataset: Dataset, output: TrainingOutput, result: ClassificationResult
) -> None:
    st.subheader("Evaluation")
    metric_columns = st.columns(5)
    metric_names = ("accuracy", "precision", "recall", "f1", "roc_auc")
    for column, metric in zip(metric_columns, metric_names, strict=True):
        value = result.metrics[metric]
        if value is None:
            column.metric(metric.replace("_", " ").title(), "Unavailable")
        else:
            column.metric(metric.replace("_", " ").title(), f"{value:.4f}")
            if metric == "roc_auc":
                column.caption("One-vs-rest macro" if len(result.class_labels) > 2 else "Binary")
        if value is None and metric == "roc_auc":
            column.caption(result.metric_availability[metric].reason)

    st.metric("Training time (seconds)", f"{result.training_seconds:.4f}")
    st.write("Resolved parameters")
    st.json(dict(output.configuration.model_parameters))

    names = _class_names(dataset, result.class_labels)
    st.subheader("Confusion matrix")
    st.plotly_chart(
        px.imshow(
            result.confusion_matrix,
            x=names,
            y=names,
            text_auto=True,
            labels={"x": "Predicted class", "y": "Actual class", "color": "Count"},
            title="Held-out observations",
        ),
        width="stretch",
        key="evaluation_confusion_matrix",
    )

    if output.coefficients is not None:
        st.subheader("Logistic Regression coefficients")
        st.caption(
            "Signed coefficients describe direction and magnitude in standardized feature space; "
            "they are not feature-importance scores."
        )
        st.plotly_chart(
            coefficient_figure(output.coefficients, dataset.feature_names, result.class_labels, class_names=names),
            width="stretch",
            key="logistic_coefficients",
        )

    if output.feature_importances is not None:
        st.subheader("Decision Tree feature importance")
        st.caption(
            "These are the tree's native feature_importances_ values, not model coefficients."
        )
        st.plotly_chart(
            feature_importance_figure(output.feature_importances, dataset.feature_names),
            width="stretch",
            key="decision_tree_feature_importances",
        )

    st.subheader("Decision boundary")
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
        st.info("A direct decision boundary is available only when the dataset has exactly two features.")
        if output.X_test is not None and output.y_test is not None and dataset.n_features > 2:
            feature_names = list(dataset.feature_names)
            x_col, y_col = st.columns(2)
            x_feature = x_col.selectbox("Preview feature (x)", feature_names, key="result_x_feature")
            y_feature = y_col.selectbox(
                "Preview feature (y)", [feature for feature in feature_names if feature != x_feature],
                key="result_y_feature",
            )
            st.plotly_chart(
                dataset_scatter(dataset, x_feature, y_feature),
                width="stretch",
                key="held_out_feature_preview",
            )


def render_training_panel(
    dataset: Dataset,
    dataset_id: str,
    dataset_parameters: dict[str, object],
    dataset_registry: DatasetRegistry,
    model_registry: ModelRegistry = DEFAULT_MODEL_REGISTRY,
) -> None:
    """Render model selection and submit one declarative TrainingRequest."""
    st.header("Train and evaluate a classifier")
    model_specs = model_registry.list()
    if not model_specs:
        st.info("No classification models are registered yet.")
        return
    spec_by_id = {spec.id: spec for spec in model_specs}
    model_id = st.selectbox(
        "Model", list(spec_by_id), format_func=lambda identifier: spec_by_id[identifier].display_name
    )
    specification = spec_by_id[model_id]
    st.caption(specification.description)

    with st.form("training_request_form"):
        st.subheader("Model hyperparameters")
        model_parameters = {
            parameter.name: parameter_widget(parameter, key_prefix=f"{model_id}_{dataset_id}")
            for parameter in specification.hyperparameters
            if parameter.name != "random_state"
        }
        test_size = st.slider("Test set size", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
        random_state = st.number_input(
            "Random state", min_value=0, max_value=2**32 - 1, value=42, step=1
        )
        submitted = st.form_submit_button("Train and evaluate", type="primary")

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
            warnings.simplefilter("always", ConvergenceWarning)
            output = runner.run(request)
        for warning in captured:
            if issubclass(warning.category, ConvergenceWarning):
                st.warning(f"Convergence warning: {warning.message}")
        result = evaluate_classification(output)
    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        st.error(f"Could not train the selected model: {error}")
        return

    _render_training_results(dataset, output, result)
