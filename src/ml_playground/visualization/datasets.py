"""Plotly figures for direct feature-space inspection of datasets."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

from ml_playground.data.contracts import Dataset
from ml_playground.models.specifications import ProblemType
from ml_playground.ui.education import dataset_name


def dataset_scatter(dataset: Dataset, x_feature: str, y_feature: str) -> go.Figure:
    """Return a 2D scatter of two actual features, colored by target when available."""
    if x_feature not in dataset.feature_names or y_feature not in dataset.feature_names:
        raise ValueError("Selected columns must be feature names from the dataset.")
    if x_feature == y_feature:
        raise ValueError("Select two different features to create a scatter plot.")

    frame = dataset.X.loc[:, [x_feature, y_feature]].copy()
    if dataset.y is not None:
        if dataset.problem_type in {ProblemType.CLASSIFICATION, ProblemType.CLUSTERING}:
            labels = {index: name for index, name in enumerate(dataset.target_names or ())}
            frame["objetivo"] = dataset.y.map(lambda value: labels.get(value, str(value))).to_numpy()
        else:
            frame["objetivo"] = dataset.y.to_numpy()
        return px.scatter(
            frame,
            x=x_feature,
            y=y_feature,
            color="objetivo",
            title=f"{dataset_name(dataset.dataset_name, dataset.dataset_name)}: {x_feature} frente a {y_feature}",
            labels={"objetivo": "Variable objetivo"},
        )
    return px.scatter(
        frame,
        x=x_feature,
        y=y_feature,
        title=f"{dataset_name(dataset.dataset_name, dataset.dataset_name)}: {x_feature} frente a {y_feature}",
    )
