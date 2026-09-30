"""Plotly figures for direct feature-space inspection of datasets."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go

from ml_playground.data.contracts import Dataset
from ml_playground.models.specifications import ProblemType


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
            frame["target"] = dataset.y.map(lambda value: labels.get(value, str(value))).to_numpy()
        else:
            frame["target"] = dataset.y.to_numpy()
        return px.scatter(
            frame,
            x=x_feature,
            y=y_feature,
            color="target",
            title=f"{dataset.dataset_name}: {x_feature} vs {y_feature}",
        )
    return px.scatter(
        frame,
        x=x_feature,
        y=y_feature,
        title=f"{dataset.dataset_name}: {x_feature} vs {y_feature}",
    )
