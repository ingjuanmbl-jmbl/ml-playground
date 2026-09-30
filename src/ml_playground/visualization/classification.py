"""Plotly figures for classifier outputs, coefficients, and tree importances."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline


def decision_boundary_figure(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    class_labels: Sequence[Any],
    *,
    class_names: Sequence[str] | None = None,
    resolution: int = 120,
) -> go.Figure:
    """Plot a 2D decision surface by predicting a mesh through the fitted full pipeline."""
    if X_test.shape[1] != 2:
        raise ValueError("A direct decision boundary requires exactly two predictor features.")
    if resolution < 20:
        raise ValueError("resolution must be at least 20.")
    x_name, y_name = X_test.columns
    x_values = X_test.iloc[:, 0].to_numpy(dtype=float)
    y_values = X_test.iloc[:, 1].to_numpy(dtype=float)
    x_pad = max(float(np.ptp(x_values)) * 0.1, 0.1)
    y_pad = max(float(np.ptp(y_values)) * 0.1, 0.1)
    x_grid = np.linspace(float(x_values.min()) - x_pad, float(x_values.max()) + x_pad, resolution)
    y_grid = np.linspace(float(y_values.min()) - y_pad, float(y_values.max()) + y_pad, resolution)
    mesh_x, mesh_y = np.meshgrid(x_grid, y_grid)
    grid = pd.DataFrame({x_name: mesh_x.ravel(), y_name: mesh_y.ravel()})
    mesh_predictions = pipeline.predict(grid)

    label_to_index = {label: index for index, label in enumerate(class_labels)}
    z_values = np.asarray([label_to_index[label] for label in mesh_predictions]).reshape(mesh_x.shape)
    names = list(class_names or [str(label) for label in class_labels])
    palette = px.colors.qualitative.Plotly
    figure = go.Figure()
    figure.add_trace(
        go.Contour(
            x=x_grid,
            y=y_grid,
            z=z_values,
            name="Predicted class",
            showscale=False,
            contours={"start": -0.5, "end": len(class_labels) - 0.5, "size": 1, "coloring": "fill"},
            colorscale=[[i / max(len(class_labels) - 1, 1), palette[i % len(palette)]] for i in range(len(class_labels))],
            opacity=0.28,
            hoverinfo="skip",
        )
    )
    test_frame = X_test.copy()
    test_frame["class"] = [names[label_to_index[label]] for label in y_test.to_numpy()]
    for index, label in enumerate(class_labels):
        points = test_frame.loc[test_frame["class"] == names[index]]
        figure.add_trace(
            go.Scatter(
                x=points[x_name], y=points[y_name], mode="markers", name=names[index],
                marker={"color": palette[index % len(palette)], "line": {"color": "white", "width": 0.5}},
            )
        )
    figure.update_layout(
        title="Decision boundary on held-out observations",
        xaxis_title=x_name,
        yaxis_title=y_name,
        legend_title="Target class",
    )
    return figure


def coefficient_figure(
    coefficients: object,
    feature_names: Sequence[str],
    class_labels: Sequence[Any],
    *,
    class_names: Sequence[str] | None = None,
) -> go.Figure:
    """Show signed Logistic Regression coefficients without converting them to importances."""
    values = np.asarray(coefficients, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("Coefficient shape must match the feature names.")
    labels = list(class_names or [str(label) for label in class_labels])
    if values.shape[0] == 1:
        positive_label = labels[1] if len(labels) > 1 else labels[0]
        figure = px.bar(
            x=list(feature_names), y=values[0],
            labels={"x": "Feature", "y": "Signed coefficient"},
            title=f"Signed coefficients ({positive_label} vs other class)",
        )
        figure.update_layout(xaxis_title="Feature", yaxis_title="Signed coefficient")
        return figure
    row_labels = labels if len(labels) == values.shape[0] else [f"Coefficient row {i + 1}" for i in range(values.shape[0])]
    return go.Figure(
        data=go.Heatmap(z=values, x=list(feature_names), y=row_labels, colorscale="RdBu", zmid=0)
    ).update_layout(
        title="Signed coefficients by class (scaled feature space)",
        xaxis_title="Feature",
        yaxis_title="Class",
    )


def feature_importance_figure(
    importances: object,
    feature_names: Sequence[str],
) -> go.Figure:
    """Plot a classifier's native feature_importances_ values in descending order."""
    values = np.asarray(importances, dtype=float)
    if values.ndim != 1 or values.size != len(feature_names):
        raise ValueError("Feature importance values must match the feature names.")
    order = np.argsort(values)[::-1]
    return px.bar(
        x=[feature_names[index] for index in order],
        y=values[order],
        labels={"x": "Feature", "y": "feature_importances_"},
        title="Decision Tree feature_importances_",
    )


__all__ = ["coefficient_figure", "decision_boundary_figure", "feature_importance_figure"]
