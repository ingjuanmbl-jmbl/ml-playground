"""Plotly visualizations for clusters and centroids in original feature space."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def clustering_scatter_figure(
    X: pd.DataFrame,
    cluster_labels: Sequence[Any],
    x_feature: str,
    y_feature: str,
    *,
    centroids: object | None = None,
) -> go.Figure:
    """Show observations colored by cluster and optional centroids in original feature units."""
    if x_feature == y_feature:
        raise ValueError("Clustering scatter axes must use different features.")
    if x_feature not in X.columns or y_feature not in X.columns:
        raise ValueError("Selected clustering features must exist in X.")
    labels = np.asarray(cluster_labels)
    if labels.ndim != 1 or labels.size != len(X):
        raise ValueError("cluster_labels must have one value per observation.")
    frame = X.copy()
    frame["cluster"] = labels.astype(str)
    figure = px.scatter(
        frame,
        x=x_feature,
        y=y_feature,
        color="cluster",
        title="Grupos encontrados por K-Means",
        labels={"cluster": "Grupo"},
    )
    if centroids is not None:
        center_values = np.asarray(centroids, dtype=float)
        if center_values.ndim != 2 or center_values.shape[1] != X.shape[1]:
            raise ValueError("Centroids must have one coordinate for each original feature.")
        x_index = X.columns.get_loc(x_feature)
        y_index = X.columns.get_loc(y_feature)
        figure.add_trace(
            go.Scatter(
                x=center_values[:, x_index],
                y=center_values[:, y_index],
                mode="markers",
                name="Centroides",
                marker={"symbol": "x", "size": 16, "color": "black", "line": {"width": 3}},
            )
        )
    figure.update_layout(xaxis_title=x_feature, yaxis_title=y_feature)
    return figure


__all__ = ["clustering_scatter_figure"]
