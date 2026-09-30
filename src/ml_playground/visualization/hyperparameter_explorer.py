"""Plotly chart for one-parameter sensitivity results."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import plotly.graph_objects as go


def hyperparameter_metric_figure(
    *,
    model_name: str,
    hyperparameter: str,
    metric: str,
    values: Sequence[Any],
    scores: Sequence[float | None],
) -> go.Figure:
    """Plot metric values in the supplied exploration order without ranking configurations."""
    if len(values) != len(scores) or not values:
        raise ValueError("Hyperparameter values and scores must be non-empty and have equal lengths.")
    numeric_axis = all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
    x_values = list(values) if numeric_axis else ["None" if value is None else str(value) for value in values]
    figure = go.Figure(
        go.Scatter(
            x=x_values,
            y=list(scores),
            mode="lines+markers",
            name=metric,
            connectgaps=False,
        )
    )
    figure.update_layout(
        title=f"{model_name}: {metric} vs {hyperparameter}",
        xaxis_title=hyperparameter,
        yaxis_title=metric,
    )
    if not numeric_axis:
        figure.update_xaxes(type="category", categoryorder="array", categoryarray=x_values)
    return figure


__all__ = ["hyperparameter_metric_figure"]
