"""Plotly figures for model comparison without ranking or winner selection."""

from __future__ import annotations

import plotly.graph_objects as go

from ml_playground.experiments.model_comparison import ModelComparison


def comparison_metric_figure(comparison: ModelComparison, metric: str) -> go.Figure:
    """Plot one metric in selected model order; unavailable values remain gaps."""
    if metric not in {"accuracy", "f1", "roc_auc", "training_seconds"}:
        raise ValueError(f"Unsupported comparison metric: {metric}.")
    values = [
        result.training_seconds if metric == "training_seconds" else result.metrics.get(metric)
        for result in comparison.models
    ]
    figure = go.Figure(
        data=[go.Bar(x=[item.model_name for item in comparison.models], y=values)]
    )
    title = "Training time (seconds)" if metric == "training_seconds" else metric.replace("_", " ").title()
    figure.update_layout(
        title=f"{title} by model — {comparison.dataset_name}",
        xaxis_title="Model",
        yaxis_title=title,
        showlegend=False,
    )
    return figure


__all__ = ["comparison_metric_figure"]
