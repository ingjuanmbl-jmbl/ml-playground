"""Model-aware scikit-learn preprocessing and pipeline construction."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeAlias

from sklearn.base import TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml_playground.models.specifications import ModelSpecification

PREPROCESSING_STEP = "preprocessing"
ESTIMATOR_STEP = "estimator"
PreprocessingStep: TypeAlias = TransformerMixin | Literal["passthrough"]


def build_preprocessor(specification: ModelSpecification) -> PreprocessingStep:
    """Create the preprocessing stage declared by the model specification.

    The current options are a fitted ``StandardScaler`` for scale-sensitive models and sklearn's
    ``passthrough`` stage for models that do not require scaling.
    """
    if specification.requires_scaling:
        return StandardScaler()
    return "passthrough"


def build_pipeline(
    specification: ModelSpecification,
    estimator_parameters: Mapping[str, object] | None = None,
) -> Pipeline:
    """Build one preprocessing-plus-estimator pipeline from a model specification.

    The returned pipeline must be split first and then fitted only with training observations.
    Scikit-learn fits the scaler as part of ``Pipeline.fit`` and reuses those training statistics
    when transforming later data passed to ``predict``.
    """
    estimator = specification.build_estimator(estimator_parameters)
    return Pipeline(
        steps=[
            (PREPROCESSING_STEP, build_preprocessor(specification)),
            (ESTIMATOR_STEP, estimator),
        ]
    )


def get_pipeline_stages(pipeline: Pipeline) -> tuple[tuple[str, object], ...]:
    """Return the pipeline's ordered, named stages."""
    return tuple(pipeline.steps)
