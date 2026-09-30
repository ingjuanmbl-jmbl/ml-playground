"""Shared deterministic train/test split construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ml_playground.training.contracts import TrainingSplit


def create_training_split(
    *,
    n_samples: int,
    test_size: float,
    random_state: int,
    target: pd.Series | None,
    stratified: bool,
) -> TrainingSplit:
    """Split positional observation indices, optionally preserving target proportions."""
    if stratified and target is None:
        raise ValueError("Stratification requires a target series.")
    indices = np.arange(n_samples)
    train, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=target.to_numpy() if stratified and target is not None else None,
    )
    return TrainingSplit(
        train_indices=tuple(int(index) for index in train),
        test_indices=tuple(int(index) for index in test),
        n_samples=n_samples,
        test_size=float(test_size),
        random_state=random_state,
        stratified=stratified,
    )


__all__ = ["create_training_split"]
