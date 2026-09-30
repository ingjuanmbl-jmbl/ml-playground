"""Training orchestration contracts.

The concrete runner is intentionally deferred until the first model implementation phase.
"""

from ml_playground.training.contracts import TrainingOutput, TrainingRequest, TrainingRunner

__all__ = ["TrainingOutput", "TrainingRequest", "TrainingRunner"]
