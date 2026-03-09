"""Training state dataclass for CortexLab v0."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainState:
    """Mutable training state tracked across the training loop.

    This is the in-memory representation. Checkpoint persistence is handled
    by CheckpointManager, which serializes a subset of these fields to state.json.
    """

    step: int = 0
    epoch: int = 0
    trained_tokens: int = 0
    best_val_loss: float = float("inf")
    rng_seed: int = 42
