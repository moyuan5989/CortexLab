"""Checkpoint management for LMForge v0.

Each checkpoint contains exactly three files per V0_DESIGN_FREEZE.md §2.3:
- adapters.safetensors
- optimizer.safetensors
- state.json
"""

from __future__ import annotations

from pathlib import Path

from lmforge.trainer.state import TrainState


class CheckpointManager:
    """Handles atomic checkpoint save/load and retention policy."""

    def __init__(self, config):
        raise NotImplementedError(
            "CheckpointManager.__init__() will be implemented in M4."
        )

    def save(self, state: TrainState, model, optimizer) -> Path:
        """Save a checkpoint atomically (tmp dir -> rename).

        Returns the checkpoint directory path.
        """
        raise NotImplementedError(
            "CheckpointManager.save() will be implemented in M4."
        )

    def load(self, ckpt_dir: Path, model, optimizer) -> TrainState:
        """Load a checkpoint and restore model/optimizer state.

        Returns the restored TrainState.
        """
        raise NotImplementedError(
            "CheckpointManager.load() will be implemented in M4."
        )
