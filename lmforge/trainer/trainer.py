"""Trainer class for LMForge v0."""

from __future__ import annotations

from lmforge.trainer.state import TrainState


class Trainer:
    """Runs LoRA SFT training with compiled step, callbacks, and checkpointing."""

    def __init__(self, model, config, train_dataset, val_dataset, callbacks=None):
        raise NotImplementedError("Trainer.__init__() will be implemented in M5.")

    def fit(self) -> TrainState:
        """Run the full training loop. Returns final TrainState."""
        raise NotImplementedError("Trainer.fit() will be implemented in M5.")

    def evaluate(self) -> float:
        """Run evaluation on the validation set. Returns val_loss."""
        raise NotImplementedError("Trainer.evaluate() will be implemented in M5.")
