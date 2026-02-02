"""Callback system for LMForge v0.

Callbacks execute OUTSIDE the compiled region, after mx.eval() safe points.
See V0_DESIGN_FREEZE.md §5 for callback boundary semantics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from lmforge.trainer.state import TrainState


class Callback:
    """Base class for training callbacks."""

    def on_train_begin(self, state: TrainState) -> None:
        pass

    def on_train_end(self, state: TrainState) -> None:
        pass

    def on_step_end(self, state: TrainState, metrics: dict) -> None:
        pass

    def on_eval_end(self, state: TrainState, metrics: dict) -> None:
        pass

    def on_save(self, state: TrainState, checkpoint_dir: Path) -> None:
        pass


class CallbackList:
    """Container that dispatches callback events to a list of callbacks."""

    def __init__(self, callbacks: Optional[list[Callback]] = None):
        self.callbacks = callbacks or []

    def on_train_begin(self, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_train_begin(state)

    def on_train_end(self, state: TrainState) -> None:
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_step_end(self, state: TrainState, metrics: dict) -> None:
        for cb in self.callbacks:
            cb.on_step_end(state, metrics)

    def on_eval_end(self, state: TrainState, metrics: dict) -> None:
        for cb in self.callbacks:
            cb.on_eval_end(state, metrics)

    def on_save(self, state: TrainState, checkpoint_dir: Path) -> None:
        for cb in self.callbacks:
            cb.on_save(state, checkpoint_dir)


class MetricsLoggerCallback(Callback):
    """Writes JSONL metrics to logs/metrics.jsonl."""

    def __init__(self, log_path: Path):
        raise NotImplementedError(
            "MetricsLoggerCallback will be implemented in M4."
        )


class ConsoleCallback(Callback):
    """Prints human-readable training progress to stdout."""

    def __init__(self, num_iters: int):
        raise NotImplementedError("ConsoleCallback will be implemented in M4.")


class WandBCallback(Callback):
    """Optional Weights & Biases integration (try/except import)."""

    def __init__(self, project: str, config: dict):
        raise NotImplementedError("WandBCallback will be implemented in M4.")
