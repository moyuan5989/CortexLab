"""Optimizer and LR scheduler factory for LMForge v0.

Schedulers are stateless functions of step number.
On resume, the scheduler is reconstructed from config and given the saved step.
"""

from __future__ import annotations


def build_optimizer(training_params, model):
    """Build an optimizer from TrainingParams config.

    Returns an MLX optimizer instance with the configured LR schedule.
    """
    raise NotImplementedError("build_optimizer() will be implemented in M4.")


def build_scheduler(training_params):
    """Build a stateless LR schedule from TrainingParams config.

    Returns a callable or MLX schedule object.
    """
    raise NotImplementedError("build_scheduler() will be implemented in M4.")
