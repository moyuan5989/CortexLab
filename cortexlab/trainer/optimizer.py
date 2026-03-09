"""Optimizer and LR scheduler factory for CortexLab v0.

Schedulers are stateless functions of step number.
On resume, the scheduler is reconstructed from config and given the saved step.
"""

from __future__ import annotations

import mlx.optimizers as optim


def build_optimizer(training_params, model):
    """Build an optimizer from TrainingParams config.

    Returns an MLX optimizer instance with the configured LR schedule.
    """
    # Build LR schedule (stateless function of step)
    lr_schedule = build_scheduler(training_params)

    # Get optimizer class
    optimizer_name = training_params.optimizer.lower()
    optimizer_kwargs = training_params.optimizer_config.copy()

    if optimizer_name == "adam":
        optimizer_cls = optim.Adam
    elif optimizer_name == "adamw":
        optimizer_cls = optim.AdamW
    elif optimizer_name == "sgd":
        optimizer_cls = optim.SGD
    elif optimizer_name == "adafactor":
        optimizer_cls = optim.Adafactor
    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}. "
            f"Supported: adam, adamw, sgd, adafactor"
        )

    # Create optimizer with LR schedule
    optimizer = optimizer_cls(
        learning_rate=lr_schedule,
        **optimizer_kwargs,
    )

    return optimizer


def build_scheduler(training_params):
    """Build a stateless LR schedule from TrainingParams config.

    Returns a callable or MLX schedule object.

    Schedulers are pure functions of step number. On resume, the scheduler
    is reconstructed from config and given the saved step. No scheduler
    internal state is checkpointed.
    """
    base_lr = training_params.learning_rate
    lr_schedule_config = training_params.lr_schedule

    # No schedule - constant LR
    if lr_schedule_config is None:
        return base_lr

    # Build warmup schedule if specified
    warmup_steps = lr_schedule_config.warmup
    warmup_init = lr_schedule_config.warmup_init

    # Get the main schedule
    schedule_name = lr_schedule_config.name
    schedule_args = lr_schedule_config.arguments

    # Build the main schedule
    if schedule_name == "cosine_decay":
        # cosine_decay(init, decay_steps, end=0.0)
        if len(schedule_args) < 2:
            raise ValueError(
                f"cosine_decay requires at least 2 arguments (init, decay_steps), "
                f"got {len(schedule_args)}"
            )
        main_schedule = optim.cosine_decay(*schedule_args)

    elif schedule_name == "linear_schedule":
        # linear_schedule(init, end, steps)
        if len(schedule_args) < 3:
            raise ValueError(
                f"linear_schedule requires 3 arguments (init, end, steps), "
                f"got {len(schedule_args)}"
            )
        main_schedule = optim.linear_schedule(*schedule_args)

    elif schedule_name == "step_decay":
        # step_decay(init, decay_rate, decay_steps)
        if len(schedule_args) < 3:
            raise ValueError(
                f"step_decay requires 3 arguments (init, decay_rate, decay_steps), "
                f"got {len(schedule_args)}"
            )
        main_schedule = optim.step_decay(*schedule_args)

    elif schedule_name == "exponential_decay":
        # exponential_decay(init, decay_rate)
        if len(schedule_args) < 2:
            raise ValueError(
                f"exponential_decay requires 2 arguments (init, decay_rate), "
                f"got {len(schedule_args)}"
            )
        main_schedule = optim.exponential_decay(*schedule_args)

    else:
        raise ValueError(
            f"Unknown LR schedule: {schedule_name}. "
            f"Supported: cosine_decay, linear_schedule, step_decay, exponential_decay"
        )

    # Apply warmup if specified
    if warmup_steps > 0:
        # Build warmup schedule: linear from warmup_init to base_lr
        warmup_schedule = optim.linear_schedule(warmup_init, base_lr, warmup_steps)
        # Join warmup and main schedule
        schedule = optim.join_schedules([warmup_schedule, main_schedule], [warmup_steps])
    else:
        schedule = main_schedule

    return schedule
