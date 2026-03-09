"""JSONL metrics writer and console reporter for CortexLab v0."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


def write_metrics_line(path: Path, metrics: dict) -> None:
    """Append a single JSON line to the metrics file."""
    # Add timestamp if not present
    if "timestamp" not in metrics:
        metrics["timestamp"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    # Write line
    with open(path, "a") as f:
        f.write(json.dumps(metrics) + "\n")


def format_console_line(metrics: dict, num_iters: int) -> str:
    """Format a metrics dict as a human-readable console line.

    Examples:
        Step 100/1000 | loss=2.891 | lr=1.00e-05 | tok/s=14521 | mem=11.2GB
        Step 200/1000 | val_loss=1.987
    """
    step = metrics.get("step", 0)
    event = metrics.get("event", "train")

    if event == "train":
        train_loss = metrics.get("train_loss", 0.0)
        lr = metrics.get("learning_rate", 0.0)
        tok_s = metrics.get("tokens_per_second", 0.0)
        mem_gb = metrics.get("peak_memory_gb", 0.0)

        return (
            f"Step {step}/{num_iters} | "
            f"loss={train_loss:.3f} | "
            f"lr={lr:.2e} | "
            f"tok/s={tok_s:.0f} | "
            f"mem={mem_gb:.1f}GB"
        )

    elif event == "eval":
        val_loss = metrics.get("val_loss", 0.0)
        return f"Step {step}/{num_iters} | val_loss={val_loss:.3f}"

    else:
        # Generic format for unknown events
        items = [f"{k}={v}" for k, v in metrics.items() if k not in ["event", "timestamp"]]
        return f"Step {step}/{num_iters} | " + " | ".join(items)
