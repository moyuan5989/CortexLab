"""JSONL metrics writer and console reporter for LMForge v0."""

from __future__ import annotations

from pathlib import Path


def write_metrics_line(path: Path, metrics: dict) -> None:
    """Append a single JSON line to the metrics file."""
    raise NotImplementedError("write_metrics_line() will be implemented in M4.")


def format_console_line(metrics: dict, num_iters: int) -> str:
    """Format a metrics dict as a human-readable console line."""
    raise NotImplementedError("format_console_line() will be implemented in M4.")
