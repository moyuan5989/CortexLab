"""Dataset format detection and validation for LMForge v0."""

from __future__ import annotations

from typing import Literal


def detect_format(samples: list[dict]) -> Literal["chat", "completions", "text"]:
    """Auto-detect dataset format from the first sample's keys.

    - Has "messages" -> chat format
    - Has "prompt" and "completion" -> completions format
    - Has "text" -> text format
    - Otherwise -> raise error listing found keys
    """
    raise NotImplementedError("detect_format() will be implemented in M2.")


def validate_samples(samples: list[dict], fmt: str) -> list[str]:
    """Validate all samples match the detected format schema.

    Returns a list of error messages (empty if all valid).
    Iterates all samples and collects all errors before reporting.
    """
    raise NotImplementedError("validate_samples() will be implemented in M2.")
