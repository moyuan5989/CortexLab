"""Tokenization and template application for LMForge v0."""

from __future__ import annotations


def tokenize_dataset(
    samples: list[dict],
    tokenizer,
    fmt: str,
    *,
    mask_prompt: bool = True,
    max_seq_length: int = 2048,
) -> list[dict]:
    """Tokenize samples and compute prompt offsets.

    Returns a list of dicts with keys: "tokens" (list[int]), "offset" (int).
    """
    raise NotImplementedError("tokenize_dataset() will be implemented in M2.")
