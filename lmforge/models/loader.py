"""Model and tokenizer loading for LMForge v0."""

from __future__ import annotations


def load_model(model_path: str, *, trust_remote_code: bool = False):
    """Load a model and tokenizer from a HuggingFace repo ID or local path.

    Returns (model, tokenizer) tuple.
    """
    raise NotImplementedError("load_model() will be implemented in M3.")
