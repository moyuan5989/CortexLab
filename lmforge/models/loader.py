"""Model and tokenizer loading for LMForge v0."""

from __future__ import annotations

from typing import Optional

from transformers import AutoTokenizer


def load_model(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    trust_remote_code: bool = False,
):
    """Load a model and tokenizer from a local path (post-resolution).

    Args:
        model_path: Local path to model directory (already resolved)
        tokenizer_path: Optional separate tokenizer path (already resolved)
        trust_remote_code: Whether to trust remote code in model/tokenizer

    Returns:
        (model, tokenizer) tuple

    Note:
        This function expects a local path. HF resolution should happen
        before calling this function (see lmforge.models.resolve).
    """
    try:
        from mlx_lm import load as mlx_lm_load
    except ImportError:
        raise ImportError(
            "mlx_lm is required for model loading. Install it with:\n"
            "  pip install mlx-lm\n\n"
            "Note: mlx_lm is not a hard dependency of lmforge to keep the core "
            "package lightweight. It's only needed when loading models for training."
        )

    # Use tokenizer_path if provided, otherwise use model_path
    tok_path = tokenizer_path if tokenizer_path is not None else model_path

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=trust_remote_code,
    )

    # Load model using mlx_lm
    # mlx_lm.load returns (model, tokenizer), but we already loaded the tokenizer
    # to have consistent behavior. Use mlx_lm's model loading.
    model, _ = mlx_lm_load(
        model_path,
        tokenizer_config={"trust_remote_code": trust_remote_code},
    )

    return model, tokenizer
