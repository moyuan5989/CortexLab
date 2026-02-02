"""LMForge — LoRA SFT training framework for MLX on Apple Silicon."""

import json
from pathlib import Path

from transformers import AutoTokenizer

from lmforge._version import __version__
from lmforge.data.cache import check_cache, compute_fingerprint, read_cache, write_cache
from lmforge.data.formats import detect_format, validate_samples
from lmforge.data.preprocessing import tokenize_dataset


def prepare(
    data_path: str,
    model: str,
    output: str | None = None,
    *,
    trust_remote_code: bool = False,
    max_seq_length: int = 2048,
    mask_prompt: bool = True,
) -> dict:
    """Pre-tokenize a dataset and write a safetensors cache to disk.

    Args:
        data_path: Path to JSONL data file
        model: HuggingFace model ID or local path (for tokenizer)
        output: Output cache directory (default: ~/.lmforge/cache/preprocessed)
        trust_remote_code: Trust remote code when loading tokenizer
        max_seq_length: Maximum sequence length
        mask_prompt: Mask prompt tokens from loss

    Returns:
        Dict of statistics (sample count, total tokens, etc.)
    """
    # Default output directory
    if output is None:
        output = "~/.lmforge/cache/preprocessed"

    # Load tokenizer
    print(f"Loading tokenizer from {model}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=trust_remote_code,
    )

    # Read JSONL
    print(f"Reading {data_path}...")
    data_path_obj = Path(data_path)
    if not data_path_obj.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    with open(data_path_obj) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    if not samples:
        raise ValueError(f"No samples found in {data_path}")

    # Detect format
    fmt = detect_format(samples)
    print(f"Detected format: {fmt}")

    # Validate samples
    print(f"Validating {len(samples)} samples...")
    errors = validate_samples(samples, fmt)
    if errors:
        error_msg = "\n".join(errors[:10])  # Show first 10 errors
        if len(errors) > 10:
            error_msg += f"\n... and {len(errors) - 10} more errors"
        raise ValueError(f"Validation failed:\n{error_msg}")

    # Compute fingerprint
    fingerprint = compute_fingerprint(data_path, tokenizer)
    print(f"Data fingerprint: {fingerprint}")

    # Check cache
    if check_cache(output, fingerprint):
        print(f"Cache hit! Loading from {output}/{fingerprint}")
        meta_path = Path(output).expanduser() / fingerprint / "meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"✓ Loaded {meta['num_samples']} samples, {meta['total_tokens']} tokens")
        return meta

    # Cache miss - tokenize
    print(f"Cache miss. Tokenizing {len(samples)} samples...")
    tokenized = tokenize_dataset(
        samples,
        tokenizer,
        fmt,
        mask_prompt=mask_prompt,
        max_seq_length=max_seq_length,
    )

    # Write cache
    print(f"Writing cache to {output}/{fingerprint}...")
    meta = write_cache(output, fingerprint, tokenized, fmt)

    print(f"✓ Preprocessed {meta['num_samples']} samples")
    print(f"  Total tokens: {meta['total_tokens']}")
    print(f"  Min/mean/max length: {meta['min_length']}/{meta['mean_length']:.1f}/{meta['max_length']}")
    print(f"  Shards: {meta['num_shards']}")

    return meta


def train(config) -> "lmforge.trainer.state.TrainState":
    """Run LoRA SFT training from a config file or TrainingConfig object.

    Args:
        config: Path to a YAML config file (str) or a TrainingConfig instance.

    Returns:
        Final TrainState after training completes.
    """
    raise NotImplementedError("train() will be implemented in M5.")
