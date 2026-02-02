"""Safetensors cache write/read and fingerprinting for LMForge v0."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np


def compute_fingerprint(data_path: str, tokenizer) -> str:
    """Compute cache fingerprint from data file, tokenizer vocab, and chat template.

    Returns a hex string: sha256(data_hash + tokenizer_hash + template_hash).
    """
    # Hash the data file bytes
    data_hash = hashlib.sha256(Path(data_path).read_bytes()).hexdigest()

    # Hash the tokenizer vocabulary
    vocab_items = sorted(tokenizer.get_vocab().items())
    vocab_str = json.dumps(vocab_items)
    tokenizer_hash = hashlib.sha256(vocab_str.encode()).hexdigest()

    # Hash the chat template (if present)
    template = getattr(tokenizer, "chat_template", None) or ""
    template_hash = hashlib.sha256(template.encode()).hexdigest()

    # Combine all hashes
    combined = data_hash + tokenizer_hash + template_hash
    fingerprint = hashlib.sha256(combined.encode()).hexdigest()

    return f"sha256:{fingerprint}"


def check_cache(cache_dir: str, fingerprint: str) -> bool:
    """Check if a valid cache exists for the given fingerprint."""
    cache_path = Path(cache_dir).expanduser() / fingerprint
    meta_path = cache_path / "meta.json"

    if not meta_path.exists():
        return False

    # Verify meta.json is valid
    try:
        with open(meta_path) as f:
            meta = json.load(f)
        # Check required fields
        required = ["schema_version", "num_samples", "num_shards", "data_fingerprint"]
        if not all(k in meta for k in required):
            return False
        # Verify fingerprint matches
        if meta["data_fingerprint"] != fingerprint:
            return False
        # Verify shards exist
        for i in range(meta["num_shards"]):
            shard_path = cache_path / f"shard_{i:03d}.safetensors"
            if not shard_path.exists():
                return False
        return True
    except (json.JSONDecodeError, KeyError):
        return False


def write_cache(
    cache_dir: str,
    fingerprint: str,
    tokenized_samples: list[dict],
    fmt: str,
) -> dict:
    """Write tokenized data to safetensors shards + meta.json.

    Returns the meta dict (statistics).

    Shard format per V0_DESIGN_FREEZE.md §2.5:
        tokens_0: int32[L0]
        tokens_1: int32[L1]
        ...
        offsets: int32[N]  # prompt offset per sample
        lengths: int32[N]  # total token count per sample
    """
    cache_path = Path(cache_dir).expanduser() / fingerprint
    cache_path.mkdir(parents=True, exist_ok=True)

    # Target shard size: ~500MB
    # Assuming int32 (4 bytes), 500MB / 4 = 125M tokens per shard
    # With average sequence length of 512, that's ~244k samples per shard
    TARGET_TOKENS_PER_SHARD = 125_000_000

    # Compute statistics
    all_lengths = [len(sample["tokens"]) for sample in tokenized_samples]
    total_tokens = sum(all_lengths)
    num_samples = len(tokenized_samples)

    # Split into shards
    shards = []
    current_shard = []
    current_shard_tokens = 0

    for sample in tokenized_samples:
        sample_tokens = len(sample["tokens"])
        if current_shard_tokens + sample_tokens > TARGET_TOKENS_PER_SHARD and current_shard:
            shards.append(current_shard)
            current_shard = []
            current_shard_tokens = 0

        current_shard.append(sample)
        current_shard_tokens += sample_tokens

    if current_shard:
        shards.append(current_shard)

    # Write shards
    for shard_idx, shard_samples in enumerate(shards):
        shard_data = {}

        # Add token arrays for each sample
        for sample_idx, sample in enumerate(shard_samples):
            tokens = np.array(sample["tokens"], dtype=np.int32)
            shard_data[f"tokens_{sample_idx}"] = mx.array(tokens)

        # Add offsets and lengths arrays
        offsets = np.array([s["offset"] for s in shard_samples], dtype=np.int32)
        lengths = np.array([len(s["tokens"]) for s in shard_samples], dtype=np.int32)
        shard_data["offsets"] = mx.array(offsets)
        shard_data["lengths"] = mx.array(lengths)

        # Write shard
        shard_path = cache_path / f"shard_{shard_idx:03d}.safetensors"
        mx.save_safetensors(str(shard_path), shard_data)

    # Write meta.json
    meta = {
        "schema_version": 1,
        "num_samples": num_samples,
        "num_shards": len(shards),
        "total_tokens": total_tokens,
        "format": fmt,
        "data_fingerprint": fingerprint,
        "max_length": max(all_lengths) if all_lengths else 0,
        "min_length": min(all_lengths) if all_lengths else 0,
        "mean_length": total_tokens / num_samples if num_samples > 0 else 0.0,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }

    meta_path = cache_path / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return meta


def read_cache(cache_dir: str, fingerprint: str) -> list[dict]:
    """Read tokenized data from safetensors cache.

    Returns a list of dicts with keys: "tokens" (array), "offset" (int).
    """
    cache_path = Path(cache_dir).expanduser() / fingerprint
    meta_path = cache_path / "meta.json"

    # Load meta
    with open(meta_path) as f:
        meta = json.load(f)

    num_shards = meta["num_shards"]
    all_samples = []

    # Load each shard
    for shard_idx in range(num_shards):
        shard_path = cache_path / f"shard_{shard_idx:03d}.safetensors"
        shard_data = mx.load(str(shard_path))

        offsets = shard_data["offsets"]
        lengths = shard_data["lengths"]
        num_samples_in_shard = len(offsets)

        # Extract each sample
        for sample_idx in range(num_samples_in_shard):
            tokens = shard_data[f"tokens_{sample_idx}"]
            offset = int(offsets[sample_idx])

            all_samples.append({
                "tokens": tokens,
                "offset": offset,
            })

    return all_samples
