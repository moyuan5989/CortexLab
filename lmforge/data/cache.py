"""Safetensors cache write/read and fingerprinting for LMForge v0."""

from __future__ import annotations

from pathlib import Path


def compute_fingerprint(data_path: str, tokenizer) -> str:
    """Compute cache fingerprint from data file, tokenizer vocab, and chat template.

    Returns a hex string: sha256(data_hash + tokenizer_hash + template_hash).
    """
    raise NotImplementedError("compute_fingerprint() will be implemented in M2.")


def check_cache(cache_dir: str, fingerprint: str) -> bool:
    """Check if a valid cache exists for the given fingerprint."""
    raise NotImplementedError("check_cache() will be implemented in M2.")


def write_cache(
    cache_dir: str,
    fingerprint: str,
    tokenized_samples: list[dict],
    fmt: str,
) -> dict:
    """Write tokenized data to safetensors shards + meta.json.

    Returns the meta dict (statistics).
    """
    raise NotImplementedError("write_cache() will be implemented in M2.")


def read_cache(cache_dir: str, fingerprint: str) -> list[dict]:
    """Read tokenized data from safetensors cache.

    Returns a list of dicts with keys: "tokens" (array), "offset" (int).
    """
    raise NotImplementedError("read_cache() will be implemented in M2.")
