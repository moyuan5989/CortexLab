"""Glob-based adapter targeting for LMForge v0.

Module path matching uses fnmatch.fnmatch() on dot-separated paths.
See V0_DESIGN_FREEZE.md §4 for full semantics.
"""

from __future__ import annotations

from typing import Optional

PRESETS: dict[str, list[str]] = {
    "attention-qv": ["*.self_attn.q_proj", "*.self_attn.v_proj"],
    "attention-all": [
        "*.self_attn.q_proj",
        "*.self_attn.k_proj",
        "*.self_attn.v_proj",
        "*.self_attn.o_proj",
    ],
    "mlp": ["*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"],
    "all-linear": [
        "*.self_attn.q_proj",
        "*.self_attn.k_proj",
        "*.self_attn.v_proj",
        "*.self_attn.o_proj",
        "*.mlp.gate_proj",
        "*.mlp.up_proj",
        "*.mlp.down_proj",
    ],
}


def get_patterns(config) -> list[str]:
    """Resolve adapter config to a list of glob patterns.

    Uses config.targets if provided, otherwise resolves config.preset via PRESETS.
    """
    raise NotImplementedError("get_patterns() will be implemented in M3.")


def named_modules(module, prefix: str = ""):
    """Yield (name, module) pairs for all submodules recursively."""
    raise NotImplementedError("named_modules() will be implemented in M3.")


def resolve_targets(
    model, patterns: list[str], num_layers: Optional[int] = None
) -> list[tuple[str, object]]:
    """Match glob patterns against model module paths.

    Returns list of (path, module) tuples for matched modules.
    Raises ValueError if no modules match, listing the attempted patterns
    and first 20 available module paths.
    """
    raise NotImplementedError("resolve_targets() will be implemented in M3.")
