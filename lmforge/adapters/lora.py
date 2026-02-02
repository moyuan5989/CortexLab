"""LoRA adapter modules for LMForge v0."""

from __future__ import annotations


class LoRALinear:
    """LoRA wrapper for nn.Linear and nn.QuantizedLinear."""

    @classmethod
    def from_base(cls, module, *, r: int, scale: float, dropout: float = 0.0):
        """Create a LoRALinear from an existing Linear module."""
        raise NotImplementedError("LoRALinear.from_base() will be implemented in M3.")

    def fuse(self):
        """Merge LoRA weights back into the base weight and return a plain Linear."""
        raise NotImplementedError("LoRALinear.fuse() will be implemented in M3.")


class LoRAEmbedding:
    """LoRA wrapper for nn.Embedding and nn.QuantizedEmbedding."""

    @classmethod
    def from_base(cls, module, *, r: int, scale: float, dropout: float = 0.0):
        """Create a LoRAEmbedding from an existing Embedding module."""
        raise NotImplementedError(
            "LoRAEmbedding.from_base() will be implemented in M3."
        )

    def fuse(self):
        """Merge LoRA weights back into the base weight and return a plain Embedding."""
        raise NotImplementedError("LoRAEmbedding.fuse() will be implemented in M3.")


def apply_lora(model, targets: list[tuple[str, object]], config) -> object:
    """Apply LoRA adapters to matched modules in-place.

    Returns the modified model.
    """
    raise NotImplementedError("apply_lora() will be implemented in M3.")
