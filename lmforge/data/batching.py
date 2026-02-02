"""Sort-by-length, fixed-batch, pad-to-32 iterator for LMForge v0."""

from __future__ import annotations


def iterate_batches(dataset, config):
    """Yield (batch_tokens, lengths) tuples per V0_DESIGN_FREEZE.md §2.2.

    batch_tokens: mx.array, dtype=int32, shape=(B, T)
    lengths:      mx.array, dtype=int32, shape=(B, 2)

    - B is config.training.batch_size
    - T is padded to nearest multiple of 32, capped at config.data.max_seq_length
    - lengths[:, 0] is prompt offset, lengths[:, 1] is total unpadded length
    - Padding value is 0
    """
    raise NotImplementedError("iterate_batches() will be implemented in M2.")
