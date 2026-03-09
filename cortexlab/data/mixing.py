"""Weighted dataset mixing iterator for multi-source training.

Samples from multiple datasets according to configured weights,
producing an infinite stream of training examples.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np


class MixedDatasetIterator:
    """Weighted random sampling across multiple datasets.

    Each dataset is cycled through independently. On each call to __next__,
    a dataset is chosen according to the normalized weight distribution,
    and the next sample from that dataset is returned.
    """

    def __init__(
        self,
        datasets: list,
        weights: list[float],
        seed: int = 42,
    ):
        if len(datasets) != len(weights):
            raise ValueError(
                f"Number of datasets ({len(datasets)}) must match "
                f"number of weights ({len(weights)})"
            )
        if not datasets:
            raise ValueError("At least one dataset is required")

        self._datasets = datasets
        self._rng = np.random.RandomState(seed)

        # Normalize weights to probabilities
        total = sum(weights)
        self._probs = [w / total for w in weights]

        # Create cycling iterators for each dataset
        self._iters: list[Iterator] = [
            _cycle_dataset(ds) for ds in datasets
        ]

    def __iter__(self):
        return self

    def __next__(self):
        # Pick dataset index based on weights
        idx = self._rng.choice(len(self._datasets), p=self._probs)
        return next(self._iters[idx])


def _cycle_dataset(dataset) -> Iterator:
    """Infinitely cycle through a dataset."""
    while True:
        for sample in dataset:
            yield sample
