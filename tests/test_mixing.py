"""Tests for dataset mixing iterator."""

from __future__ import annotations

import pytest

from lmforge.data.mixing import MixedDatasetIterator


class TestMixedDatasetIterator:
    def test_basic_iteration(self):
        """Test that iterator produces samples from all datasets."""
        ds1 = [{"input_ids": [1, 2], "labels": [1, 2]}]
        ds2 = [{"input_ids": [3, 4], "labels": [3, 4]}]
        mix = MixedDatasetIterator([ds1, ds2], [0.5, 0.5], seed=42)

        samples = [next(mix) for _ in range(20)]
        ids_seen = {s["input_ids"][0] for s in samples}
        assert 1 in ids_seen
        assert 3 in ids_seen

    def test_weighted_sampling(self):
        """Test that weights affect sampling distribution."""
        ds1 = [{"id": "a"}]
        ds2 = [{"id": "b"}]
        mix = MixedDatasetIterator([ds1, ds2], [0.9, 0.1], seed=42)

        samples = [next(mix) for _ in range(1000)]
        count_a = sum(1 for s in samples if s["id"] == "a")
        # Should be roughly 900, allow wide margin
        assert count_a > 700
        assert count_a < 990

    def test_cycling(self):
        """Test that datasets cycle when exhausted."""
        ds1 = [{"v": 1}, {"v": 2}]
        mix = MixedDatasetIterator([ds1], [1.0], seed=42)

        # Should cycle through the 2 samples repeatedly
        values = [next(mix)["v"] for _ in range(6)]
        assert values == [1, 2, 1, 2, 1, 2]

    def test_deterministic_with_seed(self):
        """Test that same seed produces same sequence."""
        ds1 = [{"v": 1}]
        ds2 = [{"v": 2}]

        mix1 = MixedDatasetIterator([ds1, ds2], [0.5, 0.5], seed=123)
        mix2 = MixedDatasetIterator([ds1, ds2], [0.5, 0.5], seed=123)

        seq1 = [next(mix1)["v"] for _ in range(50)]
        seq2 = [next(mix2)["v"] for _ in range(50)]
        assert seq1 == seq2

    def test_empty_datasets_raises(self):
        """Test that empty dataset list raises."""
        with pytest.raises(ValueError, match="At least one"):
            MixedDatasetIterator([], [], seed=42)

    def test_mismatched_lengths_raises(self):
        """Test that mismatched datasets/weights raises."""
        with pytest.raises(ValueError, match="must match"):
            MixedDatasetIterator([["a"]], [0.5, 0.5], seed=42)

    def test_single_dataset(self):
        """Test single dataset with weight 1.0."""
        ds = [{"v": i} for i in range(5)]
        mix = MixedDatasetIterator([ds], [1.0], seed=42)
        values = [next(mix)["v"] for _ in range(5)]
        assert values == [0, 1, 2, 3, 4]

    def test_three_datasets(self):
        """Test mixing three datasets."""
        ds1 = [{"src": "a"}]
        ds2 = [{"src": "b"}]
        ds3 = [{"src": "c"}]
        mix = MixedDatasetIterator([ds1, ds2, ds3], [1.0, 1.0, 1.0], seed=42)

        samples = [next(mix) for _ in range(300)]
        sources = {s["src"] for s in samples}
        assert sources == {"a", "b", "c"}
