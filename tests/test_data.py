"""Tests for data pipeline (M2)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmforge.data.batching import iterate_batches
from lmforge.data.cache import check_cache, compute_fingerprint, read_cache, write_cache
from lmforge.data.formats import detect_format, validate_samples


class TestFormatDetection:
    def test_detect_chat_format(self):
        """Test detection of chat format."""
        samples = [{"messages": [{"role": "user", "content": "Hello"}]}]
        assert detect_format(samples) == "chat"

    def test_detect_completions_format(self):
        """Test detection of completions format."""
        samples = [{"prompt": "Hello", "completion": "Hi there!"}]
        assert detect_format(samples) == "completions"

    def test_detect_text_format(self):
        """Test detection of text format."""
        samples = [{"text": "This is a sample text."}]
        assert detect_format(samples) == "text"

    def test_unknown_format_raises(self):
        """Test that unknown format raises ValueError with helpful message."""
        samples = [{"unknown_key": "value"}]
        with pytest.raises(ValueError) as exc_info:
            detect_format(samples)
        assert "unknown" in str(exc_info.value).lower()
        assert "unknown_key" in str(exc_info.value)


class TestFormatValidation:
    def test_validate_chat_samples(self):
        """Test validation of chat format samples."""
        # Valid samples
        valid_samples = [
            {"messages": [{"role": "user", "content": "Hello"}]},
            {"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]},
        ]
        errors = validate_samples(valid_samples, "chat")
        assert len(errors) == 0

        # Invalid: missing messages
        invalid_samples = [{"text": "wrong format"}]
        errors = validate_samples(invalid_samples, "chat")
        assert len(errors) > 0
        assert "messages" in errors[0].lower()

        # Invalid: messages not a list
        invalid_samples = [{"messages": "not a list"}]
        errors = validate_samples(invalid_samples, "chat")
        assert len(errors) > 0

        # Invalid: message missing role
        invalid_samples = [{"messages": [{"content": "Hello"}]}]
        errors = validate_samples(invalid_samples, "chat")
        assert len(errors) > 0
        assert "role" in errors[0].lower()

    def test_validate_completions_samples(self):
        """Test validation of completions format samples."""
        # Valid
        valid_samples = [{"prompt": "Hello", "completion": "Hi!"}]
        errors = validate_samples(valid_samples, "completions")
        assert len(errors) == 0

        # Missing prompt
        invalid_samples = [{"completion": "Hi!"}]
        errors = validate_samples(invalid_samples, "completions")
        assert len(errors) > 0
        assert "prompt" in errors[0].lower()

        # Missing completion
        invalid_samples = [{"prompt": "Hello"}]
        errors = validate_samples(invalid_samples, "completions")
        assert len(errors) > 0
        assert "completion" in errors[0].lower()

    def test_validate_text_samples(self):
        """Test validation of text format samples."""
        # Valid
        valid_samples = [{"text": "Sample text"}]
        errors = validate_samples(valid_samples, "text")
        assert len(errors) == 0

        # Missing text
        invalid_samples = [{"prompt": "wrong"}]
        errors = validate_samples(invalid_samples, "text")
        assert len(errors) > 0
        assert "text" in errors[0].lower()


class TestCacheFingerprinting:
    def test_fingerprint_format(self, tmp_dir):
        """Test that fingerprint has correct format."""
        # Create a simple JSONL file
        data_file = tmp_dir / "test.jsonl"
        data_file.write_text('{"text": "sample"}\n')

        # Mock tokenizer
        class MockTokenizer:
            def get_vocab(self):
                return {"a": 0, "b": 1}
            chat_template = None

        tokenizer = MockTokenizer()
        fingerprint = compute_fingerprint(str(data_file), tokenizer)

        assert fingerprint.startswith("sha256:")
        assert len(fingerprint) == 71  # "sha256:" + 64 hex chars

    def test_same_inputs_same_fingerprint(self, tmp_dir):
        """Test that same inputs produce same fingerprint."""
        data_file = tmp_dir / "test.jsonl"
        data_file.write_text('{"text": "sample"}\n')

        class MockTokenizer:
            def get_vocab(self):
                return {"a": 0, "b": 1}
            chat_template = None

        tokenizer = MockTokenizer()
        fp1 = compute_fingerprint(str(data_file), tokenizer)
        fp2 = compute_fingerprint(str(data_file), tokenizer)

        assert fp1 == fp2

    def test_different_data_different_fingerprint(self, tmp_dir):
        """Test that different data produces different fingerprint."""
        file1 = tmp_dir / "test1.jsonl"
        file1.write_text('{"text": "sample1"}\n')

        file2 = tmp_dir / "test2.jsonl"
        file2.write_text('{"text": "sample2"}\n')

        class MockTokenizer:
            def get_vocab(self):
                return {"a": 0, "b": 1}
            chat_template = None

        tokenizer = MockTokenizer()
        fp1 = compute_fingerprint(str(file1), tokenizer)
        fp2 = compute_fingerprint(str(file2), tokenizer)

        assert fp1 != fp2


class TestCaching:
    def test_write_and_read_cache(self, tmp_dir):
        """Test writing and reading cache."""
        cache_dir = str(tmp_dir)
        fingerprint = "sha256:test123"
        tokenized_samples = [
            {"tokens": [1, 2, 3, 4], "offset": 0},
            {"tokens": [5, 6, 7], "offset": 2},
        ]

        # Write cache
        meta = write_cache(cache_dir, fingerprint, tokenized_samples, "text")

        assert meta["num_samples"] == 2
        assert meta["total_tokens"] == 7
        assert meta["max_length"] == 4
        assert meta["min_length"] == 3

        # Read cache
        loaded = read_cache(cache_dir, fingerprint)
        assert len(loaded) == 2
        assert loaded[0]["offset"] == 0
        assert loaded[1]["offset"] == 2

    def test_check_cache(self, tmp_dir):
        """Test cache checking."""
        cache_dir = str(tmp_dir)
        fingerprint = "sha256:test456"

        # No cache yet
        assert not check_cache(cache_dir, fingerprint)

        # Write cache
        tokenized_samples = [{"tokens": [1, 2, 3], "offset": 0}]
        write_cache(cache_dir, fingerprint, tokenized_samples, "text")

        # Cache now exists
        assert check_cache(cache_dir, fingerprint)


class TestBatching:
    def test_batch_shapes_match_contract(self, sample_config_dict):
        """Test that batch shapes match V0_DESIGN_FREEZE.md §2.2 contract."""
        from lmforge.config import TrainingConfig

        # Create mock dataset
        dataset = [
            {"tokens": [1, 2, 3, 4, 5], "offset": 0},
            {"tokens": [6, 7, 8, 9], "offset": 2},
            {"tokens": [10, 11, 12], "offset": 0},
            {"tokens": [13, 14, 15, 16], "offset": 1},
        ]

        config = TrainingConfig(**sample_config_dict)
        batches = list(iterate_batches(dataset, config))

        assert len(batches) == 1  # 4 samples, batch_size=4

        batch_tokens, lengths = batches[0]

        # Check shapes
        assert batch_tokens.shape == (config.training.batch_size, batch_tokens.shape[1])
        assert lengths.shape == (config.training.batch_size, 2)

        # Check dtypes
        assert "int32" in str(batch_tokens.dtype)
        assert "int32" in str(lengths.dtype)

        # Check lengths array structure
        for i in range(config.training.batch_size):
            prompt_offset = lengths[i, 0].item()
            total_length = lengths[i, 1].item()
            assert prompt_offset >= 0
            assert total_length > 0
            assert prompt_offset <= total_length

    def test_padding_to_multiple_of_32(self, sample_config_dict):
        """Test that sequences are padded to nearest multiple of 32."""
        from lmforge.config import TrainingConfig

        # Create dataset with length 37 (should pad to 64)
        dataset = [
            {"tokens": [1] * 37, "offset": 0},
            {"tokens": [2] * 35, "offset": 0},
            {"tokens": [3] * 30, "offset": 0},
            {"tokens": [4] * 32, "offset": 0},
        ]

        config = TrainingConfig(**sample_config_dict)
        batches = list(iterate_batches(dataset, config))

        batch_tokens, _ = batches[0]
        T = batch_tokens.shape[1]

        # Should be padded to 64 (next multiple of 32 after 37)
        assert T % 32 == 0
        assert T >= 37
        assert T == 64  # Exactly 64 for max_len=37
