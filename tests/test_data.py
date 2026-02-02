"""Tests for data pipeline (M2).

All tests skip until M2 implementation is complete.
"""

from __future__ import annotations

import pytest


class TestFormatDetection:
    def test_detect_chat_format(self):
        pytest.skip("M2: data pipeline not yet implemented")

    def test_detect_completions_format(self):
        pytest.skip("M2: data pipeline not yet implemented")

    def test_detect_text_format(self):
        pytest.skip("M2: data pipeline not yet implemented")

    def test_unknown_format_raises(self):
        pytest.skip("M2: data pipeline not yet implemented")


class TestCacheFingerprinting:
    def test_same_inputs_same_fingerprint(self):
        pytest.skip("M2: data pipeline not yet implemented")

    def test_different_data_different_fingerprint(self):
        pytest.skip("M2: data pipeline not yet implemented")


class TestBatching:
    def test_batch_shapes_match_contract(self):
        pytest.skip("M2: data pipeline not yet implemented")

    def test_padding_to_multiple_of_32(self):
        pytest.skip("M2: data pipeline not yet implemented")
