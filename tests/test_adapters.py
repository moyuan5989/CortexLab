"""Tests for adapter targeting and LoRA (M3).

All tests skip until M3 implementation is complete.
"""

from __future__ import annotations

import pytest


class TestPresetResolution:
    def test_attention_qv_preset(self):
        pytest.skip("M3: adapters not yet implemented")

    def test_unknown_preset_raises(self):
        pytest.skip("M3: adapters not yet implemented")


class TestGlobMatching:
    def test_glob_matches_expected_modules(self):
        pytest.skip("M3: adapters not yet implemented")

    def test_no_match_raises_with_available_paths(self):
        pytest.skip("M3: adapters not yet implemented")


class TestLoRAApplication:
    def test_lora_applied_to_targeted_modules_only(self):
        pytest.skip("M3: adapters not yet implemented")

    def test_fuse_merges_weights_correctly(self):
        pytest.skip("M3: adapters not yet implemented")
