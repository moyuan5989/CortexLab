"""Tests for trainer infrastructure (M4).

All tests skip until M4 implementation is complete.
"""

from __future__ import annotations

import pytest


class TestOptimizerFactory:
    def test_build_adam_optimizer(self):
        pytest.skip("M4: trainer infrastructure not yet implemented")

    def test_lr_schedule_changes_over_steps(self):
        pytest.skip("M4: trainer infrastructure not yet implemented")


class TestCheckpointManager:
    def test_save_produces_three_files(self):
        pytest.skip("M4: trainer infrastructure not yet implemented")

    def test_load_restores_state(self):
        pytest.skip("M4: trainer infrastructure not yet implemented")

    def test_atomic_write_uses_tmp_dir(self):
        pytest.skip("M4: trainer infrastructure not yet implemented")

    def test_retention_keeps_last_n_plus_best(self):
        pytest.skip("M4: trainer infrastructure not yet implemented")


class TestMetricsLogger:
    def test_jsonl_output_format(self):
        pytest.skip("M4: trainer infrastructure not yet implemented")
