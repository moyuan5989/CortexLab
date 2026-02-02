# LMForge v0 — Implementation Plan

> Milestones M0–M6 mapping to CLAUDE.md §11 implementation steps.
> Each milestone must be functional and tested before proceeding to the next.

---

## M0: Scaffolding

**CLAUDE.md steps**: Prerequisite (not numbered)

**Deliverables**:
- `pyproject.toml` with all dependencies and console_script entry point
- `.gitignore` for Python
- Full package directory structure with stub files
- `examples/train.yaml` — valid config matching V0_DESIGN_FREEZE.md §2.1
- `tests/` directory with pytest skip stubs for all milestones

**Done when**:
- `pip install -e ".[dev]"` succeeds
- `python -c "from lmforge import prepare, train"` succeeds
- `lmforge --help`, `lmforge prepare --help`, `lmforge train --help` all work
- `pytest tests/ -v` collects all tests, all skip
- `python -c "from lmforge.config import TrainingConfig; TrainingConfig.from_yaml('examples/train.yaml')"` succeeds

---

## M1: Config System

**CLAUDE.md steps**: Step 1

**Deliverables**:
- `config.py` — all 7 Pydantic models fully implemented with validators
- `TrainingConfig.from_yaml()` with CLI override support
- All config validation tests passing (not skipped)

**Key files**:
- `lmforge/config.py`
- `tests/test_config.py`

**Done when**:
- Valid YAML loads into `TrainingConfig` without error
- Invalid configs raise `ValidationError` with clear messages
- `steps_per_save % grad_accumulation_steps` check works
- `targets`/`preset` mutual exclusion works
- `extra="forbid"` rejects unknown fields
- All `test_config.py` tests pass

---

## M2: Data Pipeline (Prepare + Batching)

**CLAUDE.md steps**: Steps 2–3

**Deliverables**:
- `data/formats.py` — format detection + validation
- `data/preprocessing.py` — tokenization + offset computation
- `data/cache.py` — fingerprinting + safetensors shard write/read
- `data/batching.py` — sort-by-length + fixed-batch + pad-to-32 iterator
- `prepare()` library function
- `cli/prepare_cmd.py` — `lmforge prepare` command
- All data pipeline tests passing

**Key files**:
- `lmforge/data/formats.py`
- `lmforge/data/preprocessing.py`
- `lmforge/data/cache.py`
- `lmforge/data/batching.py`
- `lmforge/__init__.py` (wire up `prepare()`)
- `lmforge/cli/prepare_cmd.py`
- `tests/test_data.py`

**Done when**:
- `lmforge prepare --data train.jsonl --model <model> --output <dir>` succeeds
- Cache shards match V0_DESIGN_FREEZE.md §2.5 layout
- Re-running with same inputs skips tokenization (cache hit)
- Batch shapes match contract: `(B, T)` tokens and `(B, 2)` lengths
- Padding is to nearest multiple of 32, capped at `max_seq_length`
- All `test_data.py` tests pass

---

## M3: Model Loading + Adapters

**CLAUDE.md steps**: Steps 4–5

**Deliverables**:
- `models/loader.py` — model + tokenizer loading
- `adapters/targeting.py` — glob matching, preset resolution, `named_modules()`
- `adapters/lora.py` — `LoRALinear`, `LoRAEmbedding`, `from_base()`, `fuse()`, `apply_lora()`
- All adapter tests passing

**Key files**:
- `lmforge/models/loader.py`
- `lmforge/adapters/targeting.py`
- `lmforge/adapters/lora.py`
- `tests/test_adapters.py`

**Done when**:
- A model loads and produces logits via forward pass
- Glob patterns resolve correctly against model module tree
- Presets expand to correct patterns
- LoRA applied only to targeted modules (verified by inspection)
- `fuse()` merges LoRA weights back correctly
- All `test_adapters.py` tests pass

---

## M4: Training Infrastructure

**CLAUDE.md steps**: Steps 6–8

**Deliverables**:
- `trainer/optimizer.py` — `build_optimizer()`, `build_scheduler()`
- `trainer/checkpoint.py` — `CheckpointManager` with atomic save/load + retention
- `trainer/callbacks.py` — full `Callback`, `CallbackList`, `MetricsLoggerCallback`, `ConsoleCallback`, `WandBCallback`
- `logging/metrics.py` — JSONL writer + console formatter
- All trainer infra tests passing

**Key files**:
- `lmforge/trainer/optimizer.py`
- `lmforge/trainer/checkpoint.py`
- `lmforge/trainer/callbacks.py`
- `lmforge/logging/metrics.py`
- `tests/test_trainer_infra.py`

**Done when**:
- Optimizer builds from config with correct LR schedule
- LR changes over steps as expected
- Checkpoint save produces exactly 3 files per V0_DESIGN_FREEZE.md §2.3
- Checkpoint load restores all state correctly
- Retention policy keeps last N + best
- Atomic writes work (tmp dir → rename)
- JSONL metrics file written correctly
- All `test_trainer_infra.py` tests pass

---

## M5: Trainer + Run Management

**CLAUDE.md steps**: Steps 9–10

**Deliverables**:
- `trainer/trainer.py` — `Trainer.fit()`, `Trainer.evaluate()`, cooperative pause
- `trainer/state.py` — `TrainState` (already complete from M0)
- `manifest.py` — `RunManifest`, `EnvironmentInfo`, hardware collection
- `lmforge/__init__.py` — `train()` top-level function (full implementation)
- `cli/train_cmd.py` — `lmforge train` command
- Run directory creation + manifest writing

**Key files**:
- `lmforge/trainer/trainer.py`
- `lmforge/manifest.py`
- `lmforge/__init__.py`
- `lmforge/cli/train_cmd.py`

**Done when**:
- `lmforge train --config train.yaml` runs end-to-end
- Run directory matches V0_DESIGN_FREEZE.md §2.4 layout
- `manifest.json` and `environment.json` written at run start
- Loss decreases over training steps
- Checkpoints saved per schedule
- Cooperative pause saves checkpoint and blocks
- `train()` library API works from Python without CLI

---

## M6: Integration Tests

**CLAUDE.md steps**: Step 11

**Deliverables**:
- End-to-end: prepare → train → verify checkpoint → resume → no loss spike
- Config validation: all known-bad configs fail with clear messages
- Adapter targeting: glob patterns resolve correctly for ≥2 model architectures
- All integration tests passing

**Key files**:
- `tests/test_integration.py`

**Done when**:
- Full pipeline works: `lmforge prepare` → `lmforge train` → checkpoint → resume
- Resume does not exhibit loss spike (< 5% deviation from moving average)
- All acceptance criteria from V0_DESIGN_FREEZE.md §7 are met
- All `test_integration.py` tests pass
