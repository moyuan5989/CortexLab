# LMForge — Development Guide

> Contributor reference for architecture, conventions, and common tasks.

---

## Project Overview

LMForge is a LoRA SFT training framework for MLX on Apple Silicon with a browser-based Studio UI.

**Package structure:**
```
lmforge/
├── adapters/           # LoRA targeting, application, fusing
├── cli/                # CLI commands (prepare, train, generate, studio, data)
├── config.py           # Pydantic config models
├── data/               # Data pipeline (formats, batching, packing, catalog)
├── inference/          # Text generation, sampling, KV cache
├── models/             # Model registry, loader, architectures
│   ├── _base/          # Shared utilities (attention, RoPE, activations)
│   └── architectures/  # Llama, Qwen2/3/3.5, Phi-3/4, Gemma
├── studio/             # FastAPI backend + React frontend
│   ├── api/            # REST endpoints
│   ├── services/       # Run/model/dataset/training services
│   └── frontend/       # Pre-built React SPA
└── trainer/            # Training loop, checkpointing, callbacks
```

---

## Design Principles

- **Library-first**: All operations are Python functions; CLI is a thin wrapper
- **Explicit targeting**: LoRA adapters applied via glob patterns (e.g., `*.self_attn.q_proj`)
- **Fail fast**: Validate all configs before loading models or data
- **Stateless LR schedules**: Pure functions of step number (reconstructed on resume)
- **No database**: Filesystem is the database (Studio reads `~/.lmforge/` directly)
- **Pydantic v2**: Config models use `extra="forbid"` — new fields must be optional with backward-compatible defaults

---

## Frozen Contracts (DO NOT BREAK)

1. **Checkpoint format**: Exactly 3 files per checkpoint: `adapters.safetensors`, `optimizer.safetensors`, `state.json`
2. **Batch contract**: `(B, T)` input_ids + `(B, T)` labels with `-100` masking
3. **Config schema**: Pydantic v2 models with `extra="forbid"`, schema version stays at 1
4. **Run directory layout**: `~/.lmforge/runs/{run_id}/` with config.yaml, manifest.json, checkpoints/, logs/
5. **Data storage**: `~/.lmforge/datasets/raw/` (JSONL) + `~/.lmforge/datasets/processed/` (Arrow)

---

## MLX Gotchas

1. **Fancy index assignment**: Direct `arr[indices] = values` may not work. Use boolean mask + `mx.where()` or inverse permutation via `mx.argsort(mx.argsort(indices))`

2. **`.at[]` accessor exists**: `arr.at[0].add(10.0)` works (like JAX)

3. **QLoRA gradient flow**: Must use `nn.value_and_grad(model, loss_fn)` — not `mx.value_and_grad()` which tries to diff through QuantizedMatmul

4. **QLoRA freeze order**: `model.freeze()` → `nn.quantize(model, bits=4)` → `apply_lora(model, targets, cfg)`

5. **API names**: `mx.set_wired_limit()` and `mx.get_peak_memory()` (not `mx.metal.*`)

---

## Testing

```bash
# Run all 510 tests
.venv/bin/python -m pytest tests/ -v

# Run a specific file
.venv/bin/python -m pytest tests/test_config.py -v
```

**Test files:**
- `test_config.py` — Config validation
- `test_data.py` — Data pipeline, batching, fingerprinting
- `test_adapters.py` — LoRA targeting and application
- `test_model_loading.py` — Model loading and architectures
- `test_resolve.py` — HF model resolution
- `test_trainer_infra.py` — Trainer components
- `test_integration.py` — End-to-end workflows
- `test_labels.py` — Per-token label construction
- `test_backend.py` — Arrow storage backend
- `test_catalog.py` — Dataset catalog and registry
- `test_v2_dpo.py` — DPO/preference training
- `test_v2_studio.py` — V2 contracts and architectures
- `test_m9_foundation.py` — Resume, inference, Gemma
- `test_m10_performance.py` — QLoRA, packing, gradient checkpointing
- `test_m11_studio.py` — Studio backend
- `test_m12_frontend.py` — Studio frontend
- `test_m13_integration.py` — Cross-feature integration

---

## Adding a New Architecture

1. Create `lmforge/models/architectures/newmodel.py` implementing:
   ```python
   class NewModelArgs(BaseModelArgs):
       @classmethod
       def from_dict(cls, config: dict) -> "NewModelArgs": ...

   class NewModel(nn.Module):
       def __call__(self, inputs: mx.array, cache=None) -> mx.array:
           # inputs: (B, T) → logits: (B, T, vocab_size)
   ```

2. Register in `lmforge/models/registry.py`:
   ```python
   SUPPORTED_ARCHITECTURES["newmodel"] = "lmforge.models.architectures.newmodel"
   ```

3. Add tests in `tests/test_model_loading.py`

---

## Adding a LoRA Preset

Edit `lmforge/adapters/targeting.py`:
```python
PRESETS["my-preset"] = ["*.module1", "*.module2"]
```

---

## Frontend Development

The Studio frontend lives in `studio-frontend/` (React + Vite + TypeScript):

```bash
cd studio-frontend
npm install
npm run build    # Output → lmforge/studio/frontend/
npm run dev      # Dev server with hot reload
```

Notes:
- Tailwind CSS v4: uses `@import "tailwindcss"` in CSS (no config file)
- Backend `RunService._sanitize_for_json()` handles inf/nan floats

---

## Git Conventions

Commit message format: `<type>: <subject>`

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `chore`

Never commit: `.env`, credentials, large binaries, `__pycache__/`, `.venv/`, `node_modules/`

---

## Dependencies

```bash
# Core
pip install -e "."

# With Studio
pip install -e ".[studio]"

# Full dev setup
pip install -e ".[dev,studio]"
```
