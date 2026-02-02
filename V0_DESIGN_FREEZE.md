# V0 Design Freeze

> LMForge v0 — LoRA SFT on MLX for Apple Silicon
> Status: FROZEN. Changes require explicit versioned amendment.

---

## 1. Overview & Goals

LMForge v0 is a closed-loop LoRA SFT training system for Apple Silicon via MLX.

### v0 guarantees

- LoRA fine-tuning of any Hugging Face-compatible LLM that MLX supports.
- Glob-based adapter targeting with explicit module path matching.
- Tier-1 checkpointing: adapter weights + optimizer state + step counter.
- State-consistent resume: training continues without loss spike after checkpoint reload.
- Library-first API: all operations callable as Python functions with typed config objects.
- Pre-tokenized data caching to disk for repeatable, fast data loading.
- Structured metrics logging (JSONL) for every run.

### v0 does NOT attempt

- Bit-identical training reproducibility across runs or resumes.
- Inference, generation, or serving.
- DPO, RLHF, full fine-tuning, or any recipe other than SFT.
- Sequence packing, dynamic batching, or streaming data.
- Distributed or multi-device training.
- Daemon, job queue, GUI backend, or REST API.
- Vision-language models, MoE-specific adapters, or custom Metal kernels.

---

## 2. Frozen Contracts

These contracts are immutable within the v0 line. Any change requires a schema version bump and a migration path.

### 2.1 Config Schema

All user-authored configs are Pydantic v2 `BaseModel` subclasses with `model_config = ConfigDict(extra="forbid")`. Every serialized config includes `schema_version: int`. The framework refuses to load a config with an unrecognized schema version.

#### TrainingConfig (top-level)

```python
class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = 1
    model: ModelConfig
    adapter: AdapterConfig
    data: DataConfig
    training: TrainingParams
    runtime: RuntimeConfig = RuntimeConfig()
```

#### ModelConfig

```python
class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str                                   # HF repo ID or local path
    tokenizer_path: Optional[str] = None        # override tokenizer (default: same as path)
    trust_remote_code: bool = False
```

#### AdapterConfig

```python
class AdapterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["lora"] = "lora"
    targets: Optional[list[str]] = None         # glob patterns
    preset: Optional[str] = None                # named preset
    num_layers: Optional[int] = None            # apply to last N layers only
    rank: int = 8
    scale: float = 20.0
    dropout: float = 0.0
```

Constraints:
- `targets` and `preset` are mutually exclusive. Exactly one must be provided.
- Available presets: `attention-qv`, `attention-all`, `mlp`, `all-linear`.

#### DataConfig

```python
class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    train: str                                  # path to train.jsonl
    valid: str                                  # path to valid.jsonl
    test: Optional[str] = None                  # path to test.jsonl (optional)
    cache_dir: str = "~/.lmforge/cache/preprocessed"
    max_seq_length: int = 2048
    mask_prompt: bool = True                    # mask prompt tokens from loss
```

#### TrainingParams

```python
class TrainingParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    batch_size: int = 4
    num_iters: int = 1000
    learning_rate: float = 1e-5
    optimizer: Literal["adam", "adamw", "sgd", "adafactor"] = "adam"
    optimizer_config: dict = {}                 # kwargs passed to optimizer constructor
    lr_schedule: Optional[LRScheduleConfig] = None
    grad_accumulation_steps: int = 1
    max_grad_norm: Optional[float] = None       # gradient clipping (None = disabled)
    seed: int = 42

    steps_per_report: int = 10
    steps_per_eval: int = 200
    steps_per_save: int = 100
    val_batches: int = 25
    keep_last_n_checkpoints: int = 3
```

Constraints:
- `steps_per_save` must be a multiple of `grad_accumulation_steps`. This guarantees no checkpoint is taken mid-accumulation.

#### LRScheduleConfig

```python
class LRScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str                                   # MLX scheduler name (e.g., "cosine_decay")
    arguments: list                             # positional args to scheduler constructor
    warmup: int = 0                             # warmup steps (linear ramp)
    warmup_init: float = 0.0                    # LR at step 0 during warmup
```

LR schedules are **stateless functions of step number**. On resume, the scheduler is reconstructed from config and given the saved step. No scheduler internal state is checkpointed.

#### RuntimeConfig

```python
class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_dir: str = "~/.lmforge/runs"
    eager: bool = False                         # disable mx.compile for debugging
    report_to: Optional[str] = None             # "wandb" or None
    wandb_project: Optional[str] = None
```

### 2.2 Batch Contract

The interface between the data pipeline and the trainer is exactly:

```
batch_tokens: mx.array  dtype=int32  shape=(B, T)
lengths:      mx.array  dtype=int32  shape=(B, 2)
```

- `B` is the batch size (fixed per run).
- `T` is the padded sequence length (padded to the nearest multiple of 32, capped at `max_seq_length`).
- `lengths[:, 0]` is the prompt offset per sample (number of prompt tokens; loss starts at 1-indexed step position equal to this value).
- `lengths[:, 1]` is the total unpadded token count per sample (used as exclusive upper bound: loss is computed for steps `< lengths[:, 1]`).
- Padding value is 0.

The loss mask is computed as:

```python
steps = mx.arange(1, targets.shape[1] + 1)
mask = (steps >= lengths[:, 0:1]) & (steps < lengths[:, 1:2])
```

The trainer receives `(batch_tokens, lengths)` tuples from the data iterator. Nothing else crosses this boundary.

### 2.3 Checkpoint Format

Directory-based. One directory per checkpoint. Atomic writes via temp-dir-then-rename.

```
step-NNNNNNN/
├── adapters.safetensors      # trainable model parameters (LoRA weights)
├── optimizer.safetensors     # optimizer state (Adam moments, etc.)
└── state.json                # training state metadata
```

Each checkpoint contains exactly three files. The run-level manifest and environment info are stored at the run root, not inside individual checkpoints.

#### `adapters.safetensors`

Contains all trainable parameters as returned by `model.trainable_parameters()`. Keys are dot-separated module paths (e.g., `model.layers.0.self_attn.q_proj.lora_a`). Format: MLX safetensors.

#### `optimizer.safetensors`

Contains the full optimizer state tree. For Adam/AdamW, this includes first and second moment estimates for every trainable parameter. Keys mirror the model parameter paths with optimizer-specific suffixes. Format: MLX safetensors.

#### `state.json`

```json
{
  "schema_version": 1,
  "step": 1000,
  "epoch": 0,
  "trained_tokens": 4096000,
  "best_val_loss": 1.823,
  "learning_rate": 5e-6,
  "rng_seed": 42
}
```

Field definitions:
- `schema_version`: Always 1 for v0. Used for forward-compatible loading.
- `step`: Iteration number at time of save.
- `epoch`: Epoch counter (v0 is iteration-based; this is metadata only).
- `trained_tokens`: Cumulative tokens processed for loss computation.
- `best_val_loss`: Lowest validation loss observed up to this checkpoint.
- `learning_rate`: LR at the moment of save. **For logging only** — on resume, the scheduler recomputes from step.
- `rng_seed`: The original training seed. On resume, the framework re-seeds with `mx.random.seed(rng_seed + step)`.

All fields are required. Unknown fields are ignored on load (forward compatibility).

#### Atomic Save

1. Create a temporary directory: `step-NNNNNNN.tmp/`
2. Write all three files into it.
3. `os.rename()` to `step-NNNNNNN/` (atomic on APFS).
4. After successful rename, enforce retention policy.

If the process dies between steps 1 and 3, the `.tmp` directory is ignored on next load.

#### Accumulation Boundary Rule

`steps_per_save` MUST be a multiple of `grad_accumulation_steps`. This is enforced by config validation. This guarantees that no checkpoint is taken mid-accumulation, eliminating the need to save gradient accumulation buffers.

#### mx.random.state

`mx.random.state` is NOT saved in v0 checkpoints. It is an opaque array whose format may change across MLX versions. Instead, `rng_seed` is saved, and on resume the framework re-seeds with `mx.random.seed(rng_seed + step)`. This provides different-but-valid random sequences after resume. Exact random state restoration is a Tier-2 concern.

### 2.4 Run Directory Layout

```
~/.lmforge/
├── runs/
│   └── {run_id}/
│       ├── config.yaml                        # frozen copy of user-authored config
│       ├── manifest.json                      # frozen config + environment + data fingerprint
│       ├── environment.json                   # python, mlx, hardware, os details
│       ├── checkpoints/
│       │   ├── step-0000100/
│       │   │   ├── adapters.safetensors
│       │   │   ├── optimizer.safetensors
│       │   │   └── state.json
│       │   ├── step-0000200/
│       │   │   ├── adapters.safetensors
│       │   │   ├── optimizer.safetensors
│       │   │   └── state.json
│       │   └── best -> step-0000100           # symlink to best val_loss checkpoint
│       └── logs/
│           └── metrics.jsonl                  # one JSON object per event
│
└── cache/
    └── preprocessed/
        └── {data_fingerprint}/
            ├── meta.json
            ├── shard_000.safetensors
            └── shard_001.safetensors
```

#### Run ID Format

`YYYYMMDD-HHMMSS-sft-{model_short}-{hash4}`

- `sft`: Recipe name (always `sft` in v0).
- `model_short`: Last component of model path, truncated to 20 characters.
- `hash4`: First 4 hex characters of SHA-256 of the serialized config.

Example: `20260201-143022-sft-Llama-3.2-3B-Instru-a3f1`

#### manifest.json

```json
{
  "schema_version": 1,
  "config": { },
  "lmforge_version": "0.1.0",
  "mlx_version": "0.22.0",
  "python_version": "3.11.9",
  "hardware": {
    "chip": "Apple M2 Ultra",
    "memory_gb": 192,
    "gpu_cores": 76,
    "os": "Darwin 24.6.0"
  },
  "data_fingerprint": "sha256:abcdef...",
  "created_at": "2026-02-01T14:30:22Z"
}
```

The `config` field contains the complete, frozen `TrainingConfig` as a dict. This is the authoritative record of what was configured for the run. Written once at run start and never modified.

#### environment.json

```json
{
  "python_version": "3.11.9",
  "mlx_version": "0.22.0",
  "lmforge_version": "0.1.0",
  "platform": "darwin",
  "os_version": "Darwin 24.6.0",
  "chip": "Apple M2 Ultra",
  "memory_gb": 192,
  "gpu_cores": 76
}
```

#### Checkpoint Retention

Default: keep last 3 checkpoints + the best checkpoint. Older checkpoints are deleted after a new one is written. The `best` symlink is updated whenever a new checkpoint has a lower `val_loss` than the current best.

### 2.5 Data Cache Format

```
~/.lmforge/cache/preprocessed/{data_fingerprint}/
├── meta.json
├── shard_000.safetensors
├── shard_001.safetensors
└── ...
```

#### Shard Layout

Each shard is a safetensors file containing:

```
shard_NNN.safetensors:
    tokens_0:   int32[L0]      # token IDs for sample 0
    tokens_1:   int32[L1]      # token IDs for sample 1
    ...
    offsets:    int32[N]        # prompt offset per sample in this shard
    lengths:    int32[N]        # total token count per sample in this shard
```

- `tokens_{i}` keys store variable-length token ID sequences for each sample.
- `offsets` is a 1D array with one entry per sample: the number of prompt tokens (used for loss masking).
- `lengths` is a 1D array with one entry per sample: the total token count.
- Target shard size: ~500MB. Samples are grouped sequentially into shards.

#### meta.json

```json
{
  "schema_version": 1,
  "num_samples": 50000,
  "num_shards": 2,
  "total_tokens": 25600000,
  "format": "chat",
  "data_fingerprint": "sha256:...",
  "tokenizer_hash": "sha256:...",
  "template_hash": "sha256:...",
  "max_length": 4096,
  "min_length": 12,
  "mean_length": 512.0,
  "created_at": "2026-02-01T14:30:22Z"
}
```

#### Fingerprinting

The data fingerprint is computed from three components:

- **Data file hash**: `sha256(raw_file_bytes)`
- **Tokenizer vocab hash**: `sha256(json.dumps(sorted(tokenizer.get_vocab().items())))`
- **Template hash**: `sha256(tokenizer.chat_template or "")`
- **Combined**: `sha256(data_hash + tokenizer_hash + template_hash)`

If any component changes, the cache is invalidated and rebuilt.

---

## 3. Reproducibility: Tier-1 Definition

### Target: State-Consistent Resume

On resume from a checkpoint, the training loss curve must not exhibit a visible discontinuity. The optimizer continues from its saved state (including momentum/adaptive moments). The learning rate schedule continues from the saved step (recomputed, not restored). The model parameters are identical to the moment of save.

### What Tier-1 saves

| Saved to Checkpoint | Not Saved to Checkpoint |
|---------------------|-------------------------|
| Adapter weights — `adapters.safetensors` | `mx.random.state` (opaque, version-dependent) |
| Optimizer state (moments) — `optimizer.safetensors` | numpy random state |
| step, epoch, trained_tokens — `state.json` | Data iterator position |
| best_val_loss, rng_seed — `state.json` | Gradient accumulation buffer |
| learning_rate (logging only, not used on restore) — `state.json` | Scheduler internal state (recomputed from config + step) |

This table matches the `state.json` schema defined in §2.3 exactly. There are no additional saved fields.

### Why bit-identical is not achievable

1. `mx.compile` traces computation graphs and fuses operations. The order of floating-point accumulation in fused kernels is not guaranteed across traces.
2. Metal compute shaders may execute reductions in non-deterministic order.
3. Different input shapes cause `mx.compile` to retrace, potentially producing different fusion patterns.
4. These are fundamental properties of the MLX/Metal stack, not implementation bugs.

### Resume semantics

- Adapter weights: restored exactly from `adapters.safetensors`.
- Optimizer state: restored exactly from `optimizer.safetensors`.
- LR schedule: reconstructed from config, given the saved step. Stateless — produces identical LR for any given step.
- RNG: re-seeded with `mx.random.seed(rng_seed + step)`. Produces different-but-valid random sequences after resume.
- Data ordering: re-derived from seed. May differ from original run after resume point (Tier-2 concern).

### Deferred tiers

**Tier 2 — Reproducible ordering**: Additionally saves `mx.random.state`, numpy random state, data iterator position, gradient accumulation buffer. On resume: exact same batch sequence continues.

**Tier 3 — Audit-grade**: Additionally saves data fingerprints, environment snapshot, framework version pinning. For compliance use cases.

---

## 4. Adapter Targeting Rules

### Mechanism: Glob-Based Path Matching

Adapter targets are specified as a list of glob patterns that match against module paths. Module paths are dot-separated strings (e.g., `model.layers.15.self_attn.q_proj`).

```yaml
adapter:
  method: lora
  targets:
    - "*.self_attn.q_proj"
    - "*.self_attn.v_proj"
  rank: 8
  scale: 20.0
  dropout: 0.0
```

Patterns are matched using Python's `fnmatch.fnmatch()` on the full dot-separated module path.

**Important**: `fnmatch` treats `.` as a regular character, not a path separator. The `*` wildcard matches any sequence of characters *including dots*. This means `*.self_attn.q_proj` matches `model.layers.0.self_attn.q_proj` because `*` matches `model.layers.0`. This is intentional and correct for the built-in presets. Users writing custom patterns should be aware that `*` is not component-aware on dot-separated paths.

### Presets

Presets are named shorthand for a list of glob patterns:

```yaml
adapter:
  method: lora
  preset: "attention-qv"
  rank: 8
```

Built-in presets for v0:

| Preset | Resolves To |
|--------|-------------|
| `attention-qv` | `["*.self_attn.q_proj", "*.self_attn.v_proj"]` |
| `attention-all` | `["*.self_attn.q_proj", "*.self_attn.k_proj", "*.self_attn.v_proj", "*.self_attn.o_proj"]` |
| `mlp` | `["*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"]` |
| `all-linear` | `["*.self_attn.q_proj", "*.self_attn.k_proj", "*.self_attn.v_proj", "*.self_attn.o_proj", "*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"]` |

A preset and explicit targets are mutually exclusive. Config validation rejects configs that specify both.

### No Type-Based Fallback

The framework does NOT scan for `nn.Linear` or any other type to determine targets. If no `targets` or `preset` is specified, config validation fails with an error listing available presets and example patterns for the loaded model.

### Resolution Error

After resolving globs against the model's module tree, if zero modules match, the framework raises an error listing:
- The patterns that were attempted.
- The first 20 available module paths in the model.

### Layer Count Targeting

An optional `num_layers` parameter limits adapter application to the last N transformer blocks:

```yaml
adapter:
  preset: "attention-qv"
  num_layers: 16      # apply to last 16 layers only
  rank: 8
```

Layer index is extracted from the module path (e.g., `layers.{N}`). Only modules where `N >= total_layers - num_layers` are eligible. Modules without a layer index in their path (e.g., `model.embed_tokens`) are excluded when `num_layers` is set.

---

## 5. Training Control Semantics

### Compiled Step (Default)

The training step function is wrapped with `@mx.compile(inputs=state, outputs=state)` by default. The compile state list includes `model.state`, `optimizer.state`, and `mx.random.state`.

**Distinction**: This compile state is the live computation state tracked by `mx.compile` during training execution. It is NOT the same as checkpoint state. In particular, `mx.random.state` is in the compile state (so the compiled graph handles RNG correctly) but is NOT saved to checkpoints (see §2.3). Checkpoint state is a strict subset defined by the three files in each checkpoint directory.

An `eager: bool = False` flag on `RuntimeConfig` disables compilation for debugging. When `eager=True`, the same step function runs without `mx.compile`.

### Callback Boundaries

Callbacks execute OUTSIDE the compiled region. The execution order per step:

```
compiled_step(batch) → (loss, n_tokens, grad_accum)
    │
    ▼
mx.eval(state, loss, n_tokens, grad_accum)    ← SAFE POINT
    │
    ▼
callbacks.on_step_end(state, metrics)
```

Callbacks MUST NOT modify model parameters, optimizer state, or any object referenced by the compiled step function. Callbacks may read state (after `mx.eval`) and write to external systems (loggers, files).

### v0 Callback Hooks

```python
class Callback:
    def on_train_begin(self, state: TrainState) -> None: ...
    def on_train_end(self, state: TrainState) -> None: ...
    def on_step_end(self, state: TrainState, metrics: dict) -> None: ...
    def on_eval_end(self, state: TrainState, metrics: dict) -> None: ...
    def on_save(self, state: TrainState, checkpoint_dir: Path) -> None: ...
```

Additional hooks (`on_epoch_begin`, `on_step_begin`, etc.) are deferred to post-v0.

### Cooperative Pause

Pause is signaled by setting a `threading.Event`. The training loop checks this event at every safe point (after `mx.eval()`). On pause:

1. Save a checkpoint at the current step.
2. Block until resume signal.
3. Continue the loop from the current state.

SIGTSTP is intercepted and converted to the cooperative pause signal. The process is NOT actually stopped by the OS.

### Failure Handling

- **OOM during step**: Catch the exception, save the last successful state, log the error, and exit.
- **SIGINT / Ctrl+C**: Intercepted in the training loop. Save checkpoint at current state, then exit.
- **Corrupted checkpoint on load**: Validate all expected files exist and state.json has required keys before restoring. If validation fails, report the error and refuse to resume.

---

## 6. Out of Scope for v0

| Feature | Reason |
|---------|--------|
| Inference / generation engine | Separate concern |
| Serving (OpenAI-compatible API) | Requires inference |
| Sequence packing | Complex mask handling |
| Dynamic batching by total tokens | Interacts with mx.compile caching |
| DPO / RLHF / KTO / ORPO recipes | Scope creep |
| Full parameter fine-tuning | Different optimizer requirements |
| Distributed / multi-device training | Multi-device complexity |
| DoRA adapters | Additional adapter math |
| IA3, prefix tuning | Additional adapter types |
| Daemon / job queue | Requires stable training core |
| GUI backend / WebSocket | Requires daemon |
| Custom Metal kernels | Optimization, not foundation |
| GGUF export | Inference concern |
| Model merging (beyond LoRA fusing) | Complex merging logic |
| Model conversion (HF → MLX) | Utility, not core |
| Gradient checkpointing / activation recomputation | Memory optimization |
| Automatic batch size detection | Convenience feature |
| HuggingFace streaming datasets | Data pipeline extension |
| Quantization (AWQ, GPTQ, dynamic) | Inference concern |
| Plugin / registry system | Extensibility concern |
| RecipeProtocol or abstract recipe interface | Over-engineering for single recipe |
| Multi-dataset mixing | Data pipeline extension |
| Evaluation harness integration (lm-eval) | Separate concern |
| Perplexity measurement | Inference concern |
| Upload to HuggingFace Hub | Utility, not core |

---

## 7. Acceptance Criteria

### Must Pass

1. **Resume without loss spike**: Train for 200 steps, save checkpoint, resume, train for 200 more steps. The loss curve at step 200 must not exhibit a visible discontinuity (< 5% deviation from the moving average at step 199).

2. **Backward-compatible checkpoints**: A checkpoint saved by any v0.x release must be loadable by any later v0.y release (y >= x). Unknown fields are ignored; missing optional fields use defaults.

3. **Adapter targeting works or errors**: If targets resolve to zero modules, the framework raises an error before any computation begins. If targets resolve to modules, exactly those modules are converted to LoRA and nothing else.

4. **Config validation catches errors at startup**: Invalid types, missing required fields, `steps_per_save % grad_accumulation_steps != 0`, mutual exclusion of `targets`/`preset`, and other constraint violations are caught before model loading.

5. **Metrics file is always written**: Every run produces a `metrics.jsonl` file with at least one event (even if training fails on step 1).

6. **Library API works without CLI**: `lmforge.train(config)` and `lmforge.prepare(data_path, model)` are callable from Python without argparse, CLI wrappers, or namespace objects.

7. **Data cache is deterministic**: Running `lmforge prepare` twice on the same data with the same tokenizer produces identical cache files (byte-for-byte same safetensors).
