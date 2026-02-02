# M7: Hugging Face Model Loading — Design Document

> Status: PROPOSED
> Author: Claude (Architect)
> Date: 2026-02-02

---

## Overview

M7 adds a **model resolution layer** that turns a Hugging Face model ID (e.g., `Qwen/Qwen3-0.8B`) into a deterministic, locally-cached, offline-capable model directory — before any training or data preparation begins.

The current system already accepts HF repo IDs in `model.path` and delegates to `mlx_lm.load()` and `AutoTokenizer.from_pretrained()`. However, this delegation is opaque: LMForge has no visibility into what revision was loaded, no guarantee of offline capability after first use, and no way to record the resolved model identity in the manifest.

M7 adds a thin resolution pre-step. It does NOT replace mlx_lm's model loading or weight conversion logic.

### Design Principles

1. **Zero friction for existing configs** — `model.path: "Qwen/Qwen3-0.8B"` continues to work. No new required fields. No new CLI steps.
2. **Deterministic after resolution** — Once a model is resolved, the exact revision is pinned for the lifetime of the run.
3. **Offline after first resolution** — All network access happens during resolution. Training never touches the network.
4. **v0 contracts preserved** — No changes to checkpoint format, batch contract, resume semantics, or training loop.

---

## 1. UX Decision

### Recommendation: Automatic resolution, no explicit `pull` command

**Users write exactly what they write today:**

```yaml
model:
  path: "Qwen/Qwen3-0.8B"
```

Resolution happens automatically at the earliest point where the model is needed: the beginning of `lmforge prepare` or `lmforge train`. If the model is not cached locally, it is downloaded. If it is already cached, no network access occurs.

**Why not `lmforge model pull`?**

| Concern | Why automatic wins |
|---------|-------------------|
| LMForge philosophy | Library-first. `train()` and `prepare()` are Python functions. Adding a mandatory pre-step breaks this — you'd need `pull()` → `prepare()` → `train()` instead of just `prepare()` → `train()`. |
| User friction | Every other MLX/HF tool auto-downloads. Requiring an explicit pull step is a paper cut that accumulates. Users coming from LLaMA-Factory, Unsloth, or mlx-lm expect `path: "org/model"` to just work. |
| Offline use case | Automatic resolution already handles this: if the model is cached, resolution succeeds offline. If it's not cached and the user is offline, the error message tells them exactly what to do. |
| Pre-downloading | Users who want explicit control already have `huggingface-cli download Qwen/Qwen3-0.8B`. LMForge doesn't need to duplicate this. |

**What users see:**

```
$ lmforge train --config train.yaml

LMForge v0 — Training
Resolving model: Qwen/Qwen3-0.8B
  → Resolved to revision a1b2c3d (cached)
  → Local path: ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.8B/snapshots/a1b2c3d

Loading model and tokenizer...
...
```

Or, on first use:

```
$ lmforge train --config train.yaml

LMForge v0 — Training
Resolving model: Qwen/Qwen3-0.8B
  → Downloading from Hugging Face Hub...
  → Downloaded 1.6 GB in 45s
  → Revision: a1b2c3d
  → Local path: ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.8B/snapshots/a1b2c3d

Loading model and tokenizer...
...
```

### Optional: `revision` field in ModelConfig

Add one new optional field to ModelConfig:

```python
class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str                                   # HF repo ID or local path
    tokenizer_path: Optional[str] = None        # override tokenizer
    trust_remote_code: bool = False
    revision: Optional[str] = None              # NEW: pin to specific HF revision
```

When `revision` is `None` (default): resolve to the latest commit hash at runtime and record it in the manifest.

When `revision` is specified: use it exactly. Fail if that revision doesn't exist.

**This is a backward-compatible schema extension.** Existing configs without `revision` continue to validate. The field has a default value (`None`). No existing behavior changes. This qualifies as an additive extension under the v0 contract, not a breaking change.

---

## 2. Resolution Pipeline

### Where: `lmforge/models/resolve.py` (new file)

A single new module containing the resolution logic. Separate from `loader.py` (which handles the actual model loading via mlx_lm).

### What the resolver returns

```python
@dataclass
class ResolvedModel:
    """Result of model resolution."""
    local_path: str          # Absolute path to local model directory
    repo_id: str | None      # HF repo ID (None if local path)
    revision: str | None     # Resolved commit hash (None if local path)
    is_local: bool           # True if path was already local
```

### When resolution happens

Resolution happens **once**, at the very beginning of `train()` or `prepare()`, before any other work. The resolved path is then passed to all downstream consumers (`load_model()`, `AutoTokenizer.from_pretrained()`, manifest writing).

```
User config                Resolution                  Loading
─────────                  ──────────                  ───────
model.path: "Qwen/..."  → resolve_model(config)     → load_model(resolved.local_path)
                           ↓                           ↓
                           ResolvedModel               (model, tokenizer)
                           ↓
                           manifest.model_resolution
```

### Resolution flow (step-by-step)

```
resolve_model(model_config: ModelConfig) → ResolvedModel
```

**Step 1: Classify the path**

```python
path = model_config.path
if Path(path).expanduser().exists():
    # Local path — no resolution needed
    return ResolvedModel(
        local_path=str(Path(path).expanduser().resolve()),
        repo_id=None,
        revision=None,
        is_local=True,
    )
```

**Step 2: Resolve HF revision**

If `model_config.revision` is specified, use it. Otherwise, query the Hub for the latest commit hash:

```python
from huggingface_hub import model_info

if model_config.revision:
    revision = model_config.revision
else:
    info = model_info(path)  # Network call
    revision = info.sha       # Full commit hash
```

This is the ONLY network call that resolution makes (aside from the download in step 3). It pins the exact revision for reproducibility.

**Step 3: Ensure model is cached locally**

```python
from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id=path,
    revision=revision,
    # Uses standard HF cache: ~/.cache/huggingface/hub/
)
```

`snapshot_download` is idempotent: if the snapshot is already cached, it returns the local path immediately with no network access.

**Step 4: Return resolved model**

```python
return ResolvedModel(
    local_path=local_path,
    repo_id=path,
    revision=revision,
    is_local=False,
)
```

### Integration with existing code

The resolved local path replaces the raw `model.path` in all downstream calls:

```python
# In train():
resolved = resolve_model(config.model)

# Model loading uses resolved local path
model, tokenizer = load_model(resolved.local_path, ...)

# Prepare uses resolved local path for tokenizer
tokenizer = AutoTokenizer.from_pretrained(resolved.local_path, ...)

# Manifest records resolution metadata
manifest.model_resolution = { ... }
```

### Tokenizer path resolution

If `model_config.tokenizer_path` is specified and is an HF repo ID, apply the same resolution logic. If it's `None`, the tokenizer comes from the model path (already resolved).

---

## 3. Caching & Fingerprinting

### Cache location: Standard HF cache (no LMForge-specific model cache)

Models are cached in the standard Hugging Face Hub cache directory:

```
~/.cache/huggingface/hub/
└── models--Qwen--Qwen3-0.8B/
    ├── refs/
    │   └── main              # Contains the commit hash
    ├── snapshots/
    │   └── a1b2c3d.../       # The actual model files
    │       ├── config.json
    │       ├── model.safetensors
    │       ├── tokenizer.json
    │       └── ...
    └── blobs/
        └── ...               # Content-addressed blob storage
```

**Why NOT a LMForge-specific cache:**

1. **No duplicate storage.** Models are multi-GB. Copying them into `~/.lmforge/cache/models/` doubles disk usage for no benefit.
2. **Ecosystem compatibility.** Other tools (`mlx_lm`, `transformers`, `huggingface-cli`) share the same cache. A model downloaded by any tool is available to all.
3. **Battle-tested cache management.** `huggingface_hub` handles partial downloads, concurrent access, integrity checks, and cleanup (`huggingface-cli cache info`).
4. **Less code to maintain.** LMForge doesn't need to implement its own blob storage, concurrent download management, or partial-download recovery.

### What uniquely identifies a cached model

The HF cache is content-addressed by `(repo_id, revision)`. This pair is sufficient for deterministic resolution:

| Component | How it's captured |
|-----------|------------------|
| HF repo ID | `model_config.path` |
| Revision / commit hash | Resolved at runtime, or pinned via `model_config.revision` |

**Conversion options (dtype, quantization) are NOT part of the cache key.** The resolution layer caches the raw HF model files. If mlx_lm performs weight conversion, that's handled by mlx_lm's own caching. LMForge does not manage or fingerprint converted weights — it delegates entirely to mlx_lm for the loading step.

### Partial downloads / failed conversions

**Partial downloads:** Handled entirely by `huggingface_hub.snapshot_download()`. It uses atomic writes internally and resumes incomplete downloads. If a download is interrupted, the next call to `snapshot_download()` resumes from where it left off. No LMForge-specific logic needed.

**Failed mlx_lm conversions:** If mlx_lm fails during weight conversion (after download), the error propagates to the user. The raw HF files remain cached (download was successful). The user can retry without re-downloading. LMForge does not attempt to cache or manage converted weights.

### Data fingerprint interaction

The data fingerprint (`sha256(data_hash + tokenizer_hash + template_hash)`) is unaffected by model resolution. The tokenizer hash is computed from the tokenizer vocabulary and chat template, which are properties of the resolved model. As long as the same revision is resolved, the same tokenizer is loaded, and the same fingerprint is produced.

If a model is updated on HF Hub (new revision), the tokenizer may change, which would produce a different fingerprint, which would trigger a cache miss in the data pipeline. This is correct behavior.

---

## 4. Reproducibility & Manifests

### Additions to `manifest.json`

Add a new top-level field `model_resolution` to the manifest:

```json
{
  "schema_version": 1,
  "config": { ... },
  "lmforge_version": "0.1.0",
  "mlx_version": "0.30.4",
  "python_version": "3.14.2",
  "hardware": { ... },
  "data_fingerprint": "sha256:...",
  "created_at": "2026-02-02T10:30:00Z",

  "model_resolution": {
    "repo_id": "Qwen/Qwen3-0.8B",
    "revision": "a1b2c3def4567890abcdef1234567890abcdef12",
    "local_path": "/Users/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.8B/snapshots/a1b2c3d...",
    "is_local": false,
    "resolved_at": "2026-02-02T10:29:55Z"
  }
}
```

For local models:

```json
{
  "model_resolution": {
    "repo_id": null,
    "revision": null,
    "local_path": "/path/to/local/model",
    "is_local": true,
    "resolved_at": "2026-02-02T10:29:55Z"
  }
}
```

### Why this preserves v0 semantics

1. **The `config` field is unchanged.** It still contains the complete, frozen `TrainingConfig` as a dict. The `model.path` field in the config records what the user specified ("Qwen/Qwen3-0.8B"), not the resolved path.

2. **`model_resolution` is additive.** Existing code that reads `manifest.json` and ignores unknown fields continues to work. The v0 contract states: "Unknown fields are ignored on load (forward compatibility)."

3. **Checkpoint format is unchanged.** Checkpoints still contain exactly 3 files. Model resolution metadata is stored at the run level (manifest), not inside checkpoints.

4. **Resume semantics are unchanged.** On resume, the model is loaded from the same path (config is frozen in the run). The manifest's `model_resolution` is informational — it records what happened, but the resumed run uses the same config.

### How this enables reproducibility

A user can reproduce a run by:

1. Reading the manifest: `"repo_id": "Qwen/Qwen3-0.8B", "revision": "a1b2c3d..."`
2. Setting `revision: "a1b2c3d..."` in their config
3. Running `lmforge train` — which resolves to the exact same model

Without the revision, re-running with the same config may pick up a newer model revision. The manifest captures the ground truth of what was actually used.

---

## 5. Failure Modes & Ergonomics

### Missing HF token / gated models

```
Error: Model 'meta-llama/Llama-3.2-3B-Instruct' requires authentication.

This is a gated model. To access it:
  1. Accept the license at https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct
  2. Set your HF token:  huggingface-cli login
  3. Or set the environment variable:  export HF_TOKEN=hf_...

Then retry your command.
```

Detection: `huggingface_hub.model_info()` raises `GatedRepoError` or `RepositoryNotFoundError` with a 401/403 status. Catch these specifically and provide the actionable message above.

### Network failure

```
Error: Cannot resolve model 'Qwen/Qwen3-0.8B': network unavailable.

If you have previously downloaded this model, it may be in your HF cache.
To check: huggingface-cli cache info

To use a local model path instead:
  model:
    path: "/path/to/local/model"
```

Detection: `requests.ConnectionError` or `huggingface_hub.HfHubHTTPError` during `model_info()` or `snapshot_download()`. If a cached version exists (check the HF cache), offer to use it with a warning:

```
Warning: Cannot reach Hugging Face Hub. Using cached version of 'Qwen/Qwen3-0.8B'
  Cached revision: a1b2c3d (downloaded 2026-01-15)
  ⚠ This may not be the latest version.
```

### Offline mode

Explicit offline mode via environment variable (`HF_HUB_OFFLINE=1`) or a future config field. When offline:

1. Skip the `model_info()` call entirely.
2. Attempt `snapshot_download()` with `local_files_only=True`.
3. If the model is cached, succeed silently.
4. If not cached, fail with a clear message:

```
Error: Model 'Qwen/Qwen3-0.8B' not found in local cache (offline mode).

Download it first:
  HF_HUB_OFFLINE=0 lmforge train --config train.yaml
  # or
  huggingface-cli download Qwen/Qwen3-0.8B
```

### Insufficient disk space

`huggingface_hub.snapshot_download()` will raise an `OSError` if disk space runs out. Catch this and report:

```
Error: Insufficient disk space to download 'Qwen/Qwen3-0.8B'.

  Required:  ~1.6 GB
  Available: 0.3 GB on /Users/user/.cache/huggingface

Free disk space or change the cache directory:
  export HF_HOME=/Volumes/external/hf_cache
```

Model size can be estimated from `model_info().siblings` (sum of file sizes) before starting the download.

### Model not found

```
Error: Model 'Qwenn/Qwen3-0.8B' not found on Hugging Face Hub.

Did you mean one of these?
  Qwen/Qwen3-0.8B
  Qwen/Qwen3-0.6B

Check the model ID at https://huggingface.co/models
```

Detection: `RepositoryNotFoundError` from `model_info()`. Fuzzy suggestions are a nice-to-have, not required for M7.

### Reusing an existing cached model

This is the common case after first download. Resolution detects the cached snapshot and completes instantly with no network access (beyond the `model_info()` call to check the latest revision, which is fast and can be skipped in offline mode).

Output:

```
Resolving model: Qwen/Qwen3-0.8B
  → Resolved to revision a1b2c3d (cached)
```

---

## 6. What NOT to Do (Explicitly)

| Anti-pattern | Why it's wrong |
|---|---|
| **Download weights into run directories** | Run directories contain config + checkpoints + logs. Copying multi-GB model weights into each run wastes disk and breaks the v0 run directory layout contract. |
| **Silent model revision changes** | If a user re-runs the same config a month later and gets a different model revision without being told, reproducibility is silently broken. Always resolve to a specific revision and record it. |
| **Training while downloading** | Network I/O during training introduces latency jitter, Metal memory pressure, and non-determinism. Resolution must complete BEFORE any model loading or training begins. |
| **LMForge-managed model cache** | Duplicating huggingface_hub's cache management (blob storage, partial downloads, deduplication) is unnecessary engineering. Use the standard HF cache. |
| **Automatic model conversion/quantization** | M7 resolves and downloads. Conversion to MLX format is mlx_lm's responsibility. LMForge does not implement or manage weight format conversion. |
| **Model registry / model card parsing** | LMForge is not a model hub browser. It resolves a specific model ID to a local path. |
| **Adding `lmforge model` CLI subcommand** | Against library-first philosophy. Users who want explicit download control use `huggingface-cli download`. |
| **Modifying checkpoint format** | Model resolution metadata goes in the run-level manifest, NOT inside checkpoint directories. The 3-file checkpoint contract is frozen. |
| **Server / daemon / background download** | No background processes. Resolution is synchronous, blocking, and predictable. |
| **Changing existing `model.path` semantics** | `model.path` continues to accept both HF repo IDs and local paths. The field name does not change. Existing configs work unchanged. |

---

## 7. Implementation Scope

### New files

| File | Contents |
|------|----------|
| `lmforge/models/resolve.py` | `resolve_model()`, `ResolvedModel` dataclass, error handling |

### Modified files

| File | Change |
|------|--------|
| `lmforge/config.py` | Add `revision: Optional[str] = None` to `ModelConfig` |
| `lmforge/models/loader.py` | Accept `ResolvedModel` or use resolved `local_path` instead of raw `model.path` |
| `lmforge/__init__.py` | Call `resolve_model()` at start of `train()` and `prepare()` |
| `lmforge/manifest.py` | Add `model_resolution` field to manifest output |
| `tests/test_integration.py` | Test resolution with mock HF responses |

### NOT modified

| File | Why unchanged |
|------|--------------|
| `lmforge/trainer/trainer.py` | Trainer receives a loaded model. Resolution is upstream. |
| `lmforge/trainer/checkpoint.py` | Checkpoint format is frozen. |
| `lmforge/data/` | Data pipeline is unchanged. Uses resolved tokenizer path. |
| `V0_DESIGN_FREEZE.md` | No frozen contracts are modified. `revision` is an additive optional field. |
| `lmforge/cli/main.py` | No new CLI subcommands. |

### Dependencies

No new dependencies. `huggingface_hub` is already installed as a transitive dependency of `transformers`.

---

## Design Decisions Summary

| Decision | Choice | Why |
|----------|--------|-----|
| **Explicit pull command?** | No — automatic resolution | Library-first philosophy. Matches user expectations from mlx-lm/Unsloth. No friction. |
| **When does resolution happen?** | At start of `prepare()` and `train()`, before any model loading | Ensures all network access happens upfront. Training never touches network. |
| **Where do models cache?** | Standard HF cache (`~/.cache/huggingface/hub/`) | No duplicate storage. Ecosystem compatibility. Battle-tested cache management. |
| **LMForge-specific model cache?** | No | Unnecessary engineering. HF cache is content-addressed and handles concurrent access, partial downloads, and cleanup. |
| **How is the revision pinned?** | Resolve latest via `model_info()`, pin commit hash, record in manifest | Deterministic after first resolution. Reproducible via `revision` field in config. |
| **New config field?** | `revision: Optional[str] = None` in ModelConfig | Backward-compatible additive extension. Enables explicit pinning for reproducibility. |
| **Model conversion?** | Delegated to mlx_lm | LMForge is not a model converter. mlx_lm handles PyTorch→MLX conversion transparently. |
| **Manifest changes?** | Add `model_resolution` top-level field | Additive. Records what was actually used. Existing tools that read manifests ignore unknown fields. |
| **Offline behavior?** | Respect `HF_HUB_OFFLINE=1`, fall back to cached versions, clear errors | Standard HF ecosystem behavior. No LMForge-specific offline flag needed. |
| **Error messages?** | Specific, actionable messages for each failure mode | Tell the user what went wrong, what to do about it, and what command to run. No stack traces for user errors. |
| **Local path handling?** | Detect local path via `Path.exists()`, skip all HF resolution | Zero overhead for users with local models. Same code path after resolution. |
