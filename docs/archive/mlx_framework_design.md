# MLX-Native LLM Engineering Framework: Design Document

> **STATUS: NON-AUTHORITATIVE REFERENCE**
> This document is a pre-design exploration of the mlx-lm codebase. It is NOT authoritative for LMForge v0 implementation.
> The authoritative v0 documents are: **V0_DESIGN_FREEZE.md** (frozen contracts) and **CLAUDE.md** (implementation guide).
> Sections 1–8 contain useful mlx-lm analysis and design patterns. Sections 9–12 contain aspirational designs
> that are **superseded** by the v0 scope — do not implement anything from those sections.

> Deep analysis of mlx-lm + architecture proposal for a new framework
> Based on source-level inspection of the mlx-lm codebase

---

## SECTION 1 — Repository Structure Map

### Directory Tree

```
mlx_lm/                          (181 Python files total)
├── __init__.py                  PUBLIC API: load, generate, batch_generate, stream_generate, convert
├── __main__.py                  Entry: delegates to cli.main()
├── _version.py                  Version string
├── cli.py                       CLI dispatcher: routes 16 subcommands to modules
│
├── # ── INFERENCE ──
├── generate.py                  CORE: generate_step, stream_generate, batch_generate, BatchGenerator
├── chat.py                      CLI: interactive multi-turn chat loop
├── sample_utils.py              Sampling: top-p, top-k, min-p, XTC, temperature
├── tokenizer_utils.py           TokenizerWrapper, streaming detokenizer
├── server.py                    OpenAI-compatible REST API (ThreadingHTTPServer, SSE streaming)
├── cache_prompt.py              Pre-compute and save KV caches
├── benchmark.py                 Throughput benchmarking
│
├── # ── TRAINING ──
├── lora.py                      Finetuning entry: config loading, train_model(), evaluate_model()
├── tuner/
│   ├── __init__.py              Exports: TrainingArgs, evaluate, train, linear_to_lora_layers
│   ├── trainer.py               CORE: train(), evaluate(), iterate_batches(), TrainingArgs, grad_checkpoint
│   ├── datasets.py              TextDataset, ChatDataset, CompletionsDataset, CacheDataset, load_dataset()
│   ├── lora.py                  LoRALinear, LoRASwitchLinear, LoRAEmbedding
│   ├── dora.py                  DoRALinear, DoRAEmbedding (weight-decomposed LoRA)
│   ├── callbacks.py             TrainingCallback, WandBCallback, SwanLabCallback
│   ├── losses.py                Cross-entropy, DPO loss functions
│   └── utils.py                 linear_to_lora_layers(), build_schedule(), load_adapters()
│
├── # ── MODELS (112 model implementations) ──
├── models/
│   ├── __init__.py              Empty (dynamic import via importlib)
│   ├── base.py                  BaseModelArgs, create_attention_mask, scaled_dot_product_attention
│   ├── cache.py                 KVCache, RotatingKVCache, QuantizedKVCache, BatchKVCache, save/load
│   ├── rope_utils.py            RoPE initialization and variants
│   ├── activations.py           SwiGLU and custom activations
│   ├── switch_layers.py         SwitchLinear, QuantizedSwitchLinear (MoE routing)
│   ├── ssm.py                   State space model components
│   ├── llama.py                 LLaMA / Mistral architecture
│   ├── qwen2.py, qwen3.py      Qwen architectures
│   ├── gemma.py, gemma2.py      Gemma architectures
│   ├── deepseek_v2.py, v3.py   DeepSeek architectures
│   ├── mamba.py, mamba2.py      SSM architectures
│   ├── mixtral.py               MoE architectures
│   └── ... (100+ more model files)
│
├── # ── CONVERSION / QUANTIZATION ──
├── convert.py                   HuggingFace → MLX format conversion
├── fuse.py                      Merge LoRA adapters into base model
├── gguf.py                      GGUF format export for llama.cpp
├── quant/
│   ├── awq.py                   Activation-Weighted Quantization
│   ├── gptq.py                  GPTQ quantization
│   ├── dwq.py                   Data-free quantization
│   ├── dynamic_quant.py         Dynamic inference-time quantization
│   └── utils.py                 Shared quantization utilities
│
├── # ── EVALUATION ──
├── evaluate.py                  lm-eval harness integration
├── perplexity.py                Perplexity measurement
│
├── # ── MISC ──
├── utils.py                     Model loading, saving, quantization, sharding, HF Hub
├── manage.py                    HF cache management
├── upload.py                    Hub upload
├── tool_parsers/                Model-specific function-calling parsers (7 files)
└── chat_templates/              Custom chat template formatters (1 file)
```

### Module Responsibility Summary

| Module | Responsibility | Type | Domain |
|--------|---------------|------|--------|
| `generate.py` | Token generation engine | Core logic | Inference |
| `utils.py` | Model loading, saving, quantization | Core logic | Shared |
| `models/cache.py` | KV cache implementations | Core logic | Inference |
| `models/base.py` | Attention primitives, base args | Core logic | Shared |
| `tuner/trainer.py` | Training loop, evaluation | Core logic | Training |
| `tuner/lora.py` | LoRA layer implementations | Core logic | Training |
| `lora.py` | Training entry point + config | CLI glue | Training |
| `server.py` | REST API server | CLI glue | Inference |
| `cli.py` | Subcommand dispatcher | CLI glue | Misc |
| `convert.py` | Format conversion | Utils | Misc |
| `fuse.py` | Adapter merging | Utils | Training |

---

## SECTION 2 — Inference Execution Trace

### Call Graph: `mlx_lm generate`

```
cli.main()
└── generate.main()                                [generate.py:1342]
    ├── setup_arg_parser()                          [generate.py:59]
    ├── mx.random.seed(args.seed)
    ├── load_prompt_cache() [optional]              [models/cache.py]
    ├── load(model_path, adapter_path, ...)         [utils.py:440]
    │   ├── _download(path_or_hf_repo)              [utils.py:205]
    │   │   └── snapshot_download()                 HF Hub or local
    │   ├── load_model(model_path)                  [utils.py:269]
    │   │   ├── load_config()                       reads config.json
    │   │   ├── mx.load(*.safetensors)              lazy weight loading
    │   │   ├── _get_classes(config)                [utils.py:162]
    │   │   │   └── importlib.import_module(f"mlx_lm.models.{model_type}")
    │   │   ├── ModelArgs.from_dict(config)         [models/base.py:11]
    │   │   ├── Model(model_args)                   instantiate model
    │   │   ├── model.sanitize(weights) [optional]
    │   │   ├── nn.quantize(model, ...)  [if quantized]
    │   │   ├── model.load_weights(weights)
    │   │   └── mx.eval(model.parameters())         force into memory
    │   ├── load_adapters(model, adapter_path) [optional]
    │   └── load_tokenizer(model_path)              [tokenizer_utils.py]
    │       └── TokenizerWrapper(AutoTokenizer.from_pretrained(...))
    ├── tokenizer.apply_chat_template(messages)
    ├── make_sampler(temp, top_p, min_p, ...)       [sample_utils.py]
    └── generate(model, tokenizer, prompt, ...)     [generate.py:745]
        └── stream_generate(model, tokenizer, ...)  [generate.py:646]
            ├── tokenizer.encode(prompt) → mx.array
            └── with wired_limit(model, [stream]):  [generate.py:225]
                └── generate_step(prompt, model, ...) [generate.py:303]
                    │
                    ├── PHASE 1: PROMPT PREFILL
                    │   cache = make_prompt_cache(model)     [models/cache.py]
                    │   while remaining_tokens > 1:
                    │       model(chunk[None], cache=cache)  forward pass
                    │       mx.eval([c.state for c in cache]) force KV computation
                    │       mx.clear_cache()                  release intermediates
                    │
                    ├── PHASE 2: FIRST TOKEN
                    │   y, logprobs = _step(last_prompt_token)
                    │   mx.async_eval(y, logprobs)           queue computation
                    │
                    └── PHASE 3: GENERATION LOOP
                        while n < max_tokens:
                            next_y, next_logprobs = _step(y) pipeline next
                            mx.async_eval(next_y, next_logprobs)
                            if n == 0: mx.eval(y)           sync first token
                            yield y.item(), logprobs
                            if n % 256 == 0: mx.clear_cache()
                            y, logprobs = next_y, next_logprobs
```

### Key Design Decisions

1. **Dedicated generation stream**: `generation_stream = mx.new_stream(mx.default_device())` — all generation ops run on a separate stream to enable async overlap.

2. **Wired memory management**: `wired_limit()` context manager sets `mx.set_wired_limit(max_recommended_working_set_size)` to prevent page-out on Apple Silicon.

3. **Two-phase pipeline**: During generation, the *next* token is computed async while the *current* token is yielded. This hides GPU latency behind Python/detokenization work.

4. **Prefill chunking**: `prefill_step_size=2048` prevents OOM by processing prompt in chunks, evaluating KV cache state after each chunk, then clearing intermediates.

5. **Speculative decoding**: Separate path (`speculative_generate_step`) that drafts N tokens with a smaller model, verifies with the main model, and accepts/rejects. KV caches are rewound on rejection.

---

## SECTION 3 — Finetuning Execution Trace

### Complete Call Graph

```
cli.main()
└── lora.main()                                      [lora.py:343]
    ├── build_parser()                                argparse setup
    ├── yaml.load(config_file)                        YAML config loading
    ├── CONFIG_DEFAULTS merge                         [lora.py:42]
    └── run(args)                                     [lora.py:312]
        ├── get_reporting_callbacks()                  [callbacks.py:109]
        │   └── WandBCallback / SwanLabCallback
        ├── load(args.model)                           [utils.py:440]
        ├── load_dataset(args, tokenizer)              [datasets.py:309]
        │   ├── load_local_dataset()                   reads train.jsonl, valid.jsonl, test.jsonl
        │   │   └── create_dataset(data, tokenizer)    [datasets.py:175]
        │   │       ├── ChatDataset                    {"messages": [...]}
        │   │       ├── CompletionsDataset             {"prompt": ..., "completion": ...}
        │   │       └── TextDataset                    {"text": ...}
        │   └── load_hf_dataset()                      HuggingFace datasets
        │
        └── train_model(args, model, train, valid)     [lora.py:209]
            ├── model.freeze()                         freeze ALL parameters
            ├── linear_to_lora_layers(model, ...)      [tuner/utils.py:38]
            │   ├── Discover: scan model.layers for nn.Linear, nn.Embedding, etc.
            │   ├── Convert: LoRALinear.from_base(layer, r, scale, dropout)
            │   └── Apply: model.update_modules(lora_layers)
            ├── model.load_weights(resume_file) [opt]  resume training
            ├── build_schedule(lr_schedule)             [tuner/utils.py:18]
            ├── optimizer = Adam/AdamW/SGD/Muon(lr=schedule)
            │
            └── train(model, optimizer, ...)            [tuner/trainer.py:204]
                ├── mx.set_wired_limit(max_recommended) memory setup
                ├── mx.distributed.init()               distributed setup
                ├── grad_checkpoint(model.layers[0])    [optional]
                ├── loss_value_and_grad = nn.value_and_grad(model, loss)
                │
                ├── state = [model.state, optimizer.state, mx.random.state]
                ├── @mx.compile(inputs=state, outputs=state)
                │   def step(batch, prev_grad, do_update):
                │       (lvalue, toks), grad = loss_value_and_grad(model, *batch)
                │       if prev_grad: grad += prev_grad    # accumulate
                │       if do_update:
                │           grad = average_gradients(grad)  # distributed sync
                │           optimizer.update(model, grad)
                │       return lvalue, toks, grad
                │
                └── TRAINING LOOP:
                    for it in range(1, args.iters + 1):
                        batch = next(iterate_batches(...))
                        │
                        ├── [EVAL] if it == 1 or it % steps_per_eval == 0:
                        │   val_loss = evaluate(model, val_dataset, ...)
                        │   callback.on_val_loss_report(val_info)
                        │
                        ├── [STEP] lvalue, toks, grad_accum = step(batch, grad_accum, do_update)
                        │   mx.eval(state, losses, n_tokens, grad_accum)
                        │
                        ├── [LOG] if it % steps_per_report == 0:
                        │   train_loss = all_sum(losses) / steps
                        │   callback.on_train_loss_report(train_info)
                        │
                        └── [SAVE] if it % steps_per_save == 0:
                            adapter_weights = tree_flatten(model.trainable_parameters())
                            mx.save_safetensors(adapter_file, adapter_weights)
                            mx.save_safetensors(f"{it:07d}_adapters.safetensors", adapter_weights)
```

### Training State Objects

```python
# Compiled step function state (passed to mx.compile)
state = [model.state, optimizer.state, mx.random.state]

# Loop-level tracking (NOT checkpointed)
losses: float = 0           # accumulated loss since last report
n_tokens: int = 0           # tokens since last report
steps: int = 0              # steps since last report
trained_tokens: int = 0     # total tokens trained
train_time: float = 0       # wall time since last report
grad_accum: tree = None     # accumulated gradients
```

### Loss Function

```python
def default_loss(model, batch, lengths):     # [trainer.py:75]
    inputs = batch[:, :-1]                    # shift right
    targets = batch[:, 1:]
    logits = model(inputs)                    # forward pass
    steps = mx.arange(1, targets.shape[1] + 1)
    mask = (steps >= lengths[:, 0:1]) & (steps <= lengths[:, 1:])
    ce = nn.losses.cross_entropy(logits, targets) * mask
    ntoks = mask.sum()
    return ce.sum() / ntoks, ntoks
```

The `lengths` tensor carries `[prompt_offset, total_length]` per sample — enabling prompt masking (train only on completion tokens).

### Memory Lifecycle (Training)

1. **Wired limit**: `mx.set_wired_limit(max_recommended_working_set_size)` at start
2. **mx.compile**: JIT-compiles `step()` — reduces memory fragmentation, enables kernel fusion
3. **mx.eval per step**: `mx.eval(state, losses, n_tokens, grad_accum)` forces computation, prevents unbounded graph growth
4. **Gradient checkpointing**: Optional — recomputes activations during backward via `mx.checkpoint(inner_fn)`
5. **No mx.clear_cache in training**: Unlike inference, the training loop does NOT call `mx.clear_cache()` — the compiled step manages memory

### Resume Logic

**Weak resume**: Only adapter weights are restored via `model.load_weights(resume_file, strict=False)`. Optimizer state, iteration counter, and random state are NOT saved. This means:
- Learning rate restarts from initial value
- Optimizer momentum/adaptive state is lost
- Gradient accumulation buffer resets
- Data ordering may differ (no seed tracking)

---

## SECTION 4 — LoRA/DoRA Implementation Deep Dive

### Layer Injection Mechanism

**Discovery** (`tuner/utils.py:85-101`):
```python
def get_keys_for_lora(p, m):
    types = (nn.Linear, nn.QuantizedLinear, SwitchLinear,
             QuantizedSwitchLinear, nn.Embedding, nn.QuantizedEmbedding)
    if hasattr(m, "to_lora") or isinstance(m, types):
        keys.add(p)

for l in model.layers:
    l.apply_to_modules(get_keys_for_lora)
```

**Conversion** (`tuner/utils.py:103-106`):
```python
for l in model.layers[-max(num_layers, 0):]:
    lora_layers = [(k, to_lora(m)) for k, m in l.named_modules() if k in keys]
    l.update_modules(tree_unflatten(lora_layers))
```

This is a **type-based visitor pattern** — not a registry. The code scans the last N transformer blocks, finds all Linear/Embedding modules, wraps them in LoRA variants, and applies in-place.

### LoRA Forward Pass

```python
# tuner/lora.py:95-98
def __call__(self, x):
    y = self.linear(x)                              # frozen base: W @ x
    z = (self.dropout(x) @ self.lora_a) @ self.lora_b  # low-rank: x @ A @ B
    return y + (self.scale * z).astype(x.dtype)     # combined
```

- `lora_a`: shape `(input_dims, r)`, init `U(-1/√d, 1/√d)`
- `lora_b`: shape `(r, output_dims)`, init `zeros`
- At init: LoRA contribution is zero (B=0), so training starts from base model behavior

### DoRA Forward Pass

```python
# tuner/dora.py:111-128
def __call__(self, x):
    w = self._dequantized_weight()
    y = x @ w.T
    z = (self.dropout(x) @ self.lora_a) @ self.lora_b
    out = y + (self.scale * z).astype(x.dtype)

    adapted = w + (self.scale * self.lora_b.T) @ self.lora_a.T
    denom = mx.stop_gradient(mx.linalg.norm(adapted, axis=1))
    out = (self.m / denom).astype(x.dtype) * out     # magnitude correction

    if "bias" in self.linear:
        out = out + self.linear.bias
    return out
```

DoRA adds a **magnitude vector** `self.m` (original weight norms) and scales output by `m / ||W + ΔW||`. The `stop_gradient` prevents gradients from flowing through the norm computation — only through the LoRA matrices and the magnitude.

### Weight Saving & Merging

**Save**: `dict(tree_flatten(model.trainable_parameters()))` → only `*.lora_a`, `*.lora_b` keys saved to safetensors

**Fuse** (`fuse.py:68-72`):
```python
fused_linears = [
    (n, m.fuse(dequantize=args.dequantize))
    for n, m in model.named_modules()
    if hasattr(m, "fuse")
]
model.update_modules(tree_unflatten(fused_linears))
```

LoRA fuse: `W_new = W_old + scale * (B^T @ A^T)`
DoRA fuse: `W_new = (m / ||W_old + ΔW||) * (W_old + ΔW)`

### Assessment: Clean Abstraction

**Strengths:**
- LoRA logic fully isolated from model code — no model files modified
- `from_base()` factory wraps existing layer non-destructively
- `fuse()` merges back cleanly; `remove_lora_layers()` reverts
- Quantized layers handled transparently (dequantize→compute→requantize)

**Weaknesses:**
- Hard assumption: `model.layers` must be a list of transformer blocks
- MoE expert routing handled via specialized `LoRASwitchLinear` — not fully generic
- No per-layer rank configuration (all LoRA layers get same rank)
- Model-specific `to_lora()` hooks create implicit coupling

---

## SECTION 5 — Design Patterns Worth Copying

### Pattern 1: Dynamic Model Registry via importlib
**What**: `_get_classes(config)` uses `importlib.import_module(f"mlx_lm.models.{model_type}")` to load model classes
**Why good**: Zero-overhead registration. Adding a new model = adding a Python file. No central registry to update
**Where**: `utils.py:162-180`
**Generalize**: Keep this pattern but add a formal `ModelProtocol` to enforce interface contracts (layers, make_cache, sanitize, shard)

### Pattern 2: BaseModelArgs.from_dict() with auto-filtering
**What**: `cls(**{k: v for k, v in params.items() if k in inspect.signature(cls).parameters})`
**Why good**: Config files can contain extra keys — models only pick what they need. No crashes on unknown keys
**Where**: `models/base.py:11-21`
**Generalize**: Standard pattern for all config dataclasses across framework

### Pattern 3: Compiled training step with state
**What**: `@partial(mx.compile, inputs=state, outputs=state)` wraps the step function
**Why good**: JIT compilation with explicit state management. Reduces overhead and memory fragmentation
**Where**: `tuner/trainer.py:234`
**Generalize**: Make this the default strategy for all training recipes, with opt-out for debugging

### Pattern 4: LoRA as composition (wrapping, not patching)
**What**: `LoRALinear.from_base(original_linear)` stores original as `.linear` attribute
**Why good**: Non-destructive. Original weights preserved. Reversible (remove_lora_layers). Clean fuse path
**Where**: `tuner/lora.py:13-32`
**Generalize**: Pattern for all adapters (LoRA, IA3, prefix tuning, etc.)

### Pattern 5: Streaming generation as generator
**What**: `generate_step()` is a Python generator yielding `(token, logprobs)` tuples
**Why good**: Composable. `stream_generate()` wraps it with detokenization. `batch_generate()` uses `BatchGenerator`. Server uses it for SSE
**Where**: `generate.py:303-466`
**Generalize**: All generation pathways should be generators — enables consistent streaming, logging, and early stopping

### Pattern 6: Wired memory management
**What**: Context manager that sets Metal wired memory limit based on model size and device capacity
**Why good**: Prevents page-out on Mac systems with unified memory. Critical for Apple Silicon performance
**Where**: `generate.py:225-262`
**Generalize**: Framework-level resource manager that handles this automatically for both training and inference

### Pattern 7: KV cache hierarchy with polymorphism
**What**: `KVCache`, `RotatingKVCache`, `QuantizedKVCache`, `BatchKVCache` — all with `update_and_fetch()`, `state` property
**Why good**: Models choose cache type per-layer. Generation code is cache-type agnostic
**Where**: `models/cache.py`
**Generalize**: Good pattern. Add formal protocol/ABC for cache types

### Pattern 8: Dataset format auto-detection
**What**: `create_dataset()` inspects first sample to determine format (chat/completions/text)
**Why good**: User provides data, framework figures out format. Reduces configuration burden
**Where**: `tuner/datasets.py:175-202`
**Generalize**: Good pattern. Extend with schema validation and clear error messages

### Pattern 9: Async eval pipeline in generation
**What**: Compute next token async while yielding current. `mx.async_eval(next_y)` + `yield current_y`
**Why good**: Hides GPU latency behind Python work. Measurable throughput improvement
**Where**: `generate.py:451-466`
**Generalize**: Make this the standard pipeline pattern. Expose as strategy option

### Pattern 10: Prompt masking via offset/length pairs
**What**: `lengths` tensor carries `[prompt_end, total_length]` — loss mask computed as `steps >= prompt_end`
**Why good**: Clean separation. No special tokens needed. Works with any chat template
**Where**: `tuner/trainer.py:75-88`
**Generalize**: Standard for all SFT recipes. Extend for multi-turn masking

### Pattern 11: Model sanitize() hook
**What**: Optional `model.sanitize(weights)` method that transforms/filters weights before loading
**Why good**: Handles naming mismatches, removes precomputed tensors (e.g., RoPE freqs), renames keys
**Where**: `utils.py:332-333`
**Generalize**: Keep as optional hook in model protocol

### Pattern 12: Safetensors with sharding
**What**: `make_shards()` splits weights into 5GB chunks. Index file maps weight names to shard files
**Why good**: Standard format. Compatible with HF ecosystem. Enables lazy loading
**Where**: `utils.py:583-604, 699-756`
**Generalize**: Standard for all model persistence

---

## SECTION 6 — Design Problems / Technical Debt

### Problem 1: Monolithic training loop
**Problem**: `train()` function is 167 lines handling validation, reporting, saving, distributed sync, gradient accumulation — all interleaved
**Why painful**: Can't customize eval schedule, can't add early stopping, can't change save strategy without modifying the function
**Evidence**: `tuner/trainer.py:204-371`
**Better**: Decompose into Trainer class with pluggable Callback/Hook system (Lightning pattern)

### Problem 2: No optimizer/scheduler state in checkpoints
**Problem**: Only adapter weights saved. Optimizer state (Adam moments), LR scheduler step, iteration counter, RNG state — all lost on resume
**Why painful**: Resume produces different training trajectory. Gradient accumulation buffer lost. Non-reproducible
**Evidence**: `tuner/trainer.py:354-360` — saves only `model.trainable_parameters()`
**Better**: Save full `TrainState` (model + optimizer + scheduler + rng + iteration) as a single checkpoint

### Problem 3: Script-level config merging
**Problem**: Config comes from 3 sources (defaults dict, YAML file, CLI args) merged via ad-hoc logic in `main()`
**Why painful**: Precedence unclear. Type coercion fragile. No validation. YAML keys and CLI flags must match manually
**Evidence**: `lora.py:343-362` — manual merge loop with `None` checks
**Better**: Use a proper config system (Pydantic, dataclasses, or Hydra-style) with typed schemas and clear precedence

### Problem 4: Tight coupling between CLI and logic
**Problem**: `lora.py` is both the CLI entry point AND the training orchestration logic
**Why painful**: Can't call `train_model()` programmatically without argparse overhead. Config is `types.SimpleNamespace`
**Evidence**: `lora.py:312-362` — run() expects namespace with specific attribute names
**Better**: Separate CLI layer from library API. Use typed config objects

### Problem 5: No data preprocessing pipeline
**Problem**: Tokenization happens lazily inside `CacheDataset.__getitem__()`. No pre-processing, validation, or caching to disk
**Why painful**: Re-tokenizes every epoch. No data validation before training starts. No packing/bucketing
**Evidence**: `tuner/datasets.py:158-172` — `CacheDataset` is just an in-memory cache
**Better**: Pre-tokenize + validate + cache to disk. Add packing, bucketing, and data mixing

### Problem 6: No reproducibility guarantees
**Problem**: No config snapshot saved with checkpoints. No data ordering recorded. No environment info
**Why painful**: Can't reproduce a training run. Different random seeds between resumes. No audit trail
**Evidence**: `adapter_config.json` saves args but not environment, data hash, or random state
**Better**: Save full run manifest: config + environment + data fingerprint + git hash

### Problem 7: Template handling fragility
**Problem**: Chat template is applied via `tokenizer.apply_chat_template()` — a HuggingFace function with model-specific Jinja2 templates
**Why painful**: Silent failures possible. Templates may not match between training and inference. No validation
**Evidence**: Multiple template-related args: `--ignore-chat-template`, `--use-default-chat-template`, `--chat-template-config`
**Better**: Validate template at load time. Enforce template consistency between training and serving. Log template output for debugging

### Problem 8: Memory management is implicit
**Problem**: Memory budget not tracked or managed. `mx.clear_cache()` called at fixed intervals. No OOM protection
**Why painful**: Users discover OOM at runtime. No guidance on batch size vs. model size. No memory profiling
**Evidence**: `generate.py:464` — `if n % 256 == 0: mx.clear_cache()` (magic number)
**Better**: Memory budget tracker. Dynamic batch sizing. OOM-safe gradient accumulation fallback

### Problem 9: Hard-coded model.layers assumption
**Problem**: LoRA injection requires `model.layers` to be a list of transformer blocks
**Why painful**: Breaks for non-standard architectures. Can't apply LoRA to embeddings at the top-level without separate code path
**Evidence**: `tuner/utils.py:100-106` — iterates `model.layers[-num_layers:]`
**Better**: Use path-based targeting: `lora_targets = ["model.layers.*.self_attn.q_proj", "model.layers.*.self_attn.v_proj"]`

### Problem 10: Server uses stdlib http.server
**Problem**: `ThreadingHTTPServer` with manual request parsing, no middleware, no async support
**Why painful**: Thread-per-request model. No connection pooling. No proper error handling framework. No middleware for auth, rate limiting, logging
**Evidence**: `server.py:1654` — `server_class=ThreadingHTTPServer`
**Better**: Use FastAPI/Starlette for production server with proper async, middleware, and OpenAPI docs

### Problem 11: No experiment tracking integration
**Problem**: Callbacks only support loss/metric reporting. No hyperparameter logging, no artifact tracking, no comparison
**Why painful**: Manual tracking of experiments. Can't compare runs. No integration with MLflow/W&B artifacts
**Evidence**: `callbacks.py` — only `on_train_loss_report` and `on_val_loss_report` hooks
**Better**: Rich callback system with lifecycle hooks: on_train_begin, on_epoch_end, on_step_end, on_save, on_eval, etc.

### Problem 12: Batch formation is simplistic
**Problem**: Sort by length, group into fixed-size batches, pad to nearest 32. No packing, no dynamic batching
**Why painful**: Wastes compute on padding tokens. Short sequences padded to match longest in batch. No sequence packing
**Evidence**: `tuner/trainer.py:91-162` — `iterate_batches()` with fixed pad_to=32
**Better**: Sequence packing (concat multiple samples into one sequence). Dynamic batching by total tokens

---

## SECTION 7 — Framework Comparison

### Unsloth
**Layer solved**: Performance-optimized LoRA/QLoRA training on consumer GPUs
**Strengths**: 2-5x faster than HF Trainer via custom CUDA kernels, fused operations, memory-efficient attention. Quantized training (4-bit base + LoRA). Gradient checkpointing. Automatic batch size finding
**Weaknesses**: NVIDIA-only. Limited model support (manually optimized per architecture). Not a general framework. Opaque optimizations. Single-GPU focus
**Ideas to adopt**: Fused LoRA kernels (write MLX Metal equivalents), automatic batch size detection, 4-bit training optimization, efficient memory profiling

### LLaMA-Factory
**Layer solved**: Productized training UX with recipe zoo
**Strengths**: Web UI for experiment configuration. 100+ model/dataset/method combinations. YAML-based recipes. Side-by-side comparison. Multiple training methods (SFT, RLHF, DPO, ORPO). Data preprocessing pipelines. Checkpoint management
**Weaknesses**: Monolithic design. Heavy HuggingFace dependency. Recipe configs can be confusing. Limited extensibility for custom methods. Performance not optimized
**Ideas to adopt**: Recipe system with YAML configs, web UI for job management, dataset preprocessing pipelines, multi-method support (SFT → DPO progression)

### PyTorch Lightning
**Layer solved**: Training engineering abstraction
**Strengths**: Clean Trainer/Callback/Module separation. Device-agnostic training. Automatic distributed. Checkpointing with full state. Logging integration. Plugin system for strategies. Reproducibility built-in. Testing utilities
**Weaknesses**: Heavy abstraction overhead. Steep learning curve. Over-engineered for simple tasks. Framework lock-in for model code
**Ideas to adopt**: Callback lifecycle (on_train_start, on_step_end, on_epoch_end, on_save, etc.), Strategy pattern for device/distributed, CheckpointIO for state management, Trainer as the central object

### HuggingFace Transformers/Accelerate
**Layer solved**: Model zoo + device abstraction
**Strengths**: Massive model library. Unified Trainer API. PEFT library for adapters. Datasets library for data. Accelerate for distributed. Hub ecosystem. Standardized config system
**Weaknesses**: Bloated. Slow. Abstraction leaks everywhere. Training is hard to customize. Memory management is poor. Template system is fragile. Config inheritance is confusing
**Ideas to adopt**: Model protocol (PreTrainedModel interface). Config inheritance from base classes. Hub integration for sharing. Standardized tokenizer interface

### Comparison Table

| Framework | Good Ideas to Adopt | Ideas to Avoid | How to Adapt for MLX |
|-----------|-------------------|----------------|---------------------|
| **Unsloth** | Fused LoRA kernels; auto batch sizing; 4-bit training paths; memory profiler | CUDA-specific kernels; single-GPU assumption; opaque optimization magic | Write MLX Metal-native fused LoRA ops; leverage MLX unified memory for dynamic sizing |
| **LLaMA-Factory** | Recipe YAML system; Web UI + job manager; dataset pipelines; multi-method progression | HF dependency; monolithic design; config explosion | Build recipe registry; daemon backend for Mac GUI; clean data pipeline abstraction |
| **Lightning** | Trainer + Callback + Module separation; Strategy pattern; full-state checkpoints; reproducibility | Over-engineering; framework lock-in; heavy base classes | Lightweight Trainer; list-based callbacks (not class hierarchy); MLX compile-aware checkpointing |
| **HF Accelerate** | Device abstraction; config inheritance; Hub integration; PEFT adapter protocol | Bloat; leaky abstractions; poor memory mgmt; fragile templates | MLX-native device strategy; clean config system; template validation at load time |

---

## SECTION 8 — Core Objects Table

| Object | Responsibility | Lifetime | File(s) |
|--------|---------------|----------|---------|
| **Model** (nn.Module) | Forward pass, attention, MLP | Loaded once, persists through training/inference | `models/llama.py`, etc. |
| **ModelArgs** (dataclass) | Model configuration | Created at load time, immutable | `models/llama.py::ModelArgs` |
| **TokenizerWrapper** | Encoding, decoding, chat templates, streaming detokenization | Loaded with model, persists | `tokenizer_utils.py` |
| **KVCache / variants** | Key-value cache for autoregressive generation | Per-generation session; reset between requests | `models/cache.py` |
| **TrainingArgs** (dataclass) | Training hyperparameters | Created at train start, immutable | `tuner/trainer.py:37` |
| **Optimizer** (mlx.optimizers) | Parameter updates (Adam, AdamW, SGD, Muon, Adafactor) | Created at train start, state tracks momentum/adaptive | `lora.py:270-285` |
| **LR Schedule** (callable) | Learning rate as function of step | Created at train start, queried each step | `tuner/utils.py:18` |
| **TrainingCallback** | Logging hooks (on_train_loss, on_val_loss) | Created at train start, called during training | `tuner/callbacks.py:16` |
| **Dataset** (TextDataset/ChatDataset/etc) | Holds raw data, tokenizes on access | Loaded once, persists through training | `tuner/datasets.py` |
| **CacheDataset** | Lazy-caching wrapper over dataset | Wraps dataset, caches processed items in memory | `tuner/datasets.py:158` |
| **GenerationResponse** (dataclass) | Per-token generation output with metadata | Created and yielded per token | `generate.py:265` |
| **BatchGenerator** | Manages batched generation with dynamic batch composition | Per batch_generate() call | `generate.py:926` |
| **ModelProvider** (server) | Lazy model loading + caching for server | Server lifetime | `server.py:428` |
| **LRUPromptCache** (server) | Caches prompt KV states for reuse | Server lifetime, LRU eviction | `server.py:173` |

---

## SECTION 9 — 10 MUST-HAVE Framework Invariants

> **SUPERSEDED**: This section contains aspirational invariants from the pre-design phase. For v0, the authoritative invariants are defined in V0_DESIGN_FREEZE.md and CLAUDE.md. Several items below (e.g., bit-identical resume, plugin registries, OOM-safe fallbacks) are explicitly out of scope for v0.

### 1. MUST: Every run is reproducible
Full config snapshot + environment + data fingerprint + git hash + random seed saved at run start. Given the same inputs, the framework produces identical results.

### 2. MUST: Full-state checkpoints
Checkpoints include model weights, optimizer state, scheduler state, random state, iteration counter, and data loader position. Resume produces bit-identical training continuation.

### 3. MUST: Stable memory usage
Memory must not grow unboundedly during training or inference. The framework must track peak memory, enforce `mx.eval()` at correct boundaries, and provide OOM-safe fallbacks (auto gradient accumulation, dynamic batch sizing).

### 4. MUST: Library-first, CLI-second
All functionality must be importable and callable as a Python library with typed config objects. CLI is a thin wrapper that parses args into config objects. No business logic in CLI layer.

### 5. MUST: Plugin-based extensions
New training methods (DPO, ORPO, KTO), new model architectures, new data formats, new loggers, and new serving backends can be added without modifying core code. Extension points via protocols and registries.

### 6. MUST: Deterministic data preprocessing
Data is pre-tokenized, validated, and cached to disk before training starts. Token counts, sequence lengths, and data statistics are logged. Template application is validated and reproducible.

### 7. MUST: Config is typed and validated
All configuration goes through typed dataclasses or Pydantic models. Invalid configs fail at load time with clear error messages, not during training. Schema versioning for backwards compatibility.

### 8. MUST: Clean separation of concerns
Model code knows nothing about training. Training code knows nothing about CLI. Serving code knows nothing about training. Each layer communicates via defined interfaces.

### 9. MUST: Observable training
Real-time metrics (loss, LR, throughput, memory) available via structured logs, callbacks, and optional streaming API. Every run produces a structured log file regardless of callback configuration.

### 10. MUST: Artifact management
Every output (adapter weights, fused models, eval results, logs) has a defined path structure, metadata, and lineage tracking. Outputs are self-describing (contain their config and provenance).

---

## SECTION 10 — New Framework Architecture

> **SUPERSEDED**: This section proposes a full framework architecture (MLXForge) that goes far beyond v0 scope. The actual v0 package layout, class interfaces, and module structure are defined in CLAUDE.md §4. Do not implement RecipeProtocol, plugin registries, distributed strategies, or any module not listed in CLAUDE.md §4.

### Proposed Name: **MLXForge**

### Directory Structure

```
mlxforge/
├── __init__.py              # Public API surface
├── _version.py
│
├── core/                    # Framework primitives
│   ├── __init__.py
│   ├── config.py            # BaseConfig, ConfigRegistry, config validation
│   ├── protocols.py         # ModelProtocol, DatasetProtocol, AdapterProtocol
│   ├── registry.py          # Plugin registry (models, recipes, adapters, data formats)
│   ├── device.py            # MLX device management, memory tracking, wired limits
│   └── distributed.py       # Distributed strategy (data parallel, tensor parallel, pipeline)
│
├── models/                  # Model loading and management
│   ├── __init__.py
│   ├── loader.py            # ModelLoader: download, load, quantize, shard
│   ├── registry.py          # Model type → module mapping (dynamic import)
│   ├── base.py              # BaseModelArgs, BaseModel protocol
│   ├── cache.py             # KVCache hierarchy (from mlx-lm, refined)
│   ├── rope.py              # RoPE utilities
│   ├── architectures/       # Model implementations (llama.py, qwen.py, etc.)
│   │   ├── __init__.py
│   │   ├── llama.py
│   │   ├── qwen.py
│   │   └── ...
│   └── adapters/            # Adapter implementations
│       ├── __init__.py
│       ├── base.py          # AdapterProtocol: from_base(), fuse(), remove()
│       ├── lora.py          # LoRALinear, LoRASwitchLinear, LoRAEmbedding
│       ├── dora.py          # DoRALinear, DoRAEmbedding
│       └── ia3.py           # (future) IA3 adapter
│
├── data/                    # Data pipeline
│   ├── __init__.py
│   ├── formats.py           # Format detection and validation (chat, completions, text, preference)
│   ├── preprocessing.py     # Tokenization, template application, masking
│   ├── packing.py           # Sequence packing (concat short seqs, split long ones)
│   ├── batching.py          # Dynamic batching, bucketing by length
│   ├── cache.py             # Disk-cached preprocessed datasets (Arrow/memory-mapped)
│   ├── loader.py            # DataLoader: local JSONL, HuggingFace, streaming
│   └── validation.py        # Schema validation, statistics, sanity checks
│
├── trainer/                 # Training engine
│   ├── __init__.py
│   ├── trainer.py           # Trainer class: main training orchestration
│   ├── state.py             # TrainState: model + optimizer + scheduler + rng + step
│   ├── config.py            # TrainingConfig dataclass
│   ├── callbacks.py         # CallbackProtocol + built-in callbacks
│   ├── checkpoint.py        # CheckpointManager: save/load full state, keep-N, best-of
│   ├── evaluation.py        # Evaluator: validation, test, custom metrics
│   ├── optimization.py      # Optimizer/scheduler factory from config
│   └── compilation.py       # mx.compile strategy management
│
├── recipes/                 # Training recipes (method-specific)
│   ├── __init__.py
│   ├── base.py              # RecipeProtocol: defines loss, data format, eval
│   ├── sft.py               # SFT recipe: cross-entropy with prompt masking
│   ├── dpo.py               # DPO recipe: preference optimization
│   └── full_finetune.py     # Full parameter fine-tuning
│
├── generate/                # Inference engine
│   ├── __init__.py
│   ├── engine.py            # GenerationEngine: prompt processing + token generation
│   ├── sampling.py          # Sampler factory (top-p, top-k, min-p, XTC, temperature)
│   ├── speculative.py       # Speculative decoding strategy
│   ├── batch.py             # BatchGenerator for concurrent generation
│   └── response.py          # GenerationResponse dataclass
│
├── export/                  # Model export and conversion
│   ├── __init__.py
│   ├── convert.py           # HuggingFace → MLXForge conversion
│   ├── fuse.py              # Adapter merging
│   ├── quantize.py          # Post-training quantization (AWQ, GPTQ, etc.)
│   ├── gguf.py              # GGUF export
│   └── hub.py               # HuggingFace Hub upload
│
├── serve/                   # Serving layer
│   ├── __init__.py
│   ├── server.py            # FastAPI server (OpenAI-compatible)
│   ├── routes.py            # API route definitions
│   ├── middleware.py         # Auth, rate limiting, CORS
│   └── cache.py             # Prompt cache management (LRU)
│
├── logging/                 # Observability
│   ├── __init__.py
│   ├── metrics.py           # MetricsCollector: structured metrics
│   ├── logger.py            # Run logger (JSON lines + console)
│   ├── integrations.py      # WandB, MLflow, TensorBoard adapters
│   └── profiling.py         # Memory profiler, throughput tracker
│
├── runtime/                 # Daemon and job management
│   ├── __init__.py
│   ├── daemon.py            # Background daemon process
│   ├── job.py               # Job dataclass and lifecycle
│   ├── queue.py             # Job queue (SQLite-backed)
│   ├── store.py             # Artifact store (run registry)
│   └── monitor.py           # Resource monitoring (GPU, memory, temperature)
│
└── cli/                     # CLI layer (thin wrapper)
    ├── __init__.py
    ├── main.py              # Click/Typer CLI with subcommands
    ├── train.py             # mlxforge train --recipe sft --config train.yaml
    ├── generate.py          # mlxforge generate --model ... --prompt ...
    ├── serve.py             # mlxforge serve --model ...
    ├── export.py            # mlxforge export --format gguf ...
    └── run.py               # mlxforge run --daemon (starts background daemon)
```

### Key Class Interfaces

```python
# ── core/protocols.py ──

@runtime_checkable
class ModelProtocol(Protocol):
    """What every model must implement."""
    args: Any
    @property
    def layers(self) -> list: ...
    def __call__(self, inputs: mx.array, cache=None, **kwargs) -> mx.array: ...
    def make_cache(self) -> list: ...
    def sanitize(self, weights: dict) -> dict: ...


class AdapterProtocol(Protocol):
    """What every adapter type must implement."""
    @staticmethod
    def from_base(layer: nn.Module, **config) -> nn.Module: ...
    def fuse(self, dequantize: bool = False) -> nn.Module: ...


class RecipeProtocol(Protocol):
    """What every training recipe must implement."""
    def loss_fn(self, model, batch) -> Tuple[mx.array, mx.array]: ...
    def create_dataset(self, data, tokenizer, config) -> Dataset: ...
    def default_config(self) -> dict: ...
```

```python
# ── trainer/trainer.py ──

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        recipe: RecipeProtocol,
        train_dataset: Dataset,
        val_dataset: Dataset,
        callbacks: List[Callback] = None,
    ):
        self.state = TrainState(
            model=model,
            optimizer=self._build_optimizer(config),
            scheduler=self._build_scheduler(config),
            step=0,
            epoch=0,
            rng_state=mx.random.state,
        )
        self.callbacks = CallbackList(callbacks or [])

    def fit(self) -> TrainResult:
        self.callbacks.on_train_begin(self.state)
        for epoch in range(self.config.num_epochs):
            self.callbacks.on_epoch_begin(self.state)
            for batch in self.data_loader:
                self.callbacks.on_step_begin(self.state)
                metrics = self._training_step(batch)
                self.callbacks.on_step_end(self.state, metrics)
                if self._should_evaluate():
                    eval_metrics = self.evaluate()
                    self.callbacks.on_eval_end(self.state, eval_metrics)
                if self._should_save():
                    self.checkpoint_manager.save(self.state)
                    self.callbacks.on_save(self.state)
            self.callbacks.on_epoch_end(self.state)
        self.callbacks.on_train_end(self.state)
        return self.state.to_result()
```

```python
# ── trainer/state.py ──

@dataclass
class TrainState:
    model: nn.Module
    optimizer: Any
    scheduler: Any
    step: int
    epoch: int
    rng_state: mx.array
    best_val_loss: float = float('inf')
    trained_tokens: int = 0

    def save(self, path: Path):
        """Save complete state for exact resume."""
        mx.savez(path / "state.npz",
            optimizer_state=self.optimizer.state,
            rng_state=self.rng_state,
            step=self.step,
            epoch=self.epoch,
            best_val_loss=self.best_val_loss,
            trained_tokens=self.trained_tokens,
        )
        adapter_weights = dict(tree_flatten(self.model.trainable_parameters()))
        mx.save_safetensors(path / "adapters.safetensors", adapter_weights)

    @classmethod
    def load(cls, path: Path, model, optimizer, scheduler):
        """Load complete state for exact resume."""
        ...
```

```python
# ── trainer/callbacks.py ──

class Callback:
    """Lifecycle hooks for training customization."""
    def on_train_begin(self, state: TrainState): pass
    def on_train_end(self, state: TrainState): pass
    def on_epoch_begin(self, state: TrainState): pass
    def on_epoch_end(self, state: TrainState): pass
    def on_step_begin(self, state: TrainState): pass
    def on_step_end(self, state: TrainState, metrics: dict): pass
    def on_eval_begin(self, state: TrainState): pass
    def on_eval_end(self, state: TrainState, metrics: dict): pass
    def on_save(self, state: TrainState): pass

class EarlyStoppingCallback(Callback): ...
class MetricsLoggerCallback(Callback): ...
class WandBCallback(Callback): ...
class MemoryMonitorCallback(Callback): ...
class GradientClippingCallback(Callback): ...
```

```python
# ── recipes/sft.py ──

class SFTRecipe(RecipeProtocol):
    """Supervised Fine-Tuning recipe."""

    def loss_fn(self, model, batch, lengths):
        inputs = batch[:, :-1]
        targets = batch[:, 1:]
        logits = model(inputs)
        mask = self._compute_mask(lengths, targets.shape[1])
        ce = nn.losses.cross_entropy(logits, targets) * mask
        ntoks = mask.sum()
        return ce.sum() / ntoks, ntoks

    def create_dataset(self, data, tokenizer, config):
        return create_dataset(data, tokenizer, config)  # auto-detect format

    def default_config(self):
        return {"learning_rate": 1e-5, "batch_size": 4, "max_seq_length": 2048}
```

---

## SECTION 11 — Mac Studio / Daemon Backend Design

> **SUPERSEDED**: Daemon, job queue, SQLite, WebSocket, REST API, and GUI backend are all explicitly out of scope for v0 (see CLAUDE.md §12 and V0_DESIGN_FREEZE.md Out of Scope). Do not implement any component from this section.

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│                  Mac GUI (SwiftUI)               │
│  ┌───────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ Job Panel │  │  Models  │  │ Live Metrics │  │
│  │ (queue,   │  │  Browser │  │ (loss, GPU,  │  │
│  │  status)  │  │          │  │  memory, tps)│  │
│  └─────┬─────┘  └────┬─────┘  └──────┬───────┘  │
│        │              │               │          │
│        └──────────────┼───────────────┘          │
│                       │                          │
│                   WebSocket                      │
│                   + REST API                     │
└───────────────────────┼──────────────────────────┘
                        │
┌───────────────────────┼──────────────────────────┐
│              MLXForge Daemon                      │
│                       │                          │
│  ┌────────────────────┼────────────────────────┐ │
│  │            API Server (FastAPI)              │ │
│  │  REST:  /jobs, /models, /runs, /metrics     │ │
│  │  WS:    /ws/metrics, /ws/logs               │ │
│  └────────────────────┼────────────────────────┘ │
│                       │                          │
│  ┌──────────┐  ┌──────┴───────┐  ┌────────────┐ │
│  │ Job Queue│  │  Run Engine  │  │  Artifact   │ │
│  │ (SQLite) │  │  (subprocess │  │  Store      │ │
│  │          │  │   per job)   │  │  (local fs) │ │
│  └──────────┘  └──────────────┘  └────────────┘ │
│                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Resource │  │  Model Cache │  │  Run        │ │
│  │ Monitor  │  │  Manager     │  │  Registry   │ │
│  │ (IOKit)  │  │  (HF cache)  │  │  (SQLite)  │ │
│  └──────────┘  └──────────────┘  └────────────┘ │
└──────────────────────────────────────────────────┘
```

### Daemon Components

**1. Job Queue (SQLite)**
```sql
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL,           -- 'train', 'eval', 'export', 'serve'
    status TEXT DEFAULT 'pending', -- pending, running, paused, completed, failed, cancelled
    config JSON NOT NULL,
    created_at TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    pid INTEGER,                  -- OS process ID when running
    error TEXT
);
```

**2. Run Registry (SQLite)**
```sql
CREATE TABLE runs (
    id TEXT PRIMARY KEY,
    job_id TEXT REFERENCES jobs(id),
    model TEXT NOT NULL,
    recipe TEXT NOT NULL,
    config JSON NOT NULL,
    metrics JSON,                 -- final metrics
    artifact_path TEXT,           -- path to saved artifacts
    created_at TIMESTAMP,
    duration_seconds REAL
);

CREATE TABLE metrics (
    run_id TEXT REFERENCES runs(id),
    step INTEGER,
    timestamp TIMESTAMP,
    data JSON                     -- {train_loss, val_loss, lr, tps, memory, ...}
);
```

**3. Artifact Store**
```
~/.mlxforge/
├── runs/
│   ├── run_20260201_143022_abc123/
│   │   ├── manifest.json         # config + environment + data fingerprint
│   │   ├── checkpoints/
│   │   │   ├── step_000100/
│   │   │   │   ├── adapters.safetensors
│   │   │   │   └── state.npz     # full TrainState
│   │   │   └── step_000200/
│   │   ├── logs/
│   │   │   ├── train.jsonl       # structured training log
│   │   │   └── eval.jsonl
│   │   └── artifacts/
│   │       └── fused_model/      # exported model
│   └── ...
├── models/                       # local model cache (symlinks to HF cache)
└── daemon.db                     # SQLite database
```

### API Endpoints

```
REST:
  POST   /api/v1/jobs                    Create job (train/eval/export/serve)
  GET    /api/v1/jobs                    List jobs (with status filter)
  GET    /api/v1/jobs/{id}               Get job detail
  POST   /api/v1/jobs/{id}/pause         Pause running job
  POST   /api/v1/jobs/{id}/resume        Resume paused job
  POST   /api/v1/jobs/{id}/cancel        Cancel job
  DELETE /api/v1/jobs/{id}               Delete job and artifacts

  GET    /api/v1/runs                    List runs (with filters)
  GET    /api/v1/runs/{id}               Get run detail + metrics
  GET    /api/v1/runs/{id}/metrics       Get time-series metrics
  GET    /api/v1/runs/{id}/artifacts     List artifacts
  GET    /api/v1/runs/{id}/logs          Get structured logs

  GET    /api/v1/models                  List available models (local + HF)
  POST   /api/v1/models/download         Download model from Hub
  DELETE /api/v1/models/{id}             Delete local model

  GET    /api/v1/system/status           GPU, memory, temperature, running jobs
  GET    /api/v1/system/health           Health check

WebSocket:
  /ws/v1/jobs/{id}/metrics               Real-time metrics stream
  /ws/v1/jobs/{id}/logs                  Real-time log stream
  /ws/v1/system/monitor                  System resource monitoring
```

### Protocol Between UI and Backend

```python
# WebSocket message format
{
    "type": "metric",           # metric | log | status | error
    "job_id": "abc123",
    "timestamp": "2026-02-01T14:30:22Z",
    "data": {
        "step": 100,
        "train_loss": 2.345,
        "learning_rate": 1e-5,
        "tokens_per_second": 15234.5,
        "peak_memory_gb": 12.3,
        "gpu_utilization": 0.89
    }
}
```

### Pause/Resume Implementation

**Pause**: Send SIGTSTP to training subprocess. Save checkpoint immediately. Update job status.
**Resume**: Reload checkpoint into new subprocess. Continue from saved step. Update job status.
**Cancel**: Send SIGTERM. Save final checkpoint. Mark job as cancelled.

---

## SECTION 12 — Minimal v0 Scope

> **SUPERSEDED**: This section's v0.1 scope differs from the actual v0 scope. It includes DoRA, sequence packing, and dynamic batching — all of which are out of scope for v0. The authoritative v0 scope is defined in V0_DESIGN_FREEZE.md §1 and CLAUDE.md §3. The roadmap items (v0.2–v0.5) are aspirational only and not commitments.

### v0.1: Core Training
- [ ] SFT recipe with LoRA/DoRA
- [ ] Typed config system (Pydantic)
- [ ] Trainer class with callback system
- [ ] Full-state checkpointing (model + optimizer + scheduler + RNG + step)
- [ ] Basic data pipeline (chat + completions + text formats)
- [ ] Sequence packing
- [ ] Dynamic batching by total tokens
- [ ] Structured logging (JSON lines)
- [ ] `mlxforge train` CLI command
- [ ] Config validation at startup

### v0.2: Inference + Export
- [ ] Generation engine (single + batch + streaming)
- [ ] Sampling strategies (top-p, top-k, min-p, temperature)
- [ ] Adapter fusing
- [ ] Model conversion (HF → MLXForge)
- [ ] GGUF export
- [ ] `mlxforge generate` and `mlxforge export` CLI commands

### v0.3: Serving + Run Management
- [ ] FastAPI OpenAI-compatible server
- [ ] Prompt KV caching (LRU)
- [ ] SQLite run registry
- [ ] Artifact store with manifest files
- [ ] `mlxforge serve` CLI command
- [ ] Basic reproducibility (config snapshot, data fingerprint)

### v0.4: DPO + Advanced Training
- [ ] DPO recipe
- [ ] Full parameter fine-tuning recipe
- [ ] Gradient accumulation
- [ ] Gradient checkpointing
- [ ] Distributed data parallel
- [ ] WandB/MLflow integration callbacks
- [ ] Memory profiling callback

### v0.5: Daemon + GUI Backend
- [ ] Background daemon process
- [ ] Job queue (SQLite)
- [ ] WebSocket streaming for metrics/logs
- [ ] REST API for job management
- [ ] Resource monitoring (IOKit)
- [ ] Pause/resume/cancel

### Out of Scope for v0
- Mac GUI (SwiftUI) — requires v0.5 daemon backend first
- Vision-language model training
- RLHF/PPO training
- Tensor parallelism for training
- Custom Metal kernels
- Model merging (beyond LoRA fusing)
- Online/continual learning
- Multi-node distributed training

### Milestone Dependencies

```
v0.1 (Core Training)
  │
  ├── v0.2 (Inference + Export)
  │     │
  │     └── v0.3 (Serving + Run Mgmt)
  │           │
  │           └── v0.5 (Daemon + GUI Backend)
  │                 │
  │                 └── v1.0 (Mac GUI)
  │
  └── v0.4 (DPO + Advanced Training)
```

---

## Appendix: Architectural Diagrams

### Training Data Flow

```
Raw Data (JSONL/HF)
    │
    ▼
┌─────────────────┐
│ Format Detection │  Auto-detect: chat / completions / text / preference
│ + Validation     │  Validate schemas, count samples, check for issues
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Preprocessing    │  Tokenize, apply chat template, compute prompt masks
│ + Caching        │  Cache to disk (Arrow format) for reuse
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Packing          │  Concat short sequences (up to max_seq_length)
│ + Bucketing      │  Group by similar length to minimize padding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Dynamic Batching │  Create batches by total token budget
│                  │  Distribute across workers (if distributed)
└────────┬────────┘
         │
         ▼
    (batch, lengths) → Trainer
```

### Training Loop Architecture

```
                    ┌─────────────────────────────┐
                    │           Trainer             │
                    │                               │
                    │  ┌────────┐  ┌────────────┐  │
Callbacks ─────────▶│  │ Recipe │  │   State    │  │
  on_step_begin     │  │ (loss) │  │ (model,    │  │
  on_step_end       │  │        │  │  opt, lr,  │  │
  on_eval           │  └───┬────┘  │  rng, step)│  │
  on_save           │      │       └─────┬──────┘  │
  on_train_end      │      │             │         │
                    │      ▼             ▼         │
                    │  ┌─────────────────────────┐ │
                    │  │  @mx.compile             │ │
                    │  │  def step(batch):        │ │
                    │  │    loss, grad = vg(model) │ │
                    │  │    optimizer.update(grad) │ │
                    │  └─────────────────────────┘ │
                    │              │                │
                    │              ▼                │
                    │  ┌─────────────────────────┐ │
                    │  │  CheckpointManager       │ │
                    │  │  save(state) / load()    │ │
                    │  │  keep_last_n=3           │ │
                    │  │  save_best=True          │ │
                    │  └─────────────────────────┘ │
                    └─────────────────────────────┘
```

### Generation Pipeline

```
Prompt (str)
    │
    ▼
┌──────────────┐
│  Tokenizer   │  encode + chat template
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌──────────────┐
│   Prefill    │────▶│   KV Cache   │  Chunked processing
│  (prompt)    │     │  (per layer) │  with mx.eval per chunk
└──────┬───────┘     └──────────────┘
       │
       ▼
┌──────────────┐
│  Generation  │  Async pipeline:
│    Loop      │  compute next_token while yielding current
│              │  mx.clear_cache every 256 tokens
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Sampler     │  top-p → min-p → XTC → top-k → categorical
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Detokenizer  │  Streaming: yields text segments as tokens arrive
└──────┬───────┘
       │
       ▼
GenerationResponse(text, token, logprobs, tps, memory, finish_reason)
```

---

*Document generated from source-level analysis of mlx-lm at commit HEAD.*
*All file references are to `/Users/jiekaiwang/workspace/mlx-lm/mlx-lm/mlx_lm/`.*
