# PyTorch Hybrid Model Loading Design

## Overview

Add PyTorch-based model loading as a fallback for unsupported architectures, enabling support for 1000+ models while keeping native MLX implementations for tier-1 models.

## Goals

1. **Broad coverage**: Support any model in HuggingFace transformers
2. **Keep performance**: Use native MLX for popular models (Llama, Qwen, Phi)
3. **Transparent**: Users don't need to know which path is used
4. **Training focus**: Only load for training (no inference features)

## Non-Goals

- Inference/generation (out of scope for v0)
- PyTorch training (always use MLX for training)
- Backwards compatibility with PyTorch checkpoints

---

## Architecture

### Router Logic

```python
def load_model(model_path, tokenizer_path=None, trust_remote_code=False):
    config = load_config(model_path)
    model_type = config.get("model_type")

    # Tier 1: Native MLX (fast path)
    if model_type in NATIVE_MLX_ARCHITECTURES:
        return load_native_mlx(model_path, config)

    # Tier 2: PyTorch conversion (fallback)
    else:
        return load_from_pytorch(model_path, config)
```

### Native MLX Architectures (Tier 1)

Keep existing implementations:
- `llama` - Llama 2/3, Mistral
- `qwen3` - Qwen3 family
- `phi3` - (add if needed)

These are **optimized**, **tested**, and **fast**.

### PyTorch Conversion (Tier 2)

For everything else, convert from transformers:

```python
def load_from_pytorch(model_path, config):
    # 1. Load PyTorch model from transformers
    from transformers import AutoModelForCausalLM
    import torch

    pt_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=trust_remote_code
    )

    # 2. Convert to MLX
    mlx_model = convert_pytorch_to_mlx(pt_model, config)

    # 3. Verify and return
    return mlx_model
```

---

## Conversion Implementation

### Phase 1: Weight-Only Conversion (v1.0)

For architectures similar to Llama/Qwen (most LLMs), we can:
1. Load PyTorch weights
2. Map to our existing MLX architecture
3. Convert tensors

```python
def convert_weights_only(pt_model, mlx_model):
    """Convert weights for similar architectures."""
    state_dict = pt_model.state_dict()

    # Convert each tensor
    mlx_weights = {}
    for name, pt_tensor in state_dict.items():
        # PyTorch tensor → NumPy → MLX
        np_array = pt_tensor.cpu().numpy()
        mlx_weights[name] = mx.array(np_array)

    mlx_model.load_weights(list(mlx_weights.items()))
    return mlx_model
```

**Supported**: Models with standard transformer architecture (90% of LLMs)
**Not supported**: Non-standard architectures (MoE with unique routing, etc.)

### Phase 2: Full Architecture Conversion (v1.1)

For truly novel architectures, convert the full model structure:

```python
def convert_architecture(pt_module, name=""):
    """Recursively convert PyTorch nn.Module to MLX nn.Module."""

    # Base cases: known layer types
    if isinstance(pt_module, torch.nn.Linear):
        return convert_linear(pt_module)
    elif isinstance(pt_module, torch.nn.LayerNorm):
        return convert_layernorm(pt_module)
    elif isinstance(pt_module, torch.nn.Embedding):
        return convert_embedding(pt_module)

    # Recursive case: container modules
    elif isinstance(pt_module, torch.nn.ModuleList):
        return [convert_architecture(m, f"{name}.{i}")
                for i, m in enumerate(pt_module)]

    # Custom modules: try to reconstruct
    else:
        return convert_custom_module(pt_module, name)
```

**Supported**: Any architecture (including MoE, custom attention, etc.)
**Complexity**: High - need to handle all PyTorch module types

---

## Implementation Phases

### Phase 1: Weight Conversion (Week 1)

**Goal**: Support Llama-like models via weight mapping

**Deliverables**:
1. `lmforge/models/converters/pytorch.py` - Weight conversion
2. Updated `loader.py` - Router logic
3. Tests for Phi-3, Gemma (Llama-like architectures)

**Example**:
```python
# Load Gemma (not natively implemented)
model, tokenizer = load_model("google/gemma-2b")
# → Uses PyTorch weights + Llama MLX architecture
# → Works for training
```

### Phase 2: Architecture Conversion (Week 2-3)

**Goal**: Support truly novel architectures (MoE, etc.)

**Deliverables**:
1. `lmforge/models/converters/architecture.py` - Full conversion
2. Support for DeepSeek V3 (MoE)
3. Support for any transformers model

**Example**:
```python
# Load DeepSeek V3 (MoE, unique architecture)
model, tokenizer = load_model("deepseek-ai/DeepSeek-V3")
# → Converts full PyTorch architecture to MLX
# → Works for training
```

### Phase 3: Optimization (Week 4)

**Goal**: Cache converted models, optimize performance

**Deliverables**:
1. Cache converted models (avoid re-conversion)
2. Benchmark converted vs native
3. Documentation and examples

---

## Dependencies

### New Required Dependencies

```toml
# pyproject.toml
[project]
dependencies = [
    "mlx>=0.20.0",
    "transformers>=4.40.0",  # Already have for tokenizers
    "torch>=2.0.0",          # NEW - for model loading
    "safetensors>=0.4.0",    # Already have
    # ... existing deps
]
```

**Size impact**:
- PyTorch CPU: ~200MB
- PyTorch + transformers models: ~2GB total

### Optional: Make PyTorch Optional

```toml
[project.optional-dependencies]
pytorch_loader = ["torch>=2.0.0"]
```

Users who only use native MLX models don't need PyTorch:
```bash
# Minimal install (native MLX only)
pip install lmforge

# Full install (with PyTorch conversion)
pip install lmforge[pytorch_loader]
```

---

## API Design

### User-Facing API (No Changes)

```python
from lmforge import train

# Works for both native and converted models
train(config="train.yaml")
```

### Explicit Control (Optional)

```python
from lmforge.models import load_model

# Auto-detect (recommended)
model = load_model("Qwen/Qwen3-0.6B")  # Uses native MLX
model = load_model("google/gemma-2b")   # Uses PyTorch conversion

# Explicit control
model = load_model("Qwen/Qwen3-0.6B", loader="native")
model = load_model("google/gemma-2b", loader="pytorch")

# Force PyTorch (for testing)
model = load_model("Qwen/Qwen3-0.6B", loader="pytorch")
```

---

## Testing Strategy

### Unit Tests
- Weight conversion correctness (PyTorch → MLX)
- Tensor shape preservation
- Numerical precision (tolerance: 1e-5)

### Integration Tests
- Load Phi-3 via PyTorch, train for 10 steps
- Load Gemma via PyTorch, verify loss decreases
- Load DeepSeek V3 (when Phase 2 complete)

### Comparison Tests
- Load Qwen3 via native vs PyTorch
- Verify outputs match (numerical precision)
- Benchmark performance difference

---

## Performance Expectations

### Native MLX (Tier 1)
- Qwen3-0.6B: **1100-1240 tok/s** ✅ (current)

### PyTorch Conversion (Tier 2)
- Gemma-2B: **~1000-1100 tok/s** (estimated)
- Overhead: ~10% slower than native (conversion is one-time)

### Conversion Time
- Weight conversion: ~5-10 seconds (one-time)
- Cached after first load

---

## Documentation Updates

### New Section in CLAUDE.md

```markdown
## 17. PyTorch Hybrid Loading (v1.0)

### Tier 1: Native MLX
- Llama, Mistral, Qwen3
- Optimized implementations
- Best performance

### Tier 2: PyTorch Conversion
- All other transformers models
- Automatic support
- ~10% performance overhead

### Usage
No changes required - auto-detected based on model_type
```

### New Tutorial

```markdown
# examples/using_pytorch_models.md

How to use models not natively implemented:
1. Install PyTorch: pip install torch
2. Use any model: lmforge train --model google/gemma-2b
3. Works automatically!
```

---

## Migration Path

### v0 → v1 (No Breaking Changes)

✅ Existing configs work unchanged
✅ Native models use same code path
✅ New models "just work"

### Deprecation Strategy

Eventually, we may deprecate native implementations if PyTorch conversion is good enough:
- v1.0: Add PyTorch conversion
- v1.x: Optimize and test
- v2.0: Consider deprecating manual implementations (if conversion is perfect)

---

## Risks & Mitigations

### Risk 1: Conversion Bugs
**Mitigation**: Extensive testing, numerical comparison tests

### Risk 2: Performance Degradation
**Mitigation**: Keep native implementations for popular models

### Risk 3: PyTorch Dependency Size
**Mitigation**: Make it optional via extras_require

### Risk 4: Incompatible Architectures
**Mitigation**: Phase 1 focuses on standard transformers, Phase 2 handles edge cases

---

## Success Metrics

### Phase 1 Success
- ✅ Load and train Phi-3 (not natively implemented)
- ✅ Load and train Gemma-2B (not natively implemented)
- ✅ Performance within 10% of native

### Phase 2 Success
- ✅ Load and train DeepSeek V3 (MoE)
- ✅ Support 20+ architectures
- ✅ Zero manual implementations needed for new models

---

## Open Questions

1. **Make PyTorch required or optional?**
   - Required: Simpler, but larger install
   - Optional: Flexible, but more complex

2. **Cache converted models?**
   - Yes: Faster subsequent loads
   - Where: `~/.lmforge/cache/converted_models/`

3. **Support quantized PyTorch models?**
   - Later: v1.1+
   - Focus on full precision first

4. **Fallback if conversion fails?**
   - Clear error message
   - Link to issue tracker
   - Document which architectures are known to work
