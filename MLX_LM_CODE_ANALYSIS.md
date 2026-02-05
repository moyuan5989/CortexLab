# mlx-lm Code Analysis & Implementation Guide

## Repository Location
`~/workspace/mlx-lm/mlx-lm/mlx_lm/models/`

## Overview

**Total model files**: 112 Python files
**Our implementations**: 2 (llama.py: 237 lines, qwen3.py: 227 lines)

---

## Key Model Implementations

### Line Count Comparison

| Model | mlx-lm | LMForge | Status |
|-------|--------|---------|--------|
| Llama | 274 lines | 237 lines | ✅ Have |
| Qwen3 | 181 lines | 227 lines | ✅ Have |
| Phi-3 | 213 lines | N/A | ❌ Need |
| Gemma2 | 205 lines | N/A | ❌ Need |

Our implementations are similar size to mlx-lm's!

---

## Critical Models to Add

### 1. Phi-3 (213 lines)

**File**: `phi3.py`

**Key Features**:
```python
@dataclass
class ModelArgs:
    partial_rotary_factor: float = 1.0  # Partial RoPE
    rope_scaling: Optional[Dict] = None  # SuScaled or LongRoPE
    max_position_embeddings: int = 131072  # Long context
```

**Differences from Llama**:
1. **Partial RoPE**: Only apply RoPE to part of the head dimension
   ```python
   rope_dim = int(head_dim * args.partial_rotary_factor)  # e.g., 0.5
   ```

2. **SuScaled RoPE**: Special RoPE scaling for long context
   ```python
   if args.rope_scaling["type"] == "su":
       self.rope = SuScaledRoPE(...)
   ```

3. **Combined QKV projection**: Single projection instead of separate Q/K/V
   ```python
   op_size = n_heads * head_dim + 2 * (n_kv_heads * head_dim)
   self.qkv_proj = nn.Linear(dim, op_size, bias=False)
   ```

**Implementation Complexity**: ⭐ Easy
- Mostly same as Llama
- SuScaledRoPE already in mlx-lm (we can port)
- Combined QKV is just a reshape difference

---

### 2. Gemma2 (205 lines)

**File**: `gemma2.py`

**Key Features**:
```python
@dataclass
class ModelArgs:
    query_pre_attn_scalar: float = 144.0  # Scale queries before attention
    attn_logit_softcapping: float = 50.0  # Cap attention logits
    final_logit_softcapping: float = 30.0  # Cap final logits
```

**Differences from Llama**:
1. **Query scaling**: Different attention scaling formula
   ```python
   self.scale = 1.0 / (args.query_pre_attn_scalar ** 0.5)
   queries = queries * self.scale
   ```

2. **Logit soft-capping**: Prevents extreme attention scores
   ```python
   scores = mx.tanh(scores / self.attn_logit_softcapping)
   scores *= self.attn_logit_softcapping
   ```

3. **GeGLU activation**: GELU variant instead of SwiGLU
   ```python
   def __call__(self, x):
       return self.down_proj(nn.gelu_approx(self.gate_proj(x)) * self.up_proj(x))
   ```

4. **RMSNorm with offset**: Slightly different normalization
   ```python
   return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
   ```

**Implementation Complexity**: ⭐⭐ Medium
- Need logit capping (new concept for us)
- GeGLU instead of SwiGLU (simple)
- Query scaling is straightforward

---

## Shared Components

### Base Classes

**File**: `base.py` (they have, we also have similar in `_base/`)

```python
@dataclass
class BaseModelArgs:
    """Base arguments for model configurations."""
    @classmethod
    def from_dict(cls, params):
        return cls(**{k: v for k, v in params.items()
                     if k in inspect.signature(cls).parameters})
```

We already have this! ✅

### Utilities We Can Reuse

1. **RoPE variants** (`rope_utils.py`):
   - `SuScaledRoPE` - Needed for Phi-3
   - `Llama3RoPE` - Already have
   - `YarnRoPE` - For long context

2. **Activations** (`activations.py`):
   - `swiglu` - Already have
   - `gelu_approx` - mlx.nn has this built-in

3. **Attention utilities** (`base.py`):
   - `create_attention_mask` - Already have
   - `scaled_dot_product_attention` - Already have

---

## Implementation Roadmap

### Phase 1: Phi-3 (2-3 hours)

**What to implement**:
1. Copy `SuScaledRoPE` from mlx-lm's `rope_utils.py`
   - Add to `lmforge/models/_base/rope.py`

2. Create `lmforge/models/architectures/phi3.py`
   - Copy structure from our `llama.py`
   - Add `partial_rotary_factor` to ModelArgs
   - Implement combined QKV projection
   - Use SuScaledRoPE for rope_scaling

3. Add to registry:
   ```python
   SUPPORTED_ARCHITECTURES["phi3"] = "lmforge.models.architectures.phi3"
   ```

**Test with**:
- microsoft/Phi-3-mini-4k-instruct

**Estimated lines**: ~220 (similar to mlx-lm)

---

### Phase 2: Gemma2 (3-4 hours)

**What to implement**:
1. Create `lmforge/models/architectures/gemma2.py`
   - Copy structure from our `llama.py`
   - Add `query_pre_attn_scalar`, `attn_logit_softcapping`
   - Implement logit capping in attention
   - Use GeGLU (gelu_approx) in MLP
   - Custom RMSNorm with 1.0 offset

2. Add to registry:
   ```python
   SUPPORTED_ARCHITECTURES["gemma2"] = "lmforge.models.architectures.gemma2"
   ```

**Test with**:
- google/gemma-2-2b
- google/gemma-2-9b

**Estimated lines**: ~210 (similar to mlx-lm)

---

## Code Reuse from mlx-lm

### What to Port (MIT Licensed)

1. **`SuScaledRoPE`** from `rope_utils.py` (~50 lines)
   - Needed for Phi-3 long context
   - Port to `lmforge/models/_base/rope.py`

2. **Logit capping pattern** from `gemma2.py`
   - Simple tanh-based capping
   - ~3 lines of code

### What NOT to Port

1. **KV caching** - We don't need for training
2. **Generation code** - Out of scope
3. **Quantization** - Training uses full precision
4. **Vision models** - Text-only for now

---

## Key Insights from mlx-lm

### 1. Architecture Similarities

Looking at their code, ~80% of LLM architectures share:
- Transformer blocks (attention + MLP)
- RMSNorm or LayerNorm
- RoPE embeddings
- SwiGLU or GeGLU activations

**Only differences** are:
- Config values (hidden_size, num_layers, etc.)
- Minor variations (combined QKV, logit capping, etc.)

### 2. Code Organization

Their approach (we already follow):
```
models/
├── base.py           # Shared utilities
├── rope_utils.py     # RoPE variants
├── activations.py    # Activation functions
├── llama.py          # Llama architecture
├── phi3.py           # Phi-3 architecture
└── gemma2.py         # Gemma2 architecture
```

Our approach (very similar):
```
models/
├── _base/
│   ├── args.py       # BaseModelArgs
│   ├── attention.py  # Attention utilities
│   ├── rope.py       # RoPE variants
│   └── activations.py # Activation functions
└── architectures/
    ├── llama.py
    ├── qwen3.py
    ├── phi3.py       # To add
    └── gemma2.py     # To add
```

We're on the right track! ✅

### 3. Implementation Quality

**mlx-lm code quality**:
- ✅ Clean, readable
- ✅ Well-documented
- ⚠️ Inference-focused (has cache, generation)
- ⚠️ Some models are experimental/untested

**Our approach**:
- ✅ Clean, readable
- ✅ Training-focused (no inference code)
- ✅ Production quality (tested end-to-end)
- ✅ Simpler (no generation, caching, etc.)

---

## Comparison: mlx-lm vs LMForge

| Aspect | mlx-lm | LMForge |
|--------|--------|---------|
| **Architectures** | 112 files | 2 files (target: 8-12) |
| **Focus** | Inference + generation | Training only |
| **Quality** | Demo/example | Production |
| **Code size** | ~20,000 lines | ~1,500 lines |
| **Complexity** | High (KV cache, generation) | Low (training only) |
| **Use case** | Run/serve models | Fine-tune models |

---

## Implementation Priority

### Must Have (v1.0)
1. **Phi-3** (~220 lines, 2-3 hours)
   - Port SuScaledRoPE
   - Implement combined QKV
   - Test with Phi-3-mini

2. **Gemma2** (~210 lines, 3-4 hours)
   - Implement logit capping
   - GeGLU activation
   - Custom RMSNorm

**Total**: ~6 hours, adds 2 major architectures

### Nice to Have (v1.1)
3. **Cohere** (~200 lines)
4. **InternLM** (~250 lines)

### Future (v2.0)
5. **MoE infrastructure** (from DeepSeek V3, Mixtral)

---

## Next Steps

1. **Review mlx-lm's MIT license** ✅ (already checked)
2. **Port SuScaledRoPE** to our `_base/rope.py`
3. **Implement Phi-3** following mlx-lm's pattern
4. **Test with microsoft/Phi-3-mini-4k-instruct**
5. **Implement Gemma2** with logit capping
6. **Test with google/gemma-2-2b**

**Time estimate**: 6-8 hours total for both models
