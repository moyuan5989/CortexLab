# Next Architectures to Add

## Current Support

✅ **Llama** (llama.py, 237 lines)
- Llama 2, 3, 3.1, 3.2
- Mistral (remapped)
- ~50% of fine-tuning use cases

✅ **Qwen3** (qwen3.py, 227 lines)
- Qwen 1.5, 2, 2.5, 3
- QK-norm variant
- ~20% of fine-tuning use cases

**Coverage**: ~70% of fine-tuning workloads

---

## Prioritized Roadmap

### Tier 1: High Priority (Easy + High Value)

#### 1. Phi-3 (Microsoft)

**Why Add**:
- Very popular for small models (2B-14B)
- High quality, well-trained
- Perfect for Mac hardware (2-4B fits in memory easily)
- Growing adoption

**Implementation Effort**: ⭐ Easy (1-2 hours)
- Almost identical to Llama architecture
- Only differences: smaller dims, some config names
- Can reuse most of llama.py code

**Code Size**: ~150 lines (simpler than Llama)

**Models Supported**:
- microsoft/Phi-3-mini-4k-instruct (3.8B)
- microsoft/Phi-3-mini-128k-instruct (3.8B)
- microsoft/Phi-3-small-8k-instruct (7B)
- microsoft/Phi-3-medium-4k-instruct (14B)

**Architecture Notes**:
```python
# Phi-3 is Llama with:
# - Smaller models (2-14B vs 7-70B)
# - Different position embeddings (RoPE with different scaling)
# - Same attention, same MLP
# Can literally inherit from Llama and override config
```

---

#### 2. Gemma (Google)

**Why Add**:
- Official Google weights (high quality)
- Popular in research/production
- Good performance for size
- Gemma 2 series (2B-27B) covers wide range

**Implementation Effort**: ⭐⭐ Medium (3-4 hours)
- Similar to Llama/Qwen3
- Has QK-norm (we already implemented this in Qwen3)
- Different attention scaling
- Some config differences

**Code Size**: ~240 lines (similar to Llama)

**Models Supported**:
- google/gemma-2b
- google/gemma-7b
- google/gemma-2-9b
- google/gemma-2-27b

**Architecture Notes**:
```python
# Gemma differences from Llama:
# - QK normalization (like Qwen3) ✅ already implemented
# - Different attention scaling (query_pre_attn_scalar)
# - GeGLU activation instead of SwiGLU
# - Different RoPE scaling
```

---

### Tier 2: Medium Priority (Medium Effort)

#### 3. Yi (01.AI)

**Why Add**:
- Strong performance (competes with Llama)
- Chinese company (good for Asia market)
- Clean architecture
- 6B-34B range

**Implementation Effort**: ⭐ Easy (2-3 hours)
- Very similar to Llama
- Mainly config differences
- Uses standard components

**Code Size**: ~200 lines

**Models Supported**:
- 01-ai/Yi-6B
- 01-ai/Yi-34B

---

#### 4. Falcon (TII)

**Why Add**:
- Strong open model from UAE
- Interesting architecture variations
- 7B-180B range

**Implementation Effort**: ⭐⭐ Medium (4-5 hours)
- Different attention (alibi vs RoPE)
- Parallel attention + MLP
- Some unique features

**Code Size**: ~280 lines

**Models Supported**:
- tiiuae/falcon-7b
- tiiuae/falcon-40b

---

### Tier 3: Low Priority (High Effort)

#### 5. DeepSeek V2/V3

**Why Add**:
- Very capable (state-of-art performance)
- Cost-effective (MoE)
- Growing adoption

**Implementation Effort**: ⭐⭐⭐⭐ Hard (2-3 days)
- MoE architecture (need router, experts)
- Multi-head Latent Attention (MLA) - novel
- Auxiliary loss for load balancing
- Complex testing required

**Code Size**: ~500 lines (new MoE infrastructure)

**Models Supported**:
- deepseek-ai/DeepSeek-V2
- deepseek-ai/DeepSeek-V3

**Blocked By**: Need MoE infrastructure first

---

#### 6. Mixtral (Mistral AI)

**Why Add**:
- Popular MoE model
- Good performance

**Implementation Effort**: ⭐⭐⭐⭐ Hard (2-3 days)
- MoE architecture
- Can reuse DeepSeek MoE if implemented

**Code Size**: ~400 lines

**Models Supported**:
- mistralai/Mixtral-8x7B-v0.1
- mistralai/Mixtral-8x22B-v0.1

**Blocked By**: Need MoE infrastructure first

---

## Implementation Priority

Based on **value per effort**, I recommend this order:

### Immediate (v1.0)
1. **Phi-3** - 1-2 hours, high value, easy ⭐
2. **Gemma** - 3-4 hours, high value, medium ⭐⭐

**Result**: Cover ~85% of fine-tuning use cases

### Near-term (v1.1)
3. **Yi** - 2-3 hours, medium value, easy ⭐

**Result**: Cover ~90% of fine-tuning use cases

### Future (v2.0)
4. **MoE infrastructure** (shared) - 1 week
5. **DeepSeek V3** - Using MoE infra
6. **Mixtral** - Using MoE infra

**Result**: Cover ~95% of fine-tuning use cases

---

## Implementation Comparison

| Model | Effort | Lines | Similar To | New Concepts |
|-------|--------|-------|------------|--------------|
| Phi-3 | ⭐ Easy | 150 | Llama | None |
| Gemma | ⭐⭐ Medium | 240 | Llama + Qwen3 | GeGLU |
| Yi | ⭐ Easy | 200 | Llama | None |
| Falcon | ⭐⭐ Medium | 280 | Llama | ALiBi, Parallel |
| DeepSeek | ⭐⭐⭐⭐ Hard | 500 | Novel | MoE, MLA |
| Mixtral | ⭐⭐⭐⭐ Hard | 400 | Novel | MoE |

---

## Recommendation: Start with Phi-3

**Why Phi-3 First**:
1. **Easiest** to implement (almost copy Llama)
2. **High value** - very popular for small models
3. **Perfect for Mac** - 4B fits comfortably in 16GB RAM
4. **Fast win** - can finish in 1-2 hours
5. **Validates approach** - proves we can add models easily

**After Phi-3, Add Gemma**:
- Slightly more complex (QK-norm, GeGLU)
- But still straightforward
- Covers Google ecosystem
- Combined with Phi-3: 85% coverage

**Save MoE for v2.0**:
- Much more complex
- Need proper infrastructure
- Not urgent (most users fine-tune smaller models)

---

## Next Steps

1. **Implement Phi-3** (~2 hours)
   - Create `lmforge/models/architectures/phi3.py`
   - Copy llama.py, adjust configs
   - Add to registry
   - Test with microsoft/Phi-3-mini-4k-instruct

2. **Implement Gemma** (~4 hours)
   - Create `lmforge/models/architectures/gemma.py`
   - Use Qwen3's QK-norm code
   - Implement GeGLU activation
   - Test with google/gemma-2b

3. **Document + Examples**
   - Update examples/ with Phi-3 and Gemma configs
   - Add to supported models list

**Total time**: ~6 hours for 85% coverage

Ready to start with Phi-3?
