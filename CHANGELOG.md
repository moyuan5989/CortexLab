# Changelog

All notable changes to MLX Forge will be documented in this file.

## [0.1.0] - 2026-03-05

Initial open-source release.

### Training
- LoRA and QLoRA (4-bit) fine-tuning on Apple Silicon via MLX
- DPO (Direct Preference Optimization) training
- Sequence packing for 2-5x throughput improvement
- Gradient checkpointing for 40-60% memory savings
- Compiled training loop with gradient accumulation
- Cosine, linear, step, and exponential LR schedules with warmup
- Checkpoint resume with stateless LR reconstruction

### Models
- Llama 2/3 (all sizes)
- Mistral (mapped to Llama architecture)
- Qwen 2/3/3.5
- Phi-3/4
- Gemma 1/2/3 (1B-27B)
- Automatic Hugging Face model downloading and caching

### Studio UI
- Browser-based training dashboard (React + FastAPI)
- Real-time loss curve visualization via WebSocket
- Model library and dataset browser
- Interactive playground for testing fine-tuned models

### Data
- 20+ curated datasets across 7 categories
- Auto-detection of chat, completions, text, and preference formats
- Multi-dataset mixing with weighted sampling
- Data validation CLI with train/val overlap detection
- Arrow-based storage backend for tokenized datasets

### CLI
- `mlx-forge train` — Run training from YAML config
- `mlx-forge generate` — Text generation with optional LoRA adapters
- `mlx-forge prepare` — Pre-tokenize datasets
- `mlx-forge studio` — Launch browser-based UI
- `mlx-forge data` — Dataset management (catalog, download, import, validate)
