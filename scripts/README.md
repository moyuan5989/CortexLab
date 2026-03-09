# CortexLab Scripts

Standalone scripts for working with CortexLab. These are **not part of the cortexlab package** - they're convenience tools for data preparation.

## download_hf_dataset.py

Convert HuggingFace datasets to CortexLab JSONL format.

### Installation

```bash
pip install datasets tqdm
```

### Quick Start

```bash
# Download Alpaca dataset (52K instruction samples)
python scripts/download_hf_dataset.py alpaca --output data/

# Download OpenAssistant conversations
python scripts/download_hf_dataset.py openassistant --output data/

# Download Dolly 15K instruction dataset
python scripts/download_hf_dataset.py dolly --output data/

# Test with small sample
python scripts/download_hf_dataset.py alpaca --output data/ --max-samples 100
```

### Supported Presets

| Preset | Dataset | Size | Description |
|--------|---------|------|-------------|
| `alpaca` | tatsu-lab/alpaca | 52K | Stanford Alpaca instruction-following |
| `openassistant` | OpenAssistant/oasst1 | ~90K | Multi-turn conversations |
| `dolly` | databricks/databricks-dolly-15k | 15K | Instruction-following by Databricks |

### Custom Datasets

For datasets not in presets, use `--format` to specify the conversion format:

```bash
# Custom dataset with Alpaca-style format
python scripts/download_hf_dataset.py custom \
    --dataset user/my-dataset \
    --format alpaca \
    --output data/

# Custom dataset with ShareGPT-style format
python scripts/download_hf_dataset.py custom \
    --dataset user/my-sharegpt-dataset \
    --format sharegpt \
    --output data/
```

Available formats:
- `alpaca` - {instruction, input, output}
- `dolly` - {instruction, context, response}
- `openassistant` - {messages: [{role, content}]}
- `sharegpt` - {conversations: [{from, value}]}

### Output Format

Creates two JSONL files in chat format:

```json
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Next Steps

After downloading, use with CortexLab:

```bash
# Prepare (tokenize and cache)
cortexlab prepare --data data/train.jsonl --model Qwen/Qwen3-0.6B

# Train
cortexlab train --config train.yaml
```

### Troubleshooting

**Error: "Required packages not installed"**
```bash
pip install datasets tqdm
```

**Error: "Gated dataset"**

Some datasets require accepting terms on HuggingFace:
1. Visit the dataset page on huggingface.co
2. Accept the terms
3. Set HF token: `export HF_TOKEN=your_token`

**Error: "Dataset requires subset"**

Some datasets have multiple configurations:
```bash
python scripts/download_hf_dataset.py custom \
    --dataset Helsinki-NLP/opus-100 \
    --subset en-zh \
    --format custom
```

### Examples

```bash
# Small test dataset (100 samples)
python scripts/download_hf_dataset.py alpaca --output data/test --max-samples 100

# Full Alpaca dataset for production
python scripts/download_hf_dataset.py alpaca --output data/alpaca

# OpenAssistant conversations
python scripts/download_hf_dataset.py openassistant --output data/oasst

# Custom dataset with Alpaca format
python scripts/download_hf_dataset.py custom \
    --dataset yahma/alpaca-cleaned \
    --format alpaca \
    --output data/alpaca-cleaned
```

## Adding New Converters

To support a new dataset format, add a converter function to `download_hf_dataset.py`:

```python
def convert_my_format(example: dict) -> dict | None:
    """Convert my dataset format to CortexLab chat format."""
    return {
        "messages": [
            {"role": "user", "content": example["question"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }
```

Then add to `PRESETS`:

```python
"my-dataset": {
    "dataset": "user/my-dataset-repo",
    "converter": convert_my_format,
    "train_split": "train",
    "val_split": "test",
    "val_ratio": None,
}
```
