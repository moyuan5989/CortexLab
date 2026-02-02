"""LMForge — LoRA SFT training framework for MLX on Apple Silicon."""

from lmforge._version import __version__


def prepare(
    data_path: str,
    model: str,
    output: str | None = None,
    *,
    trust_remote_code: bool = False,
    max_seq_length: int = 2048,
    mask_prompt: bool = True,
) -> dict:
    """Pre-tokenize a dataset and write a safetensors cache to disk.

    Returns a dict of statistics (sample count, total tokens, etc.).
    """
    raise NotImplementedError("prepare() will be implemented in M2.")


def train(config) -> "lmforge.trainer.state.TrainState":
    """Run LoRA SFT training from a config file or TrainingConfig object.

    Args:
        config: Path to a YAML config file (str) or a TrainingConfig instance.

    Returns:
        Final TrainState after training completes.
    """
    raise NotImplementedError("train() will be implemented in M5.")
