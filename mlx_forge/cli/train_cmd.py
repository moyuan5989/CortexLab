"""Handler for 'mlx_forge train' CLI command."""

from __future__ import annotations


def run_train(args) -> None:
    """Execute the train command from parsed CLI args."""
    from mlx_forge import train

    result = train(config=args.config, resume=args.resume)
    print(f"Training complete. Final step: {result.step}")
