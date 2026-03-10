"""CLI handler for `mlx_forge studio` command."""

from __future__ import annotations


def run_studio(args):
    """Start the MLX Forge Studio server."""
    try:
        from mlx_forge.studio.server import run_server
    except ImportError as e:
        print(
            "MLX Forge Studio requires additional dependencies.\n"
            "Install them with: pip install mlx_forge[studio]\n"
            f"\nMissing: {e}"
        )
        raise SystemExit(1)

    run_server(
        host=args.host,
        port=args.port,
    )
