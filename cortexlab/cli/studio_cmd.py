"""CLI handler for `cortexlab studio` command."""

from __future__ import annotations


def run_studio(args):
    """Start the CortexLab Studio server."""
    try:
        from cortexlab.studio.server import run_server
    except ImportError as e:
        print(
            "CortexLab Studio requires additional dependencies.\n"
            "Install them with: pip install cortexlab[studio]\n"
            f"\nMissing: {e}"
        )
        raise SystemExit(1)

    run_server(
        host=args.host,
        port=args.port,
    )
