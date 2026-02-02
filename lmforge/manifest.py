"""Run manifest and environment info for LMForge v0."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HardwareInfo:
    chip: str
    memory_gb: int
    gpu_cores: int
    os: str


@dataclass
class RunManifest:
    schema_version: int
    config: dict
    lmforge_version: str
    mlx_version: str
    python_version: str
    hardware: HardwareInfo
    data_fingerprint: str
    created_at: str


@dataclass
class EnvironmentInfo:
    python_version: str
    mlx_version: str
    lmforge_version: str
    platform: str
    os_version: str
    chip: str
    memory_gb: int
    gpu_cores: int


def collect_environment() -> EnvironmentInfo:
    """Collect current environment information."""
    raise NotImplementedError("collect_environment() will be implemented in M5.")


def write_manifest(run_dir: Path, config: dict, data_fingerprint: str) -> RunManifest:
    """Write manifest.json and environment.json to the run directory."""
    raise NotImplementedError("write_manifest() will be implemented in M5.")
