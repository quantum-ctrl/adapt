"""
Shared environment-backed configuration for ADAPT.
"""

import os
from pathlib import Path


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def get_host(default: str = "127.0.0.1") -> str:
    """Host used by ADAPT Edit."""
    return os.environ.get("ADAPT_HOST", default)


def get_port(default: int = 8000) -> int:
    """Port used by ADAPT Edit."""
    return _get_int("ADAPT_PORT", default)


def get_data_dir(default_base: str) -> str:
    """Data directory used by ADAPT Edit file listing APIs."""
    default_dir = Path(default_base) / "data"
    return os.path.abspath(os.environ.get("ADAPT_DATA_DIR", str(default_dir)))


def get_max_upload_size(default: int = 2_000_000_000) -> int:
    """Maximum upload size in bytes."""
    return _get_int("ADAPT_MAX_UPLOAD_SIZE", default)
