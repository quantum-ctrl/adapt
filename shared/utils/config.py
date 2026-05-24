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


def _get_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


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


def allow_filesystem_browse(default: bool = False) -> bool:
    """Whether ADAPT Edit may browse outside configured browse roots."""
    return _get_bool("ADAPT_ALLOW_FILESYSTEM_BROWSE", default)


def get_browse_roots(data_dir: str) -> list:
    """
    Directories allowed for ADAPT Edit browsing.

    ADAPT_BROWSE_ROOTS may contain multiple directories separated by the OS
    path separator. By default browsing is limited to the data dir and home.
    """
    raw = os.environ.get("ADAPT_BROWSE_ROOTS")
    if raw:
        roots = [p for p in raw.split(os.pathsep) if p]
    else:
        roots = [data_dir, str(Path.home())]
    return [os.path.abspath(os.path.expanduser(root)) for root in roots]
