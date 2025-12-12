"""
Path Utilities - Common path handling for ADAPT applications.
"""

import os
from pathlib import Path
from typing import Optional


def get_adapt_config_dir() -> Path:
    """
    Get the ADAPT configuration directory.
    
    Returns:
        Path to ~/.adapt/
    """
    return Path.home() / ".adapt"


def ensure_adapt_config_dir() -> Path:
    """
    Ensure the ADAPT configuration directory exists.
    
    Returns:
        Path to ~/.adapt/
    """
    config_dir = get_adapt_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def normalize_path(path: str) -> str:
    """
    Normalize a file path to an absolute path.
    
    Args:
        path: File path (relative or absolute).
    
    Returns:
        Absolute, normalized path.
    """
    return os.path.abspath(os.path.expanduser(path))


def is_valid_data_file(path: str) -> bool:
    """
    Check if a path points to a valid ARPES data file.
    
    Args:
        path: Path to check.
    
    Returns:
        True if the file exists and has a supported extension.
    """
    if not os.path.isfile(path):
        return False
    
    # Supported extensions (matching loaders in shared/loaders/)
    supported = {".h5", ".hdf5", ".nxs", ".ibw", ".zip", ".pxt", ".pxp"}
    ext = os.path.splitext(path)[1].lower()
    return ext in supported


def get_file_basename(path: str) -> str:
    """
    Get the basename of a file path.
    
    Args:
        path: File path.
    
    Returns:
        Filename without directory path.
    """
    return os.path.basename(path)


def get_file_extension(path: str) -> str:
    """
    Get the lowercase extension of a file path.
    
    Args:
        path: File path.
    
    Returns:
        File extension including the dot (e.g., ".h5").
    """
    return os.path.splitext(path)[1].lower()
