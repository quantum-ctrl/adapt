"""
Utils package - Common utilities for ADAPT applications.
"""

from .path_utils import (
    get_adapt_config_dir,
    ensure_adapt_config_dir,
    normalize_path,
    is_valid_data_file,
    get_file_basename,
    get_file_extension,
)

from .constants import (
    K_FACTOR,
    DEFAULT_V0,
    DEFAULT_WORK_FUNCTION,
    ELECTRON_MASS_EV,
    HC_EV_ANGSTROM,
)

from .config import (
    get_data_dir,
    get_host,
    get_max_upload_size,
    get_port,
)

__all__ = [
    # Path utilities
    "get_adapt_config_dir",
    "ensure_adapt_config_dir",
    "normalize_path",
    "is_valid_data_file",
    "get_file_basename",
    "get_file_extension",
    # Physics constants
    "K_FACTOR",
    "DEFAULT_V0",
    "DEFAULT_WORK_FUNCTION",
    "ELECTRON_MASS_EV",
    "HC_EV_ANGSTROM",
    # Configuration
    "get_data_dir",
    "get_host",
    "get_max_upload_size",
    "get_port",
]
