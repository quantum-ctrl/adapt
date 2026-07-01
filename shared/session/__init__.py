"""
Session package - Cross-application session management for ADAPT.
"""

from .session_manager import (
    write_session,
    read_session,
    clear_session,
    get_session_path,
    validate_session,
)
from .app_state import (
    get_recent_folders,
    add_recent_folder,
    get_collections,
    create_collection,
    delete_collection,
    add_file_to_collection,
    remove_file_from_collection,
)

__all__ = [
    "write_session",
    "read_session",
    "clear_session",
    "get_session_path",
    "validate_session",
    "get_recent_folders",
    "add_recent_folder",
    "get_collections",
    "create_collection",
    "delete_collection",
    "add_file_to_collection",
    "remove_file_from_collection",
]
