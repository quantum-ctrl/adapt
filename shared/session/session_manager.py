"""
Session Manager - Manages session state between ADAPT Browser and ADAPT Viewer.

This module provides functionality to write and read session data stored in
~/.adapt/session.json. It enables the Browser to pass file paths and metadata
to the Viewer for seamless cross-application data transfer.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Configure module logger
logger = logging.getLogger(__name__)

# Default session file location
SESSION_DIR = Path("/Users/mengyu/Desktop/git/ADAPT/.adapt_temp")
SESSION_FILE = SESSION_DIR / "session.json"


def get_session_path() -> Path:
    """
    Get the path to the session file.
    
    Returns:
        Path to ~/.adapt/session.json
    """
    return SESSION_FILE


def write_session(file_path: str, metadata: Optional[dict] = None) -> bool:
    """
    Write session data to the session file.
    
    This function saves the current file path and metadata to ~/.adapt/session.json
    so that ADAPT Viewer can read it and load the file.
    
    Args:
        file_path: Absolute path to the ARPES data file.
        metadata: Optional dictionary containing file metadata (axes, scan type, etc.).
    
    Returns:
        True if the session was written successfully, False otherwise.
    
    Raises:
        ValueError: If file_path is empty or None.
    
    Example:
        >>> write_session("/path/to/data.h5", {"scan_type": "kx-ky-E", "hv": 21.2})
        True
    """
    if not file_path:
        raise ValueError("file_path cannot be empty or None")
    
    # Normalize the file path
    file_path = os.path.abspath(file_path)
    
    # Verify file exists
    if not os.path.exists(file_path):
        logger.warning(f"Session file path does not exist: {file_path}")
    
    # Build session data
    session_data = {
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "file_path": file_path,
        "metadata": metadata or {}
    }
    
    try:
        # Ensure session directory exists
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        
        # Write session file
        with open(SESSION_FILE, "w", encoding="utf-8") as f:
            json.dump(session_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Session written: {file_path}")
        return True
        
    except OSError as e:
        logger.error(f"Failed to write session file: {e}")
        return False
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize session data: {e}")
        return False


def read_session() -> Optional[dict]:
    """
    Read session data from the session file.
    
    This function reads the session data from ~/.adapt/session.json.
    
    Returns:
        Dictionary containing session data with keys:
        - version: Session format version
        - timestamp: When the session was created
        - file_path: Absolute path to the data file
        - metadata: Dictionary of file metadata
        
        Returns None if the session file doesn't exist or is invalid.
    
    Example:
        >>> session = read_session()
        >>> if session:
        ...     print(session["file_path"])
        /path/to/data.h5
    """
    if not SESSION_FILE.exists():
        logger.debug("Session file does not exist")
        return None
    
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            session_data = json.load(f)
        
        # Validate required fields
        if "file_path" not in session_data:
            logger.warning("Session file missing 'file_path' field")
            return None
        
        logger.info(f"Session read: {session_data.get('file_path', 'unknown')}")
        return session_data
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse session file: {e}")
        return None
    except OSError as e:
        logger.error(f"Failed to read session file: {e}")
        return None


def clear_session() -> bool:
    """
    Clear (delete) the session file.
    
    Returns:
        True if the session was cleared successfully or didn't exist,
        False if deletion failed.
    """
    try:
        if SESSION_FILE.exists():
            SESSION_FILE.unlink()
            logger.info("Session cleared")
        return True
    except OSError as e:
        logger.error(f"Failed to clear session file: {e}")
        return False


def validate_session(session_data: dict) -> bool:
    """
    Validate session data structure.
    
    Args:
        session_data: Dictionary containing session data.
    
    Returns:
        True if the session data is valid, False otherwise.
    """
    if not isinstance(session_data, dict):
        return False
    
    # Check required fields
    if "file_path" not in session_data:
        return False
    
    # Verify file path is a string
    if not isinstance(session_data["file_path"], str):
        return False
    
    # Verify file exists (warning only)
    file_path = session_data["file_path"]
    if not os.path.exists(file_path):
        logger.warning(f"Session file path does not exist: {file_path}")
    
    return True
