"""
App State Manager - Persists cross-folder browser preferences.

Stores recently opened folders and named file collections in a single JSON
file next to the existing session.json, so both survive across app restarts
without depending on any specific data folder being writable.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List

from .session_manager import _get_session_dir

logger = logging.getLogger(__name__)

STATE_FILE = _get_session_dir() / "app_state.json"

_DEFAULT_STATE = {
    "version": "1.0",
    "recent_folders": [],
    "collections": {},
}


def _load_state() -> dict:
    if not STATE_FILE.exists():
        return dict(_DEFAULT_STATE)

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"Failed to read app state file: {e}")
        return dict(_DEFAULT_STATE)

    state.setdefault("version", "1.0")
    state.setdefault("recent_folders", [])
    state.setdefault("collections", {})
    return state


def _save_state(state: dict) -> bool:
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        return True
    except OSError as e:
        logger.error(f"Failed to write app state file: {e}")
        return False
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize app state: {e}")
        return False


def get_recent_folders() -> List[str]:
    """Get recently opened folders, most recent first."""
    return list(_load_state()["recent_folders"])


def add_recent_folder(path: str, max_items: int = 15) -> bool:
    """Add a folder to the front of the recent-folders list (MRU, deduped)."""
    if not path:
        return False

    path = os.path.abspath(path)
    state = _load_state()
    recent = [p for p in state["recent_folders"] if p != path]
    recent.insert(0, path)
    state["recent_folders"] = recent[:max_items]
    return _save_state(state)


def get_collections() -> Dict[str, dict]:
    """Get all collections as {name: {"created": iso, "files": [paths]}}."""
    return _load_state()["collections"]


def create_collection(name: str) -> bool:
    """Create a new empty collection. Returns False if it already exists."""
    if not name:
        return False

    state = _load_state()
    if name in state["collections"]:
        return False

    state["collections"][name] = {
        "created": datetime.now().isoformat(),
        "files": [],
    }
    return _save_state(state)


def delete_collection(name: str) -> bool:
    """Delete a collection. Returns False if it doesn't exist."""
    state = _load_state()
    if name not in state["collections"]:
        return False

    del state["collections"][name]
    return _save_state(state)


def add_file_to_collection(name: str, filepath: str) -> bool:
    """Add a file to an existing collection. Returns False if the collection is missing."""
    if not filepath:
        return False

    state = _load_state()
    collection = state["collections"].get(name)
    if collection is None:
        return False

    filepath = os.path.abspath(filepath)
    if filepath not in collection["files"]:
        collection["files"].append(filepath)
    return _save_state(state)


def remove_file_from_collection(name: str, filepath: str) -> bool:
    """Remove a file from a collection. Returns False if the collection is missing."""
    state = _load_state()
    collection = state["collections"].get(name)
    if collection is None:
        return False

    filepath = os.path.abspath(filepath)
    if filepath in collection["files"]:
        collection["files"].remove(filepath)
    return _save_state(state)
