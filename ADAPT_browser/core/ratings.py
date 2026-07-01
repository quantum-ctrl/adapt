"""
Ratings - Per-folder star ratings (0-5) and reject flags.

Stored as a JSON sidecar file next to the data files themselves, so ratings
travel with the data folder (shared drives, copied folders, etc.) rather
than living in a central app-wide store.
"""

import json
import logging
import os
from typing import Dict

logger = logging.getLogger(__name__)

SIDECAR_NAME = ".adapt_ratings.json"

_DEFAULT_ENTRY = {"rating": 0, "rejected": False}


def _sidecar_path(folder: str) -> str:
    return os.path.join(folder, SIDECAR_NAME)


def load_folder_ratings(folder: str) -> Dict[str, dict]:
    """Load {filename: {"rating": int, "rejected": bool}} for one folder."""
    path = _sidecar_path(folder)
    if not os.path.isfile(path):
        return {}

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to read ratings sidecar {path}: {e}")
        return {}


def _save_folder_ratings(folder: str, ratings: Dict[str, dict]) -> bool:
    path = _sidecar_path(folder)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ratings, f, indent=2, ensure_ascii=False)
        return True
    except OSError as e:
        logger.warning(f"Failed to write ratings sidecar {path} (folder may be read-only): {e}")
        return False


def get_rating(filepath: str) -> dict:
    """Get {"rating": int, "rejected": bool} for a single file (defaults if unset)."""
    folder = os.path.dirname(filepath)
    filename = os.path.basename(filepath)
    return load_folder_ratings(folder).get(filename, dict(_DEFAULT_ENTRY))


def set_rating(filepath: str, rating: int) -> bool:
    """Set the star rating (clamped to 0-5) for a file."""
    rating = max(0, min(5, rating))
    folder = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    ratings = load_folder_ratings(folder)
    entry = dict(ratings.get(filename, _DEFAULT_ENTRY))
    entry["rating"] = rating
    ratings[filename] = entry
    return _save_folder_ratings(folder, ratings)


def set_rejected(filepath: str, rejected: bool) -> bool:
    """Set the reject flag for a file."""
    folder = os.path.dirname(filepath)
    filename = os.path.basename(filepath)

    ratings = load_folder_ratings(folder)
    entry = dict(ratings.get(filename, _DEFAULT_ENTRY))
    entry["rejected"] = rejected
    ratings[filename] = entry
    return _save_folder_ratings(folder, ratings)
