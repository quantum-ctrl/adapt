"""
Metadata Index Worker - Background loading of listing metadata (Type, hv, Temp).

Structurally mirrors ThumbnailWorker in ADAPT_browser/ui/file_list_panel.py:
no loader currently exposes a header-only read, so this pays the same full-load
cost the thumbnail worker already pays, and is cached in memory for the session.
"""

from PySide6.QtCore import QObject, QRunnable, Signal

from ADAPT_browser.core.data_manager import DataManager
from ADAPT_browser.utils.meta_format import summarize_for_listing


class MetadataIndexSignals(QObject):
    """Signals for MetadataIndexWorker - must be defined outside QRunnable."""
    finished = Signal(str, dict)  # filepath, summary dict
    error = Signal(str, str)      # filepath, error message


class MetadataIndexWorker(QRunnable):
    """Worker that loads a file and extracts listing metadata in a background thread."""

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
        self.signals = MetadataIndexSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            result = DataManager.load_file_sync(self.filepath)
            summary = summarize_for_listing(result.meta)
            self.signals.finished.emit(self.filepath, summary)
        except Exception as e:
            self.signals.error.emit(self.filepath, str(e))
