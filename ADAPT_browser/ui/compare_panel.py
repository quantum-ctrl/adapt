"""
Compare Panel - Side-by-side static preview of up to two pinned files.

Deliberately simpler than ViewerPanel: no slice slider, no cursor readout.
For 3D+ data it shows the same "middle slice" projection ThumbnailWorker
uses for grid thumbnails, just rendered larger with real axis coordinates.
"""

import os
from typing import List, Optional

import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
from PySide6.QtCore import Qt, QObject, QRunnable, QThreadPool, Signal

from ADAPT_browser.core.data_manager import DataManager, DataResult
from ADAPT_browser.utils.meta_format import summarize_for_listing
from ADAPT_browser.utils.plotting import apply_colormap_to_image
from ADAPT_browser.utils.logger import logger

MAX_COMPARE_SLOTS = 2


class CompareLoadSignals(QObject):
    """Signals for CompareLoadWorker - must be defined outside QRunnable."""
    finished = Signal(int, str, object)  # slot, filepath, DataResult
    error = Signal(int, str, str)        # slot, filepath, error message


class CompareLoadWorker(QRunnable):
    """Loads a file for the compare panel in a background thread."""

    def __init__(self, slot: int, filepath: str):
        super().__init__()
        self.slot = slot
        self.filepath = filepath
        self.signals = CompareLoadSignals()
        self.setAutoDelete(True)

    def run(self):
        try:
            result = DataManager.load_file_sync(self.filepath)
            self.signals.finished.emit(self.slot, self.filepath, result)
        except Exception as e:
            logger.error(f"Compare mode failed to load {self.filepath}: {e}")
            self.signals.error.emit(self.slot, self.filepath, str(e))


class MiniViewer(QWidget):
    """A single static preview: title, image, one-line metadata summary."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self.title_label = QLabel("")
        self.title_label.setStyleSheet("font-weight: bold;")
        self.title_label.setWordWrap(True)
        layout.addWidget(self.title_label)

        pg.setConfigOptions(imageAxisOrder='row-major')
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(False)
        self.plot_widget.showGrid(x=False, y=False)
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)
        layout.addWidget(self.plot_widget, 1)

        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

        self._colormap = "bone"
        self._invert = True
        apply_colormap_to_image(self.image_item, self._colormap, self._invert)

    def set_colormap(self, cmap_name: str, invert: bool):
        self._colormap = cmap_name
        self._invert = invert
        apply_colormap_to_image(self.image_item, cmap_name, invert)

    def show_loading(self, filename: str):
        self.title_label.setText(filename)
        self.info_label.setText("Loading…")
        self.image_item.clear()

    def show_result(self, result: DataResult):
        data = result.data
        dims = list(result.dims)
        while data.ndim > 2:
            idx = data.shape[-1] // 2
            data = data[..., idx]
            dims = dims[:-1]

        display_data = np.nan_to_num(np.asarray(data, dtype=np.float64), nan=0, posinf=0, neginf=0)
        if display_data.size == 0:
            self.info_label.setText("Empty data")
            return

        vmin, vmax = np.percentile(display_data, 1), np.percentile(display_data, 99)
        apply_colormap_to_image(self.image_item, self._colormap, self._invert)

        x_dim = dims[1] if len(dims) >= 2 else None
        y_dim = dims[0] if len(dims) >= 2 else None
        x_coords = result.coords.get(x_dim) if x_dim else None
        y_coords = result.coords.get(y_dim) if y_dim else None

        if x_coords is not None and y_coords is not None and len(x_coords) > 0 and len(y_coords) > 0:
            x_min, x_max = x_coords[0], x_coords[-1]
            y_min, y_max = y_coords[0], y_coords[-1]
            self.image_item.setImage(display_data, levels=(vmin, vmax))
            self.image_item.setRect(x_min, y_min, x_max - x_min, y_max - y_min)
            self.plot_widget.setXRange(x_min, x_max, padding=0.02)
            self.plot_widget.setYRange(y_min, y_max, padding=0.02)
        else:
            self.image_item.setImage(display_data, levels=(vmin, vmax))
            self.plot_widget.autoRange()

        self.title_label.setText(os.path.basename(result.filepath))
        summary = summarize_for_listing(result.meta)
        parts = []
        if summary['Type'] is not None:
            parts.append(str(summary['Type']))
        if summary['hv'] is not None:
            parts.append(f"hv={summary['hv']:.4g}")
        if summary['Temp'] is not None:
            parts.append(f"T={summary['Temp']:.4g}")
        self.info_label.setText(" | ".join(parts) if parts else "")

    def show_error(self, filename: str, error: str):
        self.title_label.setText(filename)
        self.info_label.setText(f"Error: {error}")
        self.image_item.clear()

    def clear(self):
        self.title_label.setText("")
        self.info_label.setText("")
        self.image_item.clear()


class ComparePanel(QWidget):
    """Side-by-side comparison of up to MAX_COMPARE_SLOTS pinned files."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._colormap = "bone"
        self._invert = True
        self._thread_pool = QThreadPool.globalInstance()
        self._pending_paths: List[Optional[str]] = [None] * MAX_COMPARE_SLOTS

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.viewers: List[MiniViewer] = []
        for i in range(MAX_COMPARE_SLOTS):
            if i > 0:
                divider = QFrame()
                divider.setFrameShape(QFrame.VLine)
                layout.addWidget(divider)
            viewer = MiniViewer()
            self.viewers.append(viewer)
            layout.addWidget(viewer, 1)

        self.set_pins([])

    def set_colormap(self, cmap_name: str, invert: bool):
        self._colormap = cmap_name
        self._invert = invert
        for viewer in self.viewers:
            viewer.set_colormap(cmap_name, invert)

    def set_pins(self, paths: List[str]):
        """Load and display up to MAX_COMPARE_SLOTS pinned files."""
        for i, viewer in enumerate(self.viewers):
            if i < len(paths):
                filepath = paths[i]
                self._pending_paths[i] = filepath
                viewer.show_loading(os.path.basename(filepath))
                self._load(i, filepath)
            else:
                self._pending_paths[i] = None
                viewer.clear()

    def _load(self, slot: int, filepath: str):
        # Connect directly to bound methods (not lambdas) - PySide6 can only
        # route a cross-thread signal to the correct (main) thread when it
        # can identify the receiving QObject, which a wrapping lambda hides.
        worker = CompareLoadWorker(slot, filepath)
        worker.signals.finished.connect(self._on_loaded)
        worker.signals.error.connect(self._on_error)
        self._thread_pool.start(worker)

    def _on_loaded(self, slot: int, filepath: str, result: DataResult):
        if self._pending_paths[slot] != filepath:
            return  # Superseded by a newer pin before this load finished.
        self.viewers[slot].show_result(result)

    def _on_error(self, slot: int, filepath: str, error: str):
        if self._pending_paths[slot] != filepath:
            return
        self.viewers[slot].show_error(os.path.basename(filepath), error)
