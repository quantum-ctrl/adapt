"""
File List Panel - Shows files in the selected folder with filtering.
Supports both list view and grid view modes with data thumbnail previews.
"""

import os
from datetime import datetime
from typing import Optional, Dict
import numpy as np

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QLabel, QAbstractItemView, QStackedWidget,
    QListWidget, QListWidgetItem, QPushButton, QFrame, QSlider
)
from PySide6.QtCore import Signal, Qt, QSize, QObject, QRunnable, QThreadPool
from PySide6.QtGui import QIcon, QFont, QImage, QPixmap, QPainter, QColor

from core.data_manager import DataManager, filter_files_by_type


# File type icons/emojis (fallback when no thumbnail available)
FILE_ICONS = {
    'HDF5': '📊',
    'IBW': '📈',
    'ZIP': '📦',
    '?': '📄'
}

# Default and range for thumbnail size
DEFAULT_THUMBNAIL_SIZE = 80
MIN_THUMBNAIL_SIZE = 48
MAX_THUMBNAIL_SIZE = 160

# Colormap definitions
COLORMAPS = {
    'viridis': [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]],
    'plasma': [[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]],
    'inferno': [[0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164]],
    'magma': [[0, 0, 4], [81, 18, 124], [183, 55, 121], [254, 159, 109], [252, 253, 191]],
    'cividis': [[0, 32, 77], [61, 78, 107], [124, 123, 120], [186, 173, 107], [253, 231, 37]],
    'hot': [[11, 0, 0], [180, 0, 0], [255, 127, 0], [255, 255, 63], [255, 255, 255]],
    'jet': [[0, 0, 127], [0, 127, 255], [127, 255, 127], [255, 127, 0], [127, 0, 0]],
    'turbo': [[48, 18, 59], [86, 191, 180], [156, 240, 81], [250, 170, 33], [122, 4, 3]],
    'coolwarm': [[59, 76, 192], [130, 159, 226], [221, 221, 221], [226, 131, 106], [180, 4, 38]],
    'RdBu': [[103, 0, 31], [178, 24, 43], [239, 138, 98], [247, 247, 247], [33, 102, 172]],
    'gray': [[0, 0, 0], [64, 64, 64], [128, 128, 128], [192, 192, 192], [255, 255, 255]],
    'bone': [[0, 0, 0], [41, 41, 57], [107, 107, 131], [166, 198, 198], [255, 255, 255]],
}


class ThumbnailSignals(QObject):
    """Signals for thumbnail worker - must be defined outside QRunnable."""
    finished = Signal(str, QPixmap, str, bool)  # filepath, thumbnail (always MAX size), cmap, invert
    error = Signal(str, str)  # filepath, error message


class ThumbnailWorker(QRunnable):
    """
    Worker for generating thumbnails in background threads.
    Uses QRunnable for thread pool efficiency.
    """
    
    def __init__(self, filepath: str, colormap: str = 'bone', invert: bool = True):
        super().__init__()
        self.filepath = filepath
        self.thumbnail_size = MAX_THUMBNAIL_SIZE  # Always render at max size
        self.colormap = colormap
        self.invert = invert
        self.signals = ThumbnailSignals()
        self.setAutoDelete(True)
    
    def run(self):
        """Load file and generate thumbnail."""
        try:
            result = DataManager.load_file_sync(self.filepath)
            pixmap = self._generate_thumbnail(result.data)
            self.signals.finished.emit(self.filepath, pixmap, self.colormap, self.invert)
        except Exception as e:
            self.signals.error.emit(self.filepath, str(e))
    
    def _generate_thumbnail(self, data: np.ndarray) -> QPixmap:
        """Generate a thumbnail from numpy array data."""
        # Get 2D slice from data
        if data.ndim == 1:
            # 1D data - create a simple line plot representation
            img_data = self._generate_1d_preview(data)
        elif data.ndim == 2:
            img_data = data
        elif data.ndim >= 3:
            # For 3D+ data: take middle slice of the LAST dimension (scan index)
            # This gives XY view at middle scan position
            img_data = data
            while img_data.ndim > 2:
                idx = img_data.shape[-1] // 2
                img_data = img_data[..., idx]
        else:
            return self._create_placeholder_pixmap()
        
        # Normalize to 0-255
        img_data = np.asarray(img_data, dtype=np.float64)
        if img_data.size == 0:
            return self._create_placeholder_pixmap()
        
        img_min, img_max = np.nanmin(img_data), np.nanmax(img_data)
        if img_max > img_min:
            img_data = (img_data - img_min) / (img_max - img_min) * 255
        else:
            img_data = np.zeros_like(img_data)
        
        img_data = np.nan_to_num(img_data, 0)
        img_data = np.clip(img_data, 0, 255).astype(np.uint8)
        
        # Apply colormap
        colored = self._apply_colormap(img_data)
        
        # Flip vertically to correct orientation (Qt uses top-left origin)
        colored = np.flipud(colored)
        
        # Create QImage - need to ensure data is contiguous
        h, w = img_data.shape
        colored = np.ascontiguousarray(colored)
        qimg = QImage(colored.data, w, h, w * 3, QImage.Format_RGB888)
        
        # Scale to thumbnail size, preserving aspect ratio
        pixmap = QPixmap.fromImage(qimg.copy())  # copy to own the data
        pixmap = pixmap.scaled(
            self.thumbnail_size, self.thumbnail_size,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        )
        
        return pixmap
    
    def _generate_1d_preview(self, data: np.ndarray) -> np.ndarray:
        """Generate a 2D image representation of 1D data."""
        size = self.thumbnail_size
        img = np.zeros((size, size), dtype=np.float64)
        
        if len(data) == 0:
            return img
        
        # Normalize data to fit in image height
        data = np.asarray(data, dtype=np.float64)
        data_min, data_max = np.nanmin(data), np.nanmax(data)
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = np.zeros_like(data)
        
        # Resample to image width
        indices = np.linspace(0, len(normalized) - 1, size).astype(int)
        resampled = normalized[indices]
        
        # Draw line
        for x, val in enumerate(resampled):
            y = int((1 - val) * (size - 1))
            y = max(0, min(size - 1, y))
            img[y, x] = 1.0
            # Draw vertical line below
            img[y:, x] = np.linspace(1.0, 0.3, size - y)
        
        return img
    
    def _apply_colormap(self, gray: np.ndarray) -> np.ndarray:
        """Apply the selected colormap to grayscale data."""
        h, w = gray.shape
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Get colormap colors
        cmap_colors = COLORMAPS.get(self.colormap, COLORMAPS['bone'])
        colors = np.array(cmap_colors, dtype=np.float32)
        
        # Invert if needed
        t = gray.astype(np.float32) / 255.0
        if self.invert:
            t = 1.0 - t
        
        # Interpolate
        positions = np.linspace(0, 1, len(colors))
        for i in range(3):
            result[:, :, i] = np.interp(t.flatten(), 
                                        positions,
                                        colors[:, i]).reshape(h, w).astype(np.uint8)
        
        return result
    
    def _create_placeholder_pixmap(self) -> QPixmap:
        """Create a placeholder pixmap for failed thumbnails."""
        pixmap = QPixmap(self.thumbnail_size, self.thumbnail_size)
        pixmap.fill(QColor(100, 100, 100))
        
        painter = QPainter(pixmap)
        painter.setPen(QColor(200, 200, 200))
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "?")
        painter.end()
        
        return pixmap


class FileListPanel(QWidget):
    """
    Panel showing files in the selected folder.
    Supports filtering by file type and switching between list/grid views.
    Grid view shows data thumbnail previews.
    """
    
    # Emitted when a file is selected
    file_selected = Signal(str)  # file path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._current_folder = ""
        self._current_filter = "All"
        self._all_files = []
        self._is_grid_view = False
        self._thumbnail_size = DEFAULT_THUMBNAIL_SIZE
        self._colormap = "bone"
        self._invert = True
        self._thumbnail_cache: Dict[str, tuple] = {}  # filepath -> (max_pixmap, cmap, invert)
        self._pending_thumbnails: set = set()
        self._thread_pool = QThreadPool.globalInstance()
        self._thread_pool.setMaxThreadCount(2)  # Limit concurrent loads
        self._last_emitted_file: Optional[str] = None  # Deduplication tracking
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header with view toggle
        header_frame = QFrame()
        header_frame.setObjectName("PanelHeaderFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(8, 4, 8, 4)
        header_layout.setSpacing(8)
        
        # Header label
        self.header = QLabel("📄 Files")
        self.header.setObjectName("PanelHeader")
        header_layout.addWidget(self.header)
        
        header_layout.addStretch()
        
        # Grid size slider (inline, only visible in grid view)
        self.slider_widget = QWidget()
        slider_layout = QHBoxLayout(self.slider_widget)
        slider_layout.setContentsMargins(0, 0, 0, 0)
        slider_layout.setSpacing(4)
        
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setMinimum(MIN_THUMBNAIL_SIZE)
        self.size_slider.setMaximum(MAX_THUMBNAIL_SIZE)
        self.size_slider.setValue(DEFAULT_THUMBNAIL_SIZE)
        self.size_slider.setTickPosition(QSlider.NoTicks)
        self.size_slider.setFixedWidth(80)
        self.size_slider.valueChanged.connect(self._on_thumbnail_size_changed)
        slider_layout.addWidget(self.size_slider)
        
        self.slider_widget.setVisible(False)  # Hidden by default
        header_layout.addWidget(self.slider_widget)
        
        # View toggle buttons
        self.list_btn = QPushButton("☰")
        self.list_btn.setToolTip("List View")
        self.list_btn.setFixedSize(28, 28)
        self.list_btn.setCheckable(True)
        self.list_btn.setChecked(True)
        self.list_btn.clicked.connect(lambda: self._set_view_mode(False))
        header_layout.addWidget(self.list_btn)
        
        self.grid_btn = QPushButton("⊞")
        self.grid_btn.setToolTip("Grid View")
        self.grid_btn.setFixedSize(28, 28)
        self.grid_btn.setCheckable(True)
        self.grid_btn.clicked.connect(lambda: self._set_view_mode(True))
        header_layout.addWidget(self.grid_btn)
        
        layout.addWidget(header_frame)
        
        # Stacked widget for list/grid views
        self.view_stack = QStackedWidget()
        
        # List View (Table)
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "Modified"])
        
        # Configure table
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        
        # Column sizing
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        
        # Connect table signals
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_cell_clicked)
        self.table.currentCellChanged.connect(self._on_table_selection_changed)
        
        self.view_stack.addWidget(self.table)  # Index 0
        
        # Grid View (List with icon mode)
        self.grid_list = QListWidget()
        self.grid_list.setViewMode(QListWidget.IconMode)
        self._update_grid_sizes()
        self.grid_list.setResizeMode(QListWidget.Adjust)
        self.grid_list.setMovement(QListWidget.Static)
        self.grid_list.setSpacing(10)
        self.grid_list.setWordWrap(True)
        self.grid_list.setSelectionMode(QAbstractItemView.SingleSelection)
        
        # Connect grid signals
        self.grid_list.itemClicked.connect(self._on_grid_item_clicked)
        self.grid_list.itemDoubleClicked.connect(self._on_grid_item_clicked)
        self.grid_list.currentItemChanged.connect(self._on_grid_selection_changed)
        
        self.view_stack.addWidget(self.grid_list)  # Index 1
        
        layout.addWidget(self.view_stack)
        
        # Apply button styles
        self._update_view_buttons()
    
    def _update_grid_sizes(self):
        """Update grid list sizes based on current thumbnail size."""
        size = self._thumbnail_size
        self.grid_list.setIconSize(QSize(size, size))
        self.grid_list.setGridSize(QSize(size + 30, size + 45))
    
    def _on_thumbnail_size_changed(self, value: int):
        """Handle thumbnail size slider change."""
        self._thumbnail_size = value
        self._update_grid_sizes()
        
        # Just refresh display - cache contains max-size pixmaps that will be scaled
        if self._is_grid_view:
            self._refresh_display()
    
    def set_colormap(self, colormap: str, invert: bool):
        """Set the colormap for thumbnails."""
        if self._colormap != colormap or self._invert != invert:
            self._colormap = colormap
            self._invert = invert
            # Clear cache (colormap changed) and refresh
            self._thumbnail_cache.clear()
            if self._is_grid_view:
                self._refresh_display()
    
    def _set_view_mode(self, is_grid: bool):
        """Switch between list and grid view."""
        self._is_grid_view = is_grid
        self.view_stack.setCurrentIndex(1 if is_grid else 0)
        self.slider_widget.setVisible(is_grid)  # Show slider only in grid view
        self._update_view_buttons()
        self._refresh_display()
    
    def _update_view_buttons(self):
        """Update toggle button states."""
        self.list_btn.setChecked(not self._is_grid_view)
        self.grid_btn.setChecked(self._is_grid_view)
        
        # Style for active/inactive buttons
        active_style = """
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 4px;
                font-size: 14px;
            }
        """
        inactive_style = """
            QPushButton {
                background-color: transparent;
                border: 1px solid #ccc;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #e0e0e0;
            }
        """
        
        self.list_btn.setStyleSheet(active_style if not self._is_grid_view else inactive_style)
        self.grid_btn.setStyleSheet(active_style if self._is_grid_view else inactive_style)
    
    def set_folder(self, folder_path: str):
        """Set the folder to display files from."""
        self._current_folder = folder_path
        self._load_files()
    
    def set_filter(self, filter_type: str):
        """Set the file type filter."""
        self._current_filter = filter_type
        self._refresh_display()
    
    def _load_files(self):
        """Load files from the current folder."""
        if not self._current_folder or not os.path.isdir(self._current_folder):
            self._all_files = []
            self._refresh_display()
            return
        
        try:
            entries = os.listdir(self._current_folder)
            self._all_files = [
                f for f in entries 
                if os.path.isfile(os.path.join(self._current_folder, f))
                and DataManager.is_supported(f)
            ]
            self._all_files.sort()
        except OSError:
            self._all_files = []
        
        self._refresh_display()
    
    def _refresh_display(self):
        """Refresh the display with current filter and view mode."""
        filtered = filter_files_by_type(self._all_files, self._current_filter)
        count = len(filtered)
        self.header.setText(f"📄 Files ({count})")
        
        if self._is_grid_view:
            self._populate_grid(filtered)
        else:
            self._populate_table(filtered)
    
    def _populate_table(self, files: list):
        """Populate the table view."""
        self.table.setRowCount(len(files))
        
        for row, filename in enumerate(files):
            filepath = os.path.join(self._current_folder, filename)
            
            name_item = QTableWidgetItem(filename)
            name_item.setData(Qt.UserRole, filepath)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 0, name_item)
            
            file_type = DataManager.get_file_type(filename) or "?"
            type_item = QTableWidgetItem(file_type)
            type_item.setFlags(type_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 1, type_item)
            
            try:
                mtime = os.path.getmtime(filepath)
                time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
            except OSError:
                time_str = "?"
            time_item = QTableWidgetItem(time_str)
            time_item.setFlags(time_item.flags() & ~Qt.ItemIsEditable)
            self.table.setItem(row, 2, time_item)
    
    def _populate_grid(self, files: list):
        """Populate the grid view with thumbnail previews."""
        self.grid_list.clear()
        
        for filename in files:
            filepath = os.path.join(self._current_folder, filename)
            
            item = QListWidgetItem()
            item.setData(Qt.UserRole, filepath)
            
            max_chars = max(12, self._thumbnail_size // 6)
            display_name = filename
            if len(display_name) > max_chars:
                display_name = display_name[:max_chars - 3] + "..."
            item.setText(display_name)
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
            item.setToolTip(filename)
            
            item.setSizeHint(QSize(self._thumbnail_size + 20, self._thumbnail_size + 40))
            
            # Check cache - cache stores max-size pixmaps with (pixmap, cmap, invert)
            cached = self._thumbnail_cache.get(filepath)
            if cached and cached[1] == self._colormap and cached[2] == self._invert:
                # Scale the cached max-size pixmap to current display size
                scaled = cached[0].scaled(
                    self._thumbnail_size, self._thumbnail_size,
                    Qt.IgnoreAspectRatio,
                    Qt.SmoothTransformation
                )
                item.setIcon(QIcon(scaled))
            else:
                item.setIcon(self._create_loading_icon())
                self._schedule_thumbnail_load(filepath)
            
            self.grid_list.addItem(item)
    
    def _create_loading_icon(self) -> QIcon:
        """Create a loading placeholder icon."""
        size = self._thumbnail_size
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(60, 60, 60))
        
        painter = QPainter(pixmap)
        painter.setPen(QColor(120, 120, 120))
        font = painter.font()
        font.setPointSize(max(8, size // 10))
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "...")
        painter.end()
        
        return QIcon(pixmap)
    
    def _schedule_thumbnail_load(self, filepath: str):
        """Schedule async thumbnail generation for a file."""
        if filepath in self._pending_thumbnails:
            return
        
        self._pending_thumbnails.add(filepath)
        
        worker = ThumbnailWorker(filepath, self._colormap, self._invert)
        worker.signals.finished.connect(self._on_thumbnail_ready)
        worker.signals.error.connect(self._on_thumbnail_error)
        
        self._thread_pool.start(worker)
    
    def _on_thumbnail_ready(self, filepath: str, pixmap: QPixmap, cmap: str, invert: bool):
        """Handle completed thumbnail generation."""
        self._pending_thumbnails.discard(filepath)
        self._thumbnail_cache[filepath] = (pixmap, cmap, invert)  # Store max-size pixmap
        
        # Only update if colormap settings still match
        if cmap != self._colormap or invert != self._invert:
            return
        
        # Scale to current display size
        scaled = pixmap.scaled(
            self._thumbnail_size, self._thumbnail_size,
            Qt.IgnoreAspectRatio,
            Qt.SmoothTransformation
        )
        
        for i in range(self.grid_list.count()):
            item = self.grid_list.item(i)
            if item and item.data(Qt.UserRole) == filepath:
                item.setIcon(QIcon(scaled))
                break
    
    def _on_thumbnail_error(self, filepath: str, error: str):
        """Handle thumbnail generation error."""
        self._pending_thumbnails.discard(filepath)
        
        size = self._thumbnail_size
        pixmap = QPixmap(size, size)
        pixmap.fill(QColor(80, 40, 40))
        
        painter = QPainter(pixmap)
        painter.setPen(QColor(200, 100, 100))
        font = painter.font()
        font.setPointSize(max(8, size // 10))
        painter.setFont(font)
        painter.drawText(pixmap.rect(), Qt.AlignCenter, "Error")
        painter.end()
        
        self._thumbnail_cache[filepath] = (pixmap, self._colormap, self._invert)
        
        for i in range(self.grid_list.count()):
            item = self.grid_list.item(i)
            if item and item.data(Qt.UserRole) == filepath:
                item.setIcon(QIcon(pixmap))
                break
    
    def _on_cell_clicked(self, row: int, column: int):
        """Handle table cell click."""
        item = self.table.item(row, 0)
        if item:
            filepath = item.data(Qt.UserRole)
            if filepath and filepath != self._last_emitted_file:
                self._last_emitted_file = filepath
                self.file_selected.emit(filepath)
    
    def _on_grid_item_clicked(self, item: QListWidgetItem):
        """Handle grid item click."""
        filepath = item.data(Qt.UserRole)
        if filepath and filepath != self._last_emitted_file:
            self._last_emitted_file = filepath
            self.file_selected.emit(filepath)
    
    def _on_table_selection_changed(self, current_row: int, current_col: int, 
                                     previous_row: int, previous_col: int):
        """Handle table selection change (keyboard navigation)."""
        if current_row != previous_row and current_row >= 0:
            item = self.table.item(current_row, 0)
            if item:
                filepath = item.data(Qt.UserRole)
                if filepath and filepath != self._last_emitted_file:
                    self._last_emitted_file = filepath
                    self.file_selected.emit(filepath)
    
    def _on_grid_selection_changed(self, current: QListWidgetItem, previous: QListWidgetItem):
        """Handle grid selection change (keyboard navigation)."""
        if current and current != previous:
            filepath = current.data(Qt.UserRole)
            if filepath and filepath != self._last_emitted_file:
                self._last_emitted_file = filepath
                self.file_selected.emit(filepath)
    
    def get_selected_file(self) -> Optional[str]:
        """Get the currently selected file path."""
        if self._is_grid_view:
            items = self.grid_list.selectedItems()
            if items:
                return items[0].data(Qt.UserRole)
        else:
            items = self.table.selectedItems()
            if items:
                return items[0].data(Qt.UserRole)
        return None
    
    def clear_thumbnail_cache(self):
        """Clear the thumbnail cache to free memory."""
        self._thumbnail_cache.clear()
