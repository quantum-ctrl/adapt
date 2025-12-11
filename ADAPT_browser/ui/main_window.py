"""
Main Window - Primary application window with all panels.
"""

import os
import sys
import webbrowser

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QToolBar, QComboBox, QSplitter, QStatusBar,
    QFileDialog, QLabel, QApplication, QSizePolicy,
    QProgressBar, QCheckBox, QMessageBox
)
from PySide6.QtCore import Qt, QSize, QTimer
from PySide6.QtGui import QAction, QIcon, QPalette, QColor

from .directory_panel import DirectoryPanel
from .file_list_panel import FileListPanel
from .viewer_panel import ViewerPanel
from core.data_manager import DataManager, DataResult
from utils.logger import logger

# Add shared module to path for session manager
_shared_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)

try:
    from session import write_session
except ImportError:
    # Fallback if shared module not available
    def write_session(file_path, metadata=None):
        logger.warning("Session manager not available")
        return False


class MainWindow(QMainWindow):
    """
    Main application window for the ADAPT Data Browser.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ADAPT - Data Browser")
        self.setMinimumSize(1200, 700)
        self.resize(1400, 800)
        
        # Data manager
        self.data_manager = DataManager(self)
        self.data_manager.loading_started.connect(self._on_loading_started)
        self.data_manager.loading_finished.connect(self._on_loading_finished)
        self.data_manager.loading_error.connect(self._on_loading_error)
        self.data_manager.loading_progress.connect(self._on_loading_progress)
        
        # File selection throttling (debounce rapid clicks/key presses)
        self._pending_file_path = None
        self._file_select_timer = QTimer(self)
        self._file_select_timer.setSingleShot(True)
        self._file_select_timer.setInterval(150)  # 150ms debounce
        self._file_select_timer.timeout.connect(self._load_pending_file)
        
        # Build UI
        self._setup_toolbar()
        self._setup_central_widget()
        self._setup_status_bar()
        
        # Apply default theme (after UI is built)
        self._apply_theme('Light')
        
        # Center window
        self.center_on_screen()
        
        logger.info("MainWindow initialized")
    
    def center_on_screen(self):
        """Center the window on the screen."""
        screen = QApplication.primaryScreen()
        if screen:
            rect = screen.availableGeometry()
            center = rect.center()
            frame_gm = self.frameGeometry()
            frame_gm.moveCenter(center)
            self.move(frame_gm.topLeft())
    
    def _apply_theme(self, theme_name: str):
        """
        Apply the specified theme.
        
        Args:
            theme_name: 'Dark' or 'Light'
        """
        palette = QPalette()
        
        if theme_name == 'Light':
            # Light Theme Colors
            bg_color = QColor(240, 240, 240)
            text_color = QColor(30, 30, 30)
            base_color = QColor(255, 255, 255)
            alt_base_color = QColor(245, 245, 245)
            button_color = QColor(225, 225, 225)
            highlight_color = QColor(0, 120, 215)
            
            # Stylesheet colors
            qss_bg = "#f0f0f0"
            qss_toolbar_bg = "#e0e0e0"
            qss_toolbar_btn_hover = "#d0d0d0"
            qss_toolbar_btn_pressed = "#c0c0c0"
            qss_combo_bg = "#ffffff"
            qss_combo_border = "#cccccc"
            qss_text = "#1e1e1e"
            
        else:  # Dark Theme
            # Dark Theme Colors
            bg_color = QColor(30, 30, 30)
            text_color = QColor(212, 212, 212)
            base_color = QColor(37, 37, 38)
            alt_base_color = QColor(45, 45, 48)
            button_color = QColor(45, 45, 48)
            highlight_color = QColor(9, 71, 113)
            
            # Stylesheet colors
            qss_bg = "#1e1e1e"
            qss_toolbar_bg = "#2d2d30"
            qss_toolbar_btn_hover = "#3c3c3c"
            qss_toolbar_btn_pressed = "#505050"
            qss_combo_bg = "#3c3c3c"
            qss_combo_border = "#4c4c4c"
            qss_text = "#d4d4d4"

        # Apply Palette
        palette.setColor(QPalette.Window, bg_color)
        palette.setColor(QPalette.WindowText, text_color)
        palette.setColor(QPalette.Base, base_color)
        palette.setColor(QPalette.AlternateBase, alt_base_color)
        palette.setColor(QPalette.ToolTipBase, base_color)
        palette.setColor(QPalette.ToolTipText, text_color)
        palette.setColor(QPalette.Text, text_color)
        palette.setColor(QPalette.Button, button_color)
        palette.setColor(QPalette.ButtonText, text_color)
        palette.setColor(QPalette.BrightText, Qt.red)
        palette.setColor(QPalette.Link, QColor(42, 130, 218))
        palette.setColor(QPalette.Highlight, highlight_color)
        palette.setColor(QPalette.HighlightedText, Qt.white)
        
        # Disabled colors
        palette.setColor(QPalette.Disabled, QPalette.Text, QColor(128, 128, 128))
        palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(128, 128, 128))
        
        QApplication.instance().setPalette(palette)
        
        # Apply Stylesheet
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {qss_bg};
            }}
            QToolBar {{
                background-color: {qss_toolbar_bg};
                border: none;
                spacing: 8px;
                padding: 4px;
            }}
            QToolBar QToolButton {{
                background-color: transparent;
                border: 1px solid transparent;
                border-radius: 4px;
                padding: 6px 12px;
                color: {qss_text};
                font-size: 13px;
            }}
            QToolBar QToolButton:hover {{
                background-color: {qss_toolbar_btn_hover};
                border-color: {qss_combo_border};
            }}
            QToolBar QToolButton:pressed {{
                background-color: {qss_toolbar_btn_pressed};
            }}
            QComboBox {{
                background-color: {qss_combo_bg};
                border: 1px solid {qss_combo_border};
                border-radius: 4px;
                padding: 6px 12px;
                min-width: 100px;
                color: {qss_text};
            }}
            QComboBox:hover {{
                border-color: {highlight_color.name()};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {qss_toolbar_bg};
                border: 1px solid {qss_combo_border};
                selection-background-color: {highlight_color.name()};
                color: {qss_text};
            }}
            QStatusBar {{
                background-color: {highlight_color.name()};
                color: white;
                font-size: 12px;
            }}
            QSplitter::handle {{
                background-color: {qss_combo_bg};
            }}
            QSplitter::handle:horizontal {{
                width: 2px;
            }}
            QCheckBox {{
                color: {qss_text};
                padding: 0 8px;
            }}
            
            /* Panel Headers */
            QLabel#PanelHeader {{
                font-weight: bold;
                padding: 8px;
                background-color: {qss_toolbar_bg};
                border-bottom: 1px solid {qss_combo_border};
                color: {qss_text};
            }}
            
            /* Tree and Table Views */
            QTreeView, QTableWidget {{
                background-color: {qss_bg};
                border: none;
                font-size: 13px;
                color: {qss_text};
            }}
            QTreeView::item, QTableWidget::item {{
                padding: 4px 8px;
            }}
            QTreeView::item:hover, QTableWidget::item:hover {{
                background-color: {qss_toolbar_btn_hover};
            }}
            QTreeView::item:selected, QTableWidget::item:selected {{
                background-color: {highlight_color.name()};
                color: white;
            }}
            
            /* Grid View (QListWidget) */
            QListWidget {{
                background-color: {qss_bg};
                border: none;
                font-size: 12px;
                color: {qss_text};
            }}
            QListWidget::item {{
                padding: 8px;
                border-radius: 6px;
            }}
            QListWidget::item:hover {{
                background-color: {qss_toolbar_btn_hover};
            }}
            QListWidget::item:selected {{
                background-color: {highlight_color.name()};
                color: white;
            }}
            
            /* Tree View Specifics */
            QTreeView::branch:has-children:!has-siblings:closed,
            QTreeView::branch:closed:has-children:has-siblings {{
                image: url(none);
            }}
            QTreeView::branch:open:has-children:!has-siblings,
            QTreeView::branch:open:has-children:has-siblings {{
                image: url(none);
            }}
            
            /* Header View */
            QHeaderView::section {{
                background-color: {qss_toolbar_bg};
                padding: 6px 8px;
                border: none;
                border-bottom: 1px solid {qss_combo_border};
                font-weight: bold;
                color: {qss_text};
            }}
            
            /* Metadata Panel */
            QGroupBox#MetadataGroup {{
                font-weight: bold;
                border: 1px solid {qss_combo_border};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                color: {qss_text};
            }}
            QGroupBox#MetadataGroup::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
            QTextEdit#MetadataText {{
                background-color: {qss_combo_bg};
                border: none;
                font-family: 'Menlo', 'Monaco', 'Consolas', monospace;
                font-size: 12px;
                padding: 8px;
                color: {qss_text};
            }}
        """)
        
        # Update Viewer Panel Theme
        if hasattr(self, 'viewer_panel'):
            self.viewer_panel.update_theme(theme_name)
    
    def _setup_toolbar(self):
        """Create the toolbar."""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(16, 16))
        self.addToolBar(toolbar)
        
        # Open Folder action
        self.open_action = QAction("📂 Open Folder", self)
        self.open_action.setShortcut("Ctrl+O")
        self.open_action.triggered.connect(self._on_open_folder)
        toolbar.addAction(self.open_action)
        
        toolbar.addSeparator()
        
        # Filter label
        filter_label = QLabel(" Filter: ")
        toolbar.addWidget(filter_label)
        
        # File filter dropdown
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "HDF5", "IBW", "ZIP", "PXT"])
        self.filter_combo.currentTextChanged.connect(self._on_filter_changed)
        toolbar.addWidget(self.filter_combo)
        
        toolbar.addSeparator()
        
        # Colormap label
        cmap_label = QLabel(" Colormap: ")
        toolbar.addWidget(cmap_label)
        
        # Colormap dropdown
        self.cmap_combo = QComboBox()
        self.cmap_combo.addItems([
            "viridis", "plasma", "inferno", "magma", "cividis",
            "hot", "jet", "turbo", "coolwarm", "RdBu",
            "gray", "bone"
        ])
        self.cmap_combo.setCurrentText("viridis")
        self.cmap_combo.currentTextChanged.connect(self._on_colormap_changed)
        toolbar.addWidget(self.cmap_combo)
        
        # Invert toggle
        self.invert_checkbox = QCheckBox("Invert")
        self.invert_checkbox.toggled.connect(self._on_invert_toggled)
        toolbar.addWidget(self.invert_checkbox)
        
        toolbar.addSeparator()
        
        # Open with Viewer button
        self.open_viewer_action = QAction("🔬 Open with Viewer", self)
        self.open_viewer_action.setShortcut("Ctrl+Shift+V")
        self.open_viewer_action.setToolTip("Open selected file in ADAPT Viewer (web-based 3D visualization)")
        self.open_viewer_action.triggered.connect(self._on_open_with_viewer)
        toolbar.addAction(self.open_viewer_action)
        
        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        toolbar.addWidget(spacer)
        
        # Theme label
        theme_label = QLabel("Theme: ")
        toolbar.addWidget(theme_label)
        
        # Theme dropdown
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentText("Light")
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)
        toolbar.addWidget(self.theme_combo)
    
    def _setup_central_widget(self):
        """Create the main content area with three panels."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        layout = QHBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main splitter
        self.splitter = QSplitter(Qt.Horizontal)
        
        # Left panel: Directory tree
        self.dir_panel = DirectoryPanel()
        self.dir_panel.setMinimumWidth(180)
        self.dir_panel.setMaximumWidth(300)
        self.dir_panel.folder_selected.connect(self._on_folder_selected)
        self.splitter.addWidget(self.dir_panel)
        
        # Middle panel: File list
        self.file_panel = FileListPanel()
        self.file_panel.setMinimumWidth(200)
        # No max width - let the user resize freely
        self.file_panel.file_selected.connect(self._on_file_selected)
        self.splitter.addWidget(self.file_panel)
        
        # Right panel: Viewer
        self.viewer_panel = ViewerPanel()
        self.viewer_panel.setMinimumWidth(400)
        self.viewer_panel.cursor_moved.connect(self._on_cursor_moved)
        self.splitter.addWidget(self.viewer_panel)
        
        # Set initial sizes (roughly 15%, 20%, 65%)
        self.splitter.setSizes([200, 250, 750])
        
        layout.addWidget(self.splitter)
    
    def _setup_status_bar(self):
        """Create the status bar with progress indicator."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Progress bar for data loading
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximumWidth(200)
        self.progress_bar.setMinimumWidth(150)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(False)  # Hidden by default
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555;
                border-radius: 4px;
                background-color: #2d2d30;
                text-align: center;
                color: white;
                font-size: 11px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(
                    x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0078d7, stop:1 #00a2ed
                );
                border-radius: 3px;
            }
        """)
        self.status_bar.addWidget(self.progress_bar)
        
        # Permanent widgets
        self.cursor_label = QLabel("")
        self.shape_label = QLabel("")
        
        self.status_bar.addPermanentWidget(self.cursor_label)
        self.status_bar.addPermanentWidget(self.shape_label)
        
        self.status_bar.showMessage("Ready. Open a folder to begin.")
    
    # === Signal Handlers ===
    
    def _on_open_folder(self):
        """Handle Open Folder action."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Data Folder",
            os.path.expanduser("~"),
            QFileDialog.ShowDirsOnly
        )
        
        if folder:
            logger.info(f"Opening folder: {folder}")
            self.dir_panel.set_root_path(folder)
            self.status_bar.showMessage(f"Opened: {folder}")
    
    def _on_folder_selected(self, folder_path: str):
        """Handle folder selection in directory tree."""
        logger.debug(f"Folder selected: {folder_path}")
        self.file_panel.set_folder(folder_path)
    
    def _on_filter_changed(self, filter_type: str):
        """Handle file filter change."""
        logger.debug(f"Filter changed: {filter_type}")
        self.file_panel.set_filter(filter_type)
    
    def _on_file_selected(self, filepath: str):
        """Handle file selection - debounced to prevent rapid loading."""
        self._pending_file_path = filepath
        # Restart timer - only the last selection within 150ms will trigger load
        self._file_select_timer.start()
    
    def _load_pending_file(self):
        """Actually load the pending file after debounce delay."""
        if self._pending_file_path:
            logger.info(f"Loading file: {self._pending_file_path}")
            self.data_manager.load_file_async(self._pending_file_path)
            self._pending_file_path = None
    
    def _on_loading_started(self, filepath: str):
        """Handle loading start."""
        self.status_bar.showMessage(f"Loading {os.path.basename(filepath)}...")
        self.setCursor(Qt.WaitCursor)
        # Show and reset progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self._start_loading_animation()
    
    def _on_loading_finished(self, result: DataResult):
        """Handle loading completion."""
        self.setCursor(Qt.ArrowCursor)
        self._stop_loading_animation()
        
        # Complete the progress bar
        self.progress_bar.setValue(100)
        
        if result is not None:
            self.viewer_panel.set_data(result)
            self.shape_label.setText(f"  Shape: {result.shape} | dtype: {result.dtype}  ")
            self.status_bar.showMessage(f"Loaded: {os.path.basename(result.filepath)}")
        else:
            self.status_bar.showMessage("Failed to load file.")
        
        # Hide progress bar after a short delay
        QTimer.singleShot(1500, lambda: self.progress_bar.setVisible(False))
    
    def _on_loading_error(self, error_msg: str):
        """Handle loading error."""
        self.setCursor(Qt.ArrowCursor)
        self._stop_loading_animation()
        self.progress_bar.setVisible(False)
        self.status_bar.showMessage(f"Error: {error_msg}")
        logger.error(f"Loading error: {error_msg}")
    
    def _on_loading_progress(self, message: str):
        """Handle loading progress update."""
        self.status_bar.showMessage(message)
    
    def _on_cursor_moved(self, x: float, y: float, value: float):
        """Handle cursor movement in viewer."""
        self.cursor_label.setText(f"  ({x:.3f}, {y:.3f}) = {value:.4g}  ")
    
    def _on_colormap_changed(self, cmap_name: str):
        """Handle colormap change."""
        invert = self.invert_checkbox.isChecked()
        self.viewer_panel.set_colormap(cmap_name, invert)
        self.file_panel.set_colormap(cmap_name, invert)
    
    def _on_invert_toggled(self, checked: bool):
        """Handle invert toggle."""
        cmap = self.cmap_combo.currentText()
        self.viewer_panel.set_colormap(cmap, checked)
        self.file_panel.set_colormap(cmap, checked)
        
    def _on_theme_changed(self, theme_name: str):
        """Handle theme change."""
        self._apply_theme(theme_name)
    
    def _start_loading_animation(self):
        """Start the animated progress bar for loading indication."""
        if not hasattr(self, '_loading_timer'):
            self._loading_timer = QTimer(self)
            self._loading_timer.timeout.connect(self._animate_progress)
        
        self._loading_progress_value = 0
        self._loading_timer.start(60)  # Update every 60ms
    
    def _animate_progress(self):
        """Animate the progress bar with an ease-out curve."""
        # Use logarithmic progression that slows down as it approaches 90%
        # This creates a sense of progress without completing
        target = 90  # Never reach 100% until actually done
        current = self._loading_progress_value
        
        if current < target:
            # Increment slows as we approach target
            increment = max(0.5, (target - current) * 0.05)
            self._loading_progress_value = min(current + increment, target)
            self.progress_bar.setValue(int(self._loading_progress_value))
    
    def _stop_loading_animation(self):
        """Stop the loading animation."""
        if hasattr(self, '_loading_timer') and self._loading_timer:
            self._loading_timer.stop()
    
    def _on_open_with_viewer(self):
        """
        Handle 'Open with Viewer' action.
        
        Writes the current file path and metadata to session.json
        and opens ADAPT Viewer in the default web browser.
        """
        # Get the currently selected file from the file panel
        selected_file = self.file_panel.get_selected_file()
        
        if not selected_file:
            QMessageBox.warning(
                self,
                "No File Selected",
                "Please select a file first before opening with Viewer."
            )
            return
        
        # Verify file exists
        if not os.path.exists(selected_file):
            QMessageBox.warning(
                self,
                "File Not Found",
                f"The selected file does not exist:\n{selected_file}"
            )
            return
        
        # Get metadata from the currently loaded data (if available)
        metadata = {}
        if hasattr(self, 'viewer_panel') and self.viewer_panel._data:
            data_result = self.viewer_panel._data
            metadata = {
                "shape": list(data_result.shape) if data_result.shape else None,
                "dtype": str(data_result.dtype) if data_result.dtype else None,
                "axes": data_result.axes if hasattr(data_result, 'axes') else {},
                "attrs": data_result.metadata if hasattr(data_result, 'metadata') else {},
            }
        
        # Write session
        logger.info(f"Writing session for: {selected_file}")
        success = write_session(selected_file, metadata)
        
        if not success:
            QMessageBox.warning(
                self,
                "Session Error",
                "Failed to write session file. Check logs for details."
            )
            return
        
        # Open Viewer in default browser
        viewer_url = "http://localhost:8000/?session=1"
        logger.info(f"Opening Viewer: {viewer_url}")
        
        try:
            webbrowser.open(viewer_url)
            self.status_bar.showMessage(f"Opened Viewer for: {os.path.basename(selected_file)}")
        except Exception as e:
            logger.error(f"Failed to open browser: {e}")
            QMessageBox.warning(
                self,
                "Browser Error",
                f"Failed to open web browser:\n{e}\n\nPlease manually navigate to:\n{viewer_url}"
            )
