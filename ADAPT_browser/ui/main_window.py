"""
Main Window - Primary application window with all panels.
"""

import os
import webbrowser

from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QToolBar, QComboBox, QSplitter, QStatusBar,
    QFileDialog, QLabel, QApplication, QSizePolicy,
    QProgressBar, QCheckBox, QMessageBox, QStackedWidget, QMenu, QToolButton
)
from PySide6.QtCore import Qt, QSize, QTimer, QFileSystemWatcher
from PySide6.QtGui import QAction, QIcon, QPalette, QColor

from .directory_panel import DirectoryPanel
from .file_list_panel import FileListPanel
from .viewer_panel import ViewerPanel
from .compare_panel import ComparePanel
from .collections_panel import CollectionsPanel
from ADAPT_browser.core.data_manager import DataManager, DataResult
from ADAPT_browser.utils.logger import logger

try:
    from shared.session import write_session, get_collections, get_recent_folders, add_recent_folder
except ImportError:
    # Fallback if shared module not available
    def write_session(file_path, metadata=None):
        logger.warning("Session manager not available")
        return False

    def get_collections():
        return {}

    def get_recent_folders():
        return []

    def add_recent_folder(path):
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

        # Watch the active folder so newly acquired files appear automatically.
        self._watch_folder_enabled = True
        self._watched_folder = ""
        self._folder_watcher = QFileSystemWatcher(self)
        self._folder_watcher.directoryChanged.connect(self._on_watched_folder_changed)
        self._watch_refresh_timer = QTimer(self)
        self._watch_refresh_timer.setSingleShot(True)
        self._watch_refresh_timer.setInterval(600)
        self._watch_refresh_timer.timeout.connect(self._refresh_watched_folder)
        
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

        # Recent Folders menu
        self.recent_button = QToolButton()
        self.recent_button.setText("🕐 Recent")
        self.recent_button.setPopupMode(QToolButton.InstantPopup)
        self.recent_button.setToolTip("Recently opened folders")
        self.recent_menu = QMenu(self.recent_button)
        self.recent_button.setMenu(self.recent_menu)
        self._refresh_recent_menu()
        toolbar.addWidget(self.recent_button)

        # Watch Folder toggle
        self.watch_folder_action = QAction("👁 Watch Folder", self)
        self.watch_folder_action.setCheckable(True)
        self.watch_folder_action.setChecked(self._watch_folder_enabled)
        self.watch_folder_action.setToolTip("Automatically refresh the file list when the selected folder changes")
        self.watch_folder_action.toggled.connect(self._on_watch_folder_toggled)
        toolbar.addAction(self.watch_folder_action)
        
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
        self.open_viewer_action.setToolTip("Open selected file in ADAPT Edit (web-based 3D visualization)")
        self.open_viewer_action.triggered.connect(self._on_open_with_viewer)
        toolbar.addAction(self.open_viewer_action)

        # Compare mode toggle
        self.compare_action = QAction("🆚 Compare", self)
        self.compare_action.setCheckable(True)
        self.compare_action.setToolTip(
            "Right-click up to two files and 'Pin for Compare' to view them side by side"
        )
        self.compare_action.toggled.connect(self._on_compare_toggled)
        toolbar.addAction(self.compare_action)

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
        
        # Left column: Collections list (top) + Directory tree (bottom)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)
        left_widget.setMinimumWidth(180)
        left_widget.setMaximumWidth(300)

        self.collections_panel = CollectionsPanel()
        self.collections_panel.setMaximumHeight(160)
        self.collections_panel.collection_selected.connect(self._on_collection_selected)
        left_layout.addWidget(self.collections_panel)

        self.dir_panel = DirectoryPanel()
        self.dir_panel.folder_selected.connect(self._on_folder_selected)
        left_layout.addWidget(self.dir_panel, 1)

        self.splitter.addWidget(left_widget)

        # Middle panel: File list
        self.file_panel = FileListPanel()
        self.file_panel.setMinimumWidth(200)
        # No max width - let the user resize freely
        self.file_panel.file_selected.connect(self._on_file_selected)
        self.file_panel.compare_pins_changed.connect(self._on_compare_pins_changed)
        self.file_panel.collections_changed.connect(self.collections_panel.refresh)
        self.splitter.addWidget(self.file_panel)

        # Right panel: Viewer / Compare (stacked)
        self.viewer_panel = ViewerPanel()
        self.viewer_panel.cursor_moved.connect(self._on_cursor_moved)

        self.compare_panel = ComparePanel()

        self.right_stack = QStackedWidget()
        self.right_stack.setMinimumWidth(400)
        self.right_stack.addWidget(self.viewer_panel)   # Index 0
        self.right_stack.addWidget(self.compare_panel)  # Index 1
        self.splitter.addWidget(self.right_stack)

        # Set initial sizes (roughly 15%, 30%, 55%) - the file list needs
        # extra room now that it shows Scan Type/hv/Temp columns
        self.splitter.setSizes([200, 420, 580])
        
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
        self._set_watched_folder(folder_path)
        add_recent_folder(folder_path)
        self._refresh_recent_menu()

    def _refresh_recent_menu(self):
        """Rebuild the Recent Folders menu from persisted app state."""
        self.recent_menu.clear()
        recent = get_recent_folders()

        if not recent:
            empty_action = self.recent_menu.addAction("(no recent folders)")
            empty_action.setEnabled(False)
            return

        for path in recent:
            action = self.recent_menu.addAction(path)
            action.triggered.connect(lambda checked=False, p=path: self._open_recent_folder(p))

    def _open_recent_folder(self, path: str):
        """Reopen a folder from the Recent Folders menu."""
        if not os.path.isdir(path):
            QMessageBox.warning(self, "Folder Not Found", f"This folder no longer exists:\n{path}")
            return
        self.dir_panel.set_root_path(path)

    def _on_collection_selected(self, name: str):
        """Handle a collection being selected - show its files (a virtual, cross-folder folder)."""
        logger.debug(f"Collection selected: {name}")
        collection = get_collections().get(name, {})
        paths = collection.get('files', [])
        self.file_panel.set_file_list(paths, f"⭐ {name}")
        self._set_watched_folder("")

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
        self.compare_panel.set_colormap(cmap_name, invert)

    def _on_invert_toggled(self, checked: bool):
        """Handle invert toggle."""
        cmap = self.cmap_combo.currentText()
        self.viewer_panel.set_colormap(cmap, checked)
        self.file_panel.set_colormap(cmap, checked)
        self.compare_panel.set_colormap(cmap, checked)

    def _on_theme_changed(self, theme_name: str):
        """Handle theme change."""
        self._apply_theme(theme_name)

    def _on_compare_toggled(self, checked: bool):
        """Switch the right-hand panel between the single viewer and compare mode."""
        self.right_stack.setCurrentIndex(1 if checked else 0)

    def _on_compare_pins_changed(self, paths: list):
        """Update the compare panel when the pinned file set changes."""
        self.compare_panel.set_pins(paths)
        if paths and not self.compare_action.isChecked():
            self.compare_action.setChecked(True)

    def _on_watch_folder_toggled(self, checked: bool):
        """Enable or disable automatic refresh for the selected folder."""
        self._watch_folder_enabled = checked
        if checked:
            folder = self.file_panel.get_current_folder()
            self._set_watched_folder(folder)
            if folder:
                self.status_bar.showMessage(f"Watching folder: {folder}")
        else:
            self._clear_watched_folders()
            self.status_bar.showMessage("Folder watching disabled.")

    def _set_watched_folder(self, folder_path: str):
        """Watch one folder at a time: the folder currently shown in the file panel."""
        self._clear_watched_folders()
        self._watched_folder = ""

        if not self._watch_folder_enabled:
            return
        if not folder_path or not os.path.isdir(folder_path):
            return

        if self._folder_watcher.addPath(folder_path):
            self._watched_folder = folder_path
            logger.debug(f"Watching folder: {folder_path}")
        else:
            logger.warning(f"Failed to watch folder: {folder_path}")

    def _clear_watched_folders(self):
        """Remove all directories from the folder watcher."""
        watched = self._folder_watcher.directories()
        if watched:
            self._folder_watcher.removePaths(watched)

    def _on_watched_folder_changed(self, folder_path: str):
        """Debounce directory notifications before refreshing the file list."""
        if not self._watch_folder_enabled:
            return
        if folder_path != self._watched_folder:
            return

        logger.debug(f"Watched folder changed: {folder_path}")
        self._watch_refresh_timer.start()

    def _refresh_watched_folder(self):
        """Refresh the file list after the watched folder changes."""
        folder = self._watched_folder
        if not self._watch_folder_enabled or not folder:
            return
        if not os.path.isdir(folder):
            self._set_watched_folder("")
            self.status_bar.showMessage("Watched folder is no longer available.")
            return

        count = self.file_panel.refresh_current_folder()

        # Some platforms drop watches after directory changes; keep the active
        # folder registered without disturbing the user's current selection.
        if folder not in self._folder_watcher.directories():
            self._folder_watcher.addPath(folder)

        self.status_bar.showMessage(f"Folder updated: {count} supported files")
    
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
        and opens ADAPT Edit in the default web browser.
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
                "axes": self._to_session_jsonable(data_result.coords),
                "attrs": self._to_session_jsonable(data_result.meta),
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
        viewer_host = os.environ.get("ADAPT_HOST", "127.0.0.1")
        if viewer_host in ("0.0.0.0", "::"):
            viewer_host = "127.0.0.1"
        viewer_port = os.environ.get("ADAPT_PORT", "8000")
        viewer_url = f"http://{viewer_host}:{viewer_port}/?session=1"
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

    def _to_session_jsonable(self, value):
        """Convert numpy-heavy metadata into values json.dump can handle."""
        if isinstance(value, dict):
            return {str(k): self._to_session_jsonable(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_session_jsonable(v) for v in value]
        if hasattr(value, "tolist"):
            return self._to_session_jsonable(value.tolist())
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        return str(value)
