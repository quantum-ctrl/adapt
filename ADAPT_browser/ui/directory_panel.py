"""
Directory Panel - Folder tree navigation using QTreeView.
"""

import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTreeView, QLabel, QHeaderView, QFileSystemModel
)
from PySide6.QtCore import Signal, QDir, QModelIndex


class DirectoryPanel(QWidget):
    """
    Panel showing folder hierarchy for navigation.
    Only displays directories, not files.
    """
    
    # Emitted when a folder is selected
    folder_selected = Signal(str)  # folder path
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._root_path = ""
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Header label
        header = QLabel("📁 Folders")
        header.setObjectName("PanelHeader")
        layout.addWidget(header)
        
        # Tree view for directories
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setAnimated(True)
        self.tree_view.setIndentation(16)
        
        # File system model (directories only)
        self.model = QFileSystemModel()
        self.model.setFilter(QDir.AllDirs | QDir.NoDotAndDotDot)
        self.model.setRootPath("")
        
        self.tree_view.setModel(self.model)
        
        # Hide all columns except name
        for i in range(1, self.model.columnCount()):
            self.tree_view.hideColumn(i)
        
        # Connect signals
        self.tree_view.clicked.connect(self._on_folder_clicked)
        
        layout.addWidget(self.tree_view)
    
    def set_root_path(self, path: str):
        """
        Set the root directory to display.
        
        Args:
            path: Absolute path to the root directory
        """
        if os.path.isdir(path):
            self._root_path = path
            self.model.setRootPath(path)
            root_index = self.model.index(path)
            self.tree_view.setRootIndex(root_index)
            
            # Expand root and select first item
            self.tree_view.expand(root_index)
            
            # Emit signal for the root folder
            self.folder_selected.emit(path)
    
    def _on_folder_clicked(self, index: QModelIndex):
        """Handle folder click."""
        path = self.model.filePath(index)
        if path:
            self.folder_selected.emit(path)
    
    def get_current_folder(self) -> str:
        """Get the currently selected folder path."""
        indexes = self.tree_view.selectedIndexes()
        if indexes:
            return self.model.filePath(indexes[0])
        return self._root_path
