"""
Collections Panel - Named, cross-folder groups of favorite files.
"""

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QPushButton, QInputDialog, QMessageBox
)
from PySide6.QtCore import Signal

try:
    from shared.session import get_collections, create_collection, delete_collection
except ImportError:
    def get_collections():
        return {}

    def create_collection(name):
        return False

    def delete_collection(name):
        return False


class CollectionsPanel(QWidget):
    """List of named collections; selecting one shows its files in the file panel."""

    # Emitted when a collection is clicked
    collection_selected = Signal(str)  # collection name

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self.refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        header_frame = QWidget()
        header_frame.setObjectName("PanelHeaderFrame")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(8, 4, 8, 4)

        header = QLabel("⭐ Collections")
        header.setObjectName("PanelHeader")
        header_layout.addWidget(header)
        header_layout.addStretch()

        add_btn = QPushButton("+")
        add_btn.setFixedSize(22, 22)
        add_btn.setToolTip("Create a new collection")
        add_btn.clicked.connect(self._on_add_clicked)
        header_layout.addWidget(add_btn)

        remove_btn = QPushButton("-")
        remove_btn.setFixedSize(22, 22)
        remove_btn.setToolTip("Delete the selected collection")
        remove_btn.clicked.connect(self._on_remove_clicked)
        header_layout.addWidget(remove_btn)

        layout.addWidget(header_frame)

        self.list_widget = QListWidget()
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget)

    def refresh(self):
        """Reload collection names from persisted app state."""
        current = self.list_widget.currentItem()
        current_name = current.text() if current else None

        self.list_widget.clear()
        for name in sorted(get_collections().keys()):
            self.list_widget.addItem(QListWidgetItem(name))

        if current_name:
            for i in range(self.list_widget.count()):
                if self.list_widget.item(i).text() == current_name:
                    self.list_widget.setCurrentRow(i)
                    break

    def _on_add_clicked(self):
        name, ok = QInputDialog.getText(self, "New Collection", "Collection name:")
        name = name.strip()
        if not ok or not name:
            return
        if not create_collection(name):
            QMessageBox.warning(self, "Collection Exists", f"A collection named '{name}' already exists.")
            return
        self.refresh()

    def _on_remove_clicked(self):
        item = self.list_widget.currentItem()
        if not item:
            return
        name = item.text()
        reply = QMessageBox.question(
            self,
            "Delete Collection",
            f"Delete collection '{name}'? This only removes the collection, not the underlying files.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            delete_collection(name)
            self.refresh()

    def _on_item_clicked(self, item: QListWidgetItem):
        self.collection_selected.emit(item.text())
