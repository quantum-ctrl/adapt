#!/usr/bin/env python3
"""
ADAPT - Data Browser - Entry Point

A desktop application for browsing and visualizing scientific ARPES data files.
Supports HDF5, IBW, and ZIP file formats.

Usage:
    python app.py [initial_folder]
"""

import sys
import os

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow
from utils.logger import logger


def main():
    """Main entry point."""
    # High DPI scaling is enabled by default in Qt6, no need to set attributes
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("ADAPT - Data Browser")
    app.setOrganizationName("ADAPT")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Handle command line argument for initial folder
    if len(sys.argv) > 1:
        initial_folder = sys.argv[1]
        if os.path.isdir(initial_folder):
            logger.info(f"Opening initial folder: {initial_folder}")
            window.dir_panel.set_root_path(initial_folder)
    
    logger.info("Application started")
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
