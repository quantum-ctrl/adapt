#!/usr/bin/env python3
"""
ADAPT - Data Browser - Entry Point

A desktop application for browsing and visualizing scientific ARPES data files.
Supports HDF5, IBW, and ZIP file formats.

Usage:
    uv run adapt browser [initial_folder]
"""

import sys
import os

from PySide6.QtWidgets import QApplication

from ADAPT_browser.ui.main_window import MainWindow
from ADAPT_browser.utils.logger import logger


def main(initial_folder=None):
    """Main entry point."""
    # High DPI scaling is enabled by default in Qt6, no need to set attributes
    
    # Create application
    qt_args = [sys.argv[0]]
    if initial_folder is None:
        qt_args.extend(sys.argv[1:])
    app = QApplication(qt_args)
    app.setApplicationName("ADAPT - Data Browser")
    app.setOrganizationName("ADAPT")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    # Handle command line argument for initial folder
    if initial_folder is None and len(sys.argv) > 1:
        initial_folder = sys.argv[1]

    if initial_folder and os.path.isdir(initial_folder):
        logger.info(f"Opening initial folder: {initial_folder}")
        window.dir_panel.set_root_path(initial_folder)
    
    logger.info("Application started")
    
    # Run event loop
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
