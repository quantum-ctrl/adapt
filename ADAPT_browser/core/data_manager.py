"""
ADAPT - ARPES Data Analysis & Processing Tool - Data Manager - Unified interface for loading scientific data files.

Integrates existing loaders (HDF5, IBW, ZIP) and provides a consistent
output format for the UI layer.
"""

import os
import sys
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

from PySide6.QtCore import QObject, Signal, QThread

from utils.logger import logger

# Add shared loaders to path
_shared_loaders_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'shared'))
if _shared_loaders_path not in sys.path:
    sys.path.insert(0, _shared_loaders_path)


@dataclass
class DataResult:
    """Container for loaded data."""
    data: np.ndarray          # N-D array
    dims: list                # Dimension labels like ["energy", "angle", "scan"]
    coords: Dict[str, np.ndarray]  # Coordinate arrays for each dimension
    meta: Dict[str, Any]      # Metadata
    filepath: str             # Source file path
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    @property
    def shape(self) -> tuple:
        return self.data.shape
    
    @property 
    def dtype(self):
        return self.data.dtype


class LoaderWorker(QObject):
    """Worker for loading files in a background thread."""
    
    finished = Signal(object)  # Emits DataResult or None
    error = Signal(str)        # Emits error message
    progress = Signal(str)     # Emits status message
    
    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
    
    def run(self):
        """Load the file and emit result."""
        try:
            self.progress.emit(f"Loading {os.path.basename(self.filepath)}...")
            result = DataManager.load_file_sync(self.filepath)
            self.finished.emit(result)
        except Exception as e:
            logger.error(f"Failed to load {self.filepath}: {e}")
            self.error.emit(str(e))
            self.finished.emit(None)


class DataManager(QObject):
    """
    Manages data loading with support for multiple file formats.
    Uses background threading for large files.
    """
    
    # Signals
    loading_started = Signal(str)   # filepath
    loading_finished = Signal(object)  # DataResult or None
    loading_error = Signal(str)     # error message
    loading_progress = Signal(str)  # status message
    
    # Supported extensions
    SUPPORTED_EXTENSIONS = {
        'h5': 'HDF5',
        'hdf5': 'HDF5', 
        'ibw': 'IBW',
        'zip': 'ZIP',
        'pxt': 'PXT',
        'pxp': 'PXP'
    }
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._worker = None
        self._thread = None
        self._current_data: Optional[DataResult] = None
    
    @property
    def current_data(self) -> Optional[DataResult]:
        """Get the currently loaded data."""
        return self._current_data
    
    @classmethod
    def is_supported(cls, filepath: str) -> bool:
        """Check if a file is supported."""
        ext = os.path.splitext(filepath)[1].lower().lstrip('.')
        return ext in cls.SUPPORTED_EXTENSIONS
    
    @classmethod
    def get_file_type(cls, filepath: str) -> Optional[str]:
        """Get the file type label."""
        ext = os.path.splitext(filepath)[1].lower().lstrip('.')
        return cls.SUPPORTED_EXTENSIONS.get(ext)
    
    @staticmethod
    def load_file_sync(filepath: str) -> DataResult:
        """
        Load a file synchronously. 
        
        Args:
            filepath: Path to the file
            
        Returns:
            DataResult with normalized data
            
        Raises:
            ValueError: If file type is not supported
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        ext = os.path.splitext(filepath)[1].lower().lstrip('.')
        
        # Import loaders lazily
        if ext in ('h5', 'hdf5'):
            # Try ADRESS loader first, then SIS
            try:
                from loaders import load_adress
                da = load_adress(filepath)
            except Exception as e1:
                try:
                    from loaders import load_sis
                    da = load_sis(filepath)
                except Exception as e2:
                    raise ValueError(f"Failed to load HDF5 file.\nADRESS: {e1}\nSIS: {e2}")
                    
        elif ext == 'ibw':
            from loaders import load_ibw
            da = load_ibw(filepath)
            
        elif ext == 'zip':
            from loaders import load_ses
            da = load_ses(filepath)

        elif ext in ('pxt', 'pxp'):
            from loaders import load_pxt
            da = load_pxt(filepath)
            
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Convert xarray.DataArray to DataResult
        return DataManager._convert_xarray(da, filepath)
    
    @staticmethod
    def _convert_xarray(da, filepath: str) -> DataResult:
        """Convert xarray.DataArray to DataResult."""
        # Extract data
        data = da.values
        
        # Extract dimensions
        dims = list(da.dims)
        
        # Extract coordinates
        coords = {}
        for dim in dims:
            if dim in da.coords:
                coords[dim] = da.coords[dim].values
            else:
                coords[dim] = np.arange(da.sizes[dim])
        
        # Extract metadata from attrs
        meta = dict(da.attrs)
        
        return DataResult(
            data=data,
            dims=dims,
            coords=coords,
            meta=meta,
            filepath=filepath
        )
    
    def load_file_async(self, filepath: str):
        """
        Load a file asynchronously in a background thread.
        
        Emits loading_finished signal when done.
        """
        # Cancel any existing load
        self.cancel_loading()
        
        self.loading_started.emit(filepath)
        
        # Create worker and thread
        self._thread = QThread()
        self._worker = LoaderWorker(filepath)
        self._worker.moveToThread(self._thread)
        
        # Connect signals
        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_load_finished)
        self._worker.error.connect(self.loading_error.emit)
        self._worker.progress.connect(self.loading_progress.emit)
        self._worker.finished.connect(self._thread.quit)
        
        # Clean up worker and thread after thread finishes (not before!)
        self._thread.finished.connect(self._cleanup_thread)
        
        # Start loading
        self._thread.start()
    
    def _on_load_finished(self, result: Optional[DataResult]):
        """Handle load completion."""
        self._current_data = result
        self.loading_finished.emit(result)
    
    def _cleanup_thread(self):
        """Clean up thread and worker after thread has finished."""
        if self._worker:
            self._worker.deleteLater()
            self._worker = None
        if self._thread:
            self._thread.deleteLater()
            self._thread = None
    
    def cancel_loading(self):
        """Cancel any ongoing loading operation safely."""
        if self._thread is None:
            return
        
        if self._thread.isRunning():
            # Disconnect signals first to prevent callbacks during cleanup
            if self._worker:
                try:
                    self._worker.finished.disconnect()
                except (RuntimeError, TypeError):
                    pass
                try:
                    self._worker.error.disconnect()
                except (RuntimeError, TypeError):
                    pass
                try:
                    self._worker.progress.disconnect()
                except (RuntimeError, TypeError):
                    pass
            
            # Also disconnect thread.finished to prevent double cleanup
            try:
                self._thread.finished.disconnect(self._cleanup_thread)
            except (RuntimeError, TypeError):
                pass
            
            # Request thread to quit
            self._thread.quit()
            
            # Wait with timeout - blocking file I/O may take time
            if not self._thread.wait(5000):  # 5 second timeout
                # Thread didn't finish, terminate forcefully
                logger.warning("Force terminating stuck loading thread")
                self._thread.terminate()
                self._thread.wait(2000)
        
        # Clean up references
        if self._worker:
            try:
                self._worker.deleteLater()
            except RuntimeError:
                pass
            self._worker = None
        
        if self._thread:
            try:
                self._thread.deleteLater()
            except RuntimeError:
                pass
            self._thread = None


def get_supported_extensions() -> list:
    """Get list of supported file extensions with dots."""
    return [f".{ext}" for ext in DataManager.SUPPORTED_EXTENSIONS.keys()]


def filter_files_by_type(files: list, file_type: str) -> list:
    """
    Filter a list of files by type.
    
    Args:
        files: List of filenames
        file_type: One of 'All', 'HDF5', 'IBW', 'ZIP'
        
    Returns:
        Filtered list of filenames
    """
    if file_type == 'All':
        # Return all supported files
        return [f for f in files if DataManager.is_supported(f)]
    
    type_extensions = {
        'HDF5': ['h5', 'hdf5'],
        'IBW': ['ibw'],
        'ZIP': ['zip'],
        'PXT': ['pxt', 'pxp']
    }
    
    allowed = type_extensions.get(file_type, [])
    return [f for f in files 
            if os.path.splitext(f)[1].lower().lstrip('.') in allowed]
