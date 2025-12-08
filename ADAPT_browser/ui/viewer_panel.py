"""
Viewer Panel - Data visualization using PyQtGraph.
"""

import os
import numpy as np
from typing import Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QSplitter, QTextEdit, QFrame, QGroupBox, QPushButton, QButtonGroup
)
from PySide6.QtCore import Signal, Qt

import pyqtgraph as pg

from core.data_manager import DataResult
from utils.meta_format import format_metadata, format_shape_dtype
from utils.logger import logger


class ViewerPanel(QWidget):
    """
    Panel for displaying 2D/3D data with PyQtGraph.
    Includes slider for 3D slicing and metadata display.
    """
    
    # Emitted when cursor position changes
    cursor_moved = Signal(float, float, float)  # x, y, value
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: Optional[DataResult] = None
        self._current_slice_index = 0
        self._current_colormap = 'bone'
        self._invert_colormap = True
        self._slice_plane = 'XY'  # XY, YZ, or XZ
        self._cached_display_data = None  # Cache for mouse cursor performance
        self._cached_display_dims = (None, None)  # (x_dim, y_dim) for cached data
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the UI components."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Main splitter for viewer and metadata
        self.splitter = QSplitter(Qt.Vertical)
        
        # === Top: Image Viewer ===
        viewer_widget = QWidget()
        viewer_layout = QVBoxLayout(viewer_widget)
        viewer_layout.setContentsMargins(8, 8, 8, 8)
        viewer_layout.setSpacing(4)
        
        # Configure PyQtGraph
        pg.setConfigOptions(imageAxisOrder='row-major')
        
        # Use PlotWidget with ImageItem for proper axis support
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setAspectLocked(False)
        
        # Create ImageItem for displaying data
        self.image_item = pg.ImageItem()
        self.plot_widget.addItem(self.image_item)
        
        # Configure plot appearance
        self.plot_widget.showGrid(x=False, y=False)
        self.plot_widget.getPlotItem().getViewBox().invertY(False)
        
        # Set default colormap
        self._apply_colormap('bone_r')
        
        # Enable mouse tracking for cursor info
        self.plot_widget.scene().sigMouseMoved.connect(self._on_mouse_moved)
        
        viewer_layout.addWidget(self.plot_widget)
        
        # === Slider for 3D data ===
        self.slider_frame = QFrame()
        self.slider_frame.setVisible(False)
        slider_layout = QHBoxLayout(self.slider_frame)
        slider_layout.setContentsMargins(8, 4, 8, 4)
        
        self.slice_label = QLabel("Slice:")
        self.slice_label.setStyleSheet("font-weight: bold;")
        slider_layout.addWidget(self.slice_label)
        
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self._on_slider_changed)
        slider_layout.addWidget(self.slider, 1)
        
        self.slice_value_label = QLabel("0 / 0")
        self.slice_value_label.setMinimumWidth(100)
        slider_layout.addWidget(self.slice_value_label)
        
        viewer_layout.addWidget(self.slider_frame)
        
        # === Slice plane buttons for 3D data ===
        self.plane_frame = QFrame()
        self.plane_frame.setVisible(False)
        plane_layout = QHBoxLayout(self.plane_frame)
        plane_layout.setContentsMargins(8, 4, 8, 4)
        plane_layout.setSpacing(4)
        
        plane_label = QLabel("Slice Plane:")
        plane_label.setStyleSheet("font-weight: bold;")
        plane_layout.addWidget(plane_label)
        
        # Button group for exclusive selection
        self.plane_button_group = QButtonGroup(self)
        
        self.xy_btn = QPushButton("XY")
        self.xy_btn.setCheckable(True)
        self.xy_btn.setChecked(True)
        self.plane_button_group.addButton(self.xy_btn, 0)
        plane_layout.addWidget(self.xy_btn)
        
        self.yz_btn = QPushButton("YZ")
        self.yz_btn.setCheckable(True)
        self.plane_button_group.addButton(self.yz_btn, 1)
        plane_layout.addWidget(self.yz_btn)
        
        self.xz_btn = QPushButton("XZ")
        self.xz_btn.setCheckable(True)
        self.plane_button_group.addButton(self.xz_btn, 2)
        plane_layout.addWidget(self.xz_btn)
        
        plane_layout.addStretch()
        
        self.plane_button_group.idClicked.connect(self._on_plane_changed)
        
        viewer_layout.addWidget(self.plane_frame)
        
        self.splitter.addWidget(viewer_widget)
        
        # Apply initial button styles (will be updated by theme)
        self._current_theme = 'Dark'  # default
        self._update_plane_button_styles()
        
        # === Bottom: Metadata Panel ===
        self.meta_group = QGroupBox("📋 Metadata")
        self.meta_group.setObjectName("MetadataGroup")
        meta_layout = QVBoxLayout(self.meta_group)
        
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        self.meta_text.setObjectName("MetadataText")
        meta_layout.addWidget(self.meta_text)
        
        self.splitter.addWidget(self.meta_group)
        
        # Set splitter sizes (80% viewer, 20% metadata)
        self.splitter.setSizes([400, 100])
        
        layout.addWidget(self.splitter)
        
        # Placeholder when no data
        self._show_placeholder()
    
    def _show_placeholder(self):
        """Show placeholder when no data is loaded."""
        placeholder = np.zeros((100, 100))
        self.image_item.setImage(placeholder)
        self.meta_text.setText("Select a file to view data.")
    
    def set_data(self, data: DataResult):
        """
        Load and display data.
        
        Args:
            data: DataResult from data manager
        """
        self._data = data
        self._current_slice_index = 0
        self._slice_plane = 'XY'  # Reset to default plane
        
        # Handle 3D data
        if data.ndim >= 3:
            # Show slider and plane buttons
            self.slider_frame.setVisible(True)
            self.plane_frame.setVisible(True)
            
            # Reset plane button selection
            self.xy_btn.setChecked(True)
            
            # Setup slider for the slice dimension (last for XY plane)
            self._update_slider_for_plane()
            
            # Display first slice
            self._display_slice(0)
        else:

            # 2D data - hide slider and plane buttons
            self.slider_frame.setVisible(False)
            self.plane_frame.setVisible(False)
            # For 2D, assume standard orientation (d0=Y, d1=X)
            if len(data.dims) >= 2:
                self._display_2d(data.data, data.dims[1], data.dims[0])
            else:
                self._display_2d(data.data)
        
        # Update metadata
        self._update_metadata(data)
        
        # Set axis labels (will be updated by _display_slice for 3D)
        if data.ndim < 3:
            self._update_axis_labels(data)
    
    def _get_plane_dims(self):
        """
        Get dimension indices for current plane.
        Returns (slice_dim_idx, x_dim_idx, y_dim_idx).
        Assuming data shape (d0, d1, d2) -> (Z, Y, X).
        """
        if self._slice_plane == 'YZ':
            # Slice X (d2), View Y(d1) vs Z(d0)
            return 2, 1, 0
        elif self._slice_plane == 'XZ':
            # Slice Y (d1), View X(d2) vs Z(d0)
            return 1, 2, 0
        else: # XY
            # Slice Z (d0), View X(d2) vs Y(d1)
            return 0, 2, 1

    def _update_slider_for_plane(self):
        """Update slider range and label for current plane."""
        if self._data is None or self._data.ndim < 3:
            return
            
        slice_dim_idx, _, _ = self._get_plane_dims()
        dim_size = self._data.shape[slice_dim_idx]
        
        self.slider.blockSignals(True)
        self.slider.setMaximum(dim_size - 1)
        self.slider.setValue(self._current_slice_index)
        self.slider.blockSignals(False)
        
        # Update label
        dim_name = self._data.dims[slice_dim_idx]
        self.slice_label.setText(f"{dim_name}:")
        self._update_slice_label()

    def _on_plane_changed(self, id):
        """Handle plane button click."""
        planes = {0: 'XY', 1: 'YZ', 2: 'XZ'}
        new_plane = planes.get(id, 'XY')
        
        if new_plane != self._slice_plane:
            self._slice_plane = new_plane
            # Reset slice index to middle or 0? Let's keep 0 for now
            self._current_slice_index = 0
            self._update_slider_for_plane()
            self._display_slice(self._current_slice_index)

    def _display_2d(self, data_2d: np.ndarray, x_dim: str = None, y_dim: str = None):
        """Display a 2D array with real axis coordinates."""
        # Handle NaN/Inf
        display_data = np.nan_to_num(data_2d, nan=0, posinf=0, neginf=0)
        
        # Auto-scale using percentile for better contrast
        vmin = np.percentile(display_data, 1)
        vmax = np.percentile(display_data, 99)
        
        # Set image with real coordinates using transform
        if self._data is not None and x_dim and y_dim:
            # Get coordinates for axes
            x_coords = self._data.coords.get(x_dim)
            y_coords = self._data.coords.get(y_dim)
            
            if x_coords is not None and y_coords is not None:
                # Calculate transform: position and scale
                x_min, x_max = x_coords[0], x_coords[-1]
                y_min, y_max = y_coords[0], y_coords[-1]
                
                # Scale factors
                x_scale = (x_max - x_min) / max(display_data.shape[1] - 1, 1)
                y_scale = (y_max - y_min) / max(display_data.shape[0] - 1, 1)
                
                # Set image with transform
                self.image_item.setImage(display_data, levels=(vmin, vmax))
                self.image_item.setRect(x_min, y_min, x_max - x_min, y_max - y_min)
                
                # Auto-range to fit data
                self.plot_widget.setXRange(x_min, x_max, padding=0.02)
                self.plot_widget.setYRange(y_min, y_max, padding=0.02)
            else:
                # Fallback: no coordinates available
                self.image_item.setImage(display_data, levels=(vmin, vmax))
                self.image_item.setRect(0, 0, display_data.shape[1], display_data.shape[0])
                self.plot_widget.autoRange()
        else:
            # Fallback for data without proper structure
            self.image_item.setImage(display_data, levels=(vmin, vmax))
            self.image_item.setRect(0, 0, display_data.shape[1], display_data.shape[0])
            self.plot_widget.autoRange()
    
    def _display_slice(self, slice_idx: int):
        """Display a specific slice of 3D data."""
        if self._data is None or self._data.ndim < 3:
            return
        
        slice_dim_idx, x_dim_idx, y_dim_idx = self._get_plane_dims()
        
        # Ensure slice index is valid
        if slice_idx >= self._data.shape[slice_dim_idx]:
            slice_idx = self._data.shape[slice_dim_idx] - 1
            self._current_slice_index = slice_idx
            self.slider.blockSignals(True)
            self.slider.setValue(slice_idx)
            self.slider.blockSignals(False)
            
        # Slice the data
        # We need to construct a slicer tuple
        slicer = [slice(None)] * self._data.ndim
        slicer[slice_dim_idx] = slice_idx
        slice_data = self._data.data[tuple(slicer)]
        
        # Now we have a 2D array, but we need to transpose it if necessary
        # to match (Y, X) for display
        # The remaining dims are (d_a, d_b) where a < b
        # We want y_dim_idx to be axis 0, x_dim_idx to be axis 1
        
        # Map original indices to 0 and 1
        remaining_dims = [i for i in range(3) if i != slice_dim_idx]
        # current shape is (dim[remaining_dims[0]], dim[remaining_dims[1]])
        
        # We want the axis corresponding to y_dim_idx to be 0
        # and x_dim_idx to be 1
        
        target_y_axis = remaining_dims.index(y_dim_idx)
        target_x_axis = remaining_dims.index(x_dim_idx)
        
        if target_y_axis != 0:
            slice_data = slice_data.T
            
        x_dim_name = self._data.dims[x_dim_idx]
        y_dim_name = self._data.dims[y_dim_idx]
        
        self._display_2d(slice_data, x_dim_name, y_dim_name)
        
        # Cache the display data for mouse cursor performance
        self._cached_display_data = slice_data
        self._cached_display_dims = (x_dim_name, y_dim_name)
        
        # Update labels
        self._update_axis_labels(self._data, x_dim_name, y_dim_name)
    
    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        self._current_slice_index = value
        self._update_slice_label()
        self._display_slice(value)
    
    def _update_slice_label(self):
        """Update the slice value label."""
        if self._data is None or self._data.ndim < 3:
            return
        
        slice_dim_idx, _, _ = self._get_plane_dims()
        dim_name = self._data.dims[slice_dim_idx]
        
        if dim_name in self._data.coords:
            coord_val = self._data.coords[dim_name][self._current_slice_index]
            total = self._data.shape[slice_dim_idx]
            self.slice_value_label.setText(f"{coord_val:.2f} ({self._current_slice_index + 1}/{total})")
        else:
            total = self._data.shape[slice_dim_idx]
            self.slice_value_label.setText(f"{self._current_slice_index + 1} / {total}")
    
    def _update_metadata(self, data: DataResult):
        """Update the metadata display."""
        info_lines = [
            f"File: {os.path.basename(data.filepath)}",
            format_shape_dtype(data.data),
            f"Dims: {data.dims}",
            "",
        ]
        
        # Format metadata
        meta_str = format_metadata(data.meta)
        info_lines.append(meta_str)
        
        self.meta_text.setText("\n".join(info_lines))
    
    def _update_axis_labels(self, data: DataResult, x_label: str = None, y_label: str = None):
        """Update axis labels based on data dimensions."""
        plot_item = self.plot_widget.getPlotItem()
        
        if x_label is None and len(data.dims) >= 2:
            x_label = data.dims[1]
        if y_label is None and len(data.dims) >= 2:
            y_label = data.dims[0]
            
        if x_label and y_label:
            # Add units if available in metadata
            x_unit = data.meta.get(f'{x_label}_unit', '')
            y_unit = data.meta.get(f'{y_label}_unit', '')
            
            if x_unit:
                x_label = f"{x_label} ({x_unit})"
            if y_unit:
                y_label = f"{y_label} ({y_unit})"
            
            plot_item.setLabel('bottom', x_label)
            plot_item.setLabel('left', y_label)
        else:
            plot_item.setLabel('bottom', 'X')
            plot_item.setLabel('left', 'Y')
    
    def set_colormap(self, cmap_name: str, invert: bool = False):
        """Set the colormap for the image display."""
        self._current_colormap = cmap_name
        self._invert_colormap = invert
        self._apply_colormap(cmap_name, invert)
        
        # Refresh display if data is loaded
        if self._data is not None:
            if self._data.ndim >= 3:
                self._display_slice(self._current_slice_index)
            else:
                # For 2D, assume standard orientation
                if len(self._data.dims) >= 2:
                    self._display_2d(self._data.data, self._data.dims[1], self._data.dims[0])
                else:
                    self._display_2d(self._data.data)
    
    def _apply_colormap(self, cmap_name: str, invert: bool = False):
        """Apply a colormap to the image item."""
        try:
            # Use matplotlib colormaps via pyqtgraph
            from matplotlib import colormaps
            cmap = colormaps.get_cmap(cmap_name)
            
            # Create lookup table from colormap (256 colors)
            if invert:
                lut = (cmap(np.linspace(1, 0, 256)) * 255).astype(np.uint8)
            else:
                lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)
                
            self.image_item.setLookupTable(lut)
        except Exception as e:
            # Fallback to default
            logger.warning(f"Failed to apply colormap {cmap_name}: {e}")
    
    def _on_mouse_moved(self, pos):
        """Handle mouse movement for cursor info."""
        if self._data is None:
            return
        
        # Map position to data coordinates
        if self.image_item is None:
            return
        
        mouse_point = self.plot_widget.getPlotItem().vb.mapSceneToView(pos)
        real_x, real_y = mouse_point.x(), mouse_point.y()
        
        # Get current displayed data - use cache if available for performance
        if self._data.ndim >= 3:
            # Use cached display data to avoid expensive re-slicing on every mouse move
            if self._cached_display_data is not None:
                display_data = self._cached_display_data
                x_dim, y_dim = self._cached_display_dims
            else:
                # Fallback: compute slice (shouldn't happen in normal use)
                slice_dim_idx, x_dim_idx, y_dim_idx = self._get_plane_dims()
                
                slicer = [slice(None)] * self._data.ndim
                slicer[slice_dim_idx] = self._current_slice_index
                display_data = self._data.data[tuple(slicer)]
                
                remaining_dims = [i for i in range(3) if i != slice_dim_idx]
                target_y_axis = remaining_dims.index(y_dim_idx)
                if target_y_axis != 0:
                    display_data = display_data.T
                    
                x_dim = self._data.dims[x_dim_idx]
                y_dim = self._data.dims[y_dim_idx]
        else:
            display_data = self._data.data
            if len(self._data.dims) >= 2:
                x_dim = self._data.dims[1]
                y_dim = self._data.dims[0]
            else:
                x_dim, y_dim = None, None
        
        # Convert real coordinates back to array indices
        if x_dim and y_dim:
            x_coords = self._data.coords.get(x_dim)
            y_coords = self._data.coords.get(y_dim)
            
            if x_coords is not None and y_coords is not None and len(x_coords) > 1 and len(y_coords) > 1:
                # Find nearest indices
                x_idx = np.searchsorted(x_coords, real_x)
                y_idx = np.searchsorted(y_coords, real_y)
                
                # Clamp to valid range
                x_idx = max(0, min(x_idx, len(x_coords) - 1))
                y_idx = max(0, min(y_idx, len(y_coords) - 1))
                
                # Check if within data bounds
                if 0 <= y_idx < display_data.shape[0] and 0 <= x_idx < display_data.shape[1]:
                    value = display_data[y_idx, x_idx]
                    self.cursor_moved.emit(real_x, real_y, float(value))
            else:
                # Fallback to integer indices
                x_idx, y_idx = int(real_x), int(real_y)
                if 0 <= y_idx < display_data.shape[0] and 0 <= x_idx < display_data.shape[1]:
                    value = display_data[y_idx, x_idx]
                    self.cursor_moved.emit(real_x, real_y, float(value))
        else:
            # Simple case
            x_idx, y_idx = int(real_x), int(real_y)
            if 0 <= y_idx < display_data.shape[0] and 0 <= x_idx < display_data.shape[1]:
                value = display_data[y_idx, x_idx]
                self.cursor_moved.emit(real_x, real_y, float(value))
    
    def clear(self):
        """Clear the display."""
        self._data = None
        self.slider_frame.setVisible(False)
        self._show_placeholder()

    def _update_plane_button_styles(self):
        """Update plane button styles based on current theme."""
        if self._current_theme == 'Light':
            button_style = """
                QPushButton {
                    background-color: #e0e0e0;
                    border: 1px solid #cccccc;
                    border-radius: 4px;
                    padding: 4px 12px;
                    color: #1e1e1e;
                    font-weight: bold;
                    min-width: 40px;
                }
                QPushButton:hover {
                    background-color: #d0d0d0;
                }
                QPushButton:checked {
                    background-color: #0078d7;
                    border-color: #0078d7;
                    color: white;
                }
            """
        else:  # Dark
            button_style = """
                QPushButton {
                    background-color: #3c3c3c;
                    border: 1px solid #4c4c4c;
                    border-radius: 4px;
                    padding: 4px 12px;
                    color: #d4d4d4;
                    font-weight: bold;
                    min-width: 40px;
                }
                QPushButton:hover {
                    background-color: #4c4c4c;
                }
                QPushButton:checked {
                    background-color: #094771;
                    border-color: #0e639c;
                }
            """
        
        self.xy_btn.setStyleSheet(button_style)
        self.yz_btn.setStyleSheet(button_style)
        self.xz_btn.setStyleSheet(button_style)

    def update_theme(self, theme_name: str):
        """
        Update viewer colors based on theme.
        
        Args:
            theme_name: 'Dark' or 'Light'
        """
        self._current_theme = theme_name
        self._update_plane_button_styles()
        
        if theme_name == 'Light':
            bg_color = '#ffffff'
            fg_color = '#000000'
            axis_pen = '#000000'
            text_pen = '#000000'
        else:  # Dark
            bg_color = '#1e1e1e'
            fg_color = '#cccccc'
            axis_pen = '#cccccc'
            text_pen = '#cccccc'
            
        self.plot_widget.setBackground(bg_color)
        
        # Update axes
        plot_item = self.plot_widget.getPlotItem()
        
        # Update axis pens
        for axis in ['left', 'bottom']:
            ax = plot_item.getAxis(axis)
            ax.setPen(axis_pen)
            ax.setTextPen(text_pen)
            
        # Update title if any (not currently used but good practice)
        plot_item.setTitle(None)

