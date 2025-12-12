"""
Visualization Module for ARPES Data

This module provides common plotting functions for ARPES data visualization.

Usage:
------
    from processing.visualization import plot_3d_data
    
    # Plot 3 orthogonal views of 3D data
    fig = plot_3d_data(data, title_prefix="My Data - ")
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure
import xarray as xr
from typing import Optional, Tuple, Union


def plot_3d_data(
    data: xr.DataArray,
    title_prefix: str = "",
    figsize: Tuple[int, int] = (15, 4),
    cmap: str = 'bone_r',
    use_kz_coords: bool = False,
    crosshair_pos: Optional[Tuple[float, float, float]] = None,
    slice_pos: Optional[Tuple[float, float, float]] = None,
    interactive: bool = True
) -> matplotlib.figure.Figure:
    """
    Plot 3 orthogonal views of 3D data.
    
    For ARPES data with dimensions (Eb, theta, scan):
    - XY view: Energy vs Theta/k (summed or sliced over scan)
    - YZ view: Theta/k vs Scan/kz (summed or sliced over energy)
    - XZ view: Energy vs Scan/kz (summed or sliced over theta/k)
    
    Parameters
    ----------
    data : xarray.DataArray
        3D data with dimensions (energy, angle/k, scan/hv)
    title_prefix : str
        Prefix for subplot titles
    figsize : tuple
        Figure size
    cmap : str
        Colormap name (default: 'bone_r')
    use_kz_coords : bool, optional
        If True and 'kz_full' coordinate exists, use proper 2D kz coordinates 
        for the YZ view (k vs kz). This correctly displays curved iso-hv lines
        where kz decreases with increasing |k| (off-normal emission).
        Default is False.
    crosshair_pos : tuple of 3 floats, optional
        (E, k/angle, scan/kz) position to draw crosshair lines.
        If None, no crosshairs are drawn. Default is None.
    slice_pos : tuple of 3 floats, optional
        (E, k/angle, scan/kz) position for slice mode. When provided:
        - XY shows slice at scan=slice_pos[2]
        - YZ shows slice at energy=slice_pos[0]
        - XZ shows slice at angle/k=slice_pos[1]
        Crosshairs are automatically drawn to show slice positions.
        If None, views show sums (default behavior). Default is None.
    interactive : bool, optional
        If True, show sliders below plots to control colormap min/max.
        Default is True.
    
    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    from matplotlib.widgets import Slider
    
    dims = list(data.dims)
    
    # Adjust figure height for sliders
    if interactive:
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0], figsize[1] + 0.8))
        plt.subplots_adjust(bottom=0.18)
    else:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Get dimension names
    energy_dim = dims[0]  # 'energy'
    angle_dim = dims[1]   # 'angle' or 'k'
    scan_dim = dims[2]    # 'scan' or 'hv'
    
    # Get coordinate arrays
    e_coords = data.coords[energy_dim].values
    k_coords = data.coords[angle_dim].values
    z_coords = data.coords[scan_dim].values
    
    # Determine if using slice mode
    use_slices = slice_pos is not None
    if use_slices:
        e_slice, k_slice, z_slice = slice_pos
        # Find nearest indices
        e_idx = np.argmin(np.abs(e_coords - e_slice))
        k_idx = np.argmin(np.abs(k_coords - k_slice))
        z_idx = np.argmin(np.abs(z_coords - z_slice))
        # Get actual coordinate values at indices
        e_val = e_coords[e_idx]
        k_val = k_coords[k_idx]
        z_val = z_coords[z_idx]
    
    # XY View: Energy vs Theta/k
    if use_slices:
        xy_data = data.isel({scan_dim: z_idx})
        xy_title = f'{title_prefix}E vs {angle_dim} @ {scan_dim}={z_val:.2f}'
    else:
        xy_data = data.sum(dim=scan_dim)
        xy_title = f'{title_prefix}XY: {energy_dim} vs {angle_dim}'
    
    im0 = axes[0].pcolormesh(
        k_coords, e_coords, xy_data.values,
        shading='auto', cmap=cmap
    )
    x_label = f'{angle_dim}' + (' (Å⁻¹)' if 'k' in angle_dim else ' (deg)')
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(f'{energy_dim} (eV)')
    axes[0].set_title(xy_title)
    plt.colorbar(im0, ax=axes[0], label='Intensity')
    
    # YZ View: Theta/k vs Scan/kz
    if use_slices:
        yz_data = data.isel({energy_dim: e_idx})
        yz_title = f'{title_prefix}{angle_dim} vs {scan_dim} @ E={e_val:.3f}'
    else:
        yz_data = data.sum(dim=energy_dim)
        yz_title = f'{title_prefix}YZ: {angle_dim} vs {scan_dim}'
    
    # Check if we should use kz coordinates for proper curved display
    if use_kz_coords and 'kz_full' in data.coords:
        # kz_full has shape (energy, k, hv) - average over energy for YZ plot
        kz_full = data.coords['kz_full'].values
        if use_slices:
            kz_2d = kz_full[e_idx, :, :]  # Use specific energy slice
        else:
            kz_2d = np.nanmean(kz_full, axis=0)  # shape: (N_k, N_hv)
        
        # Handle NaN values - pcolormesh cannot accept them in coordinates
        # Fill NaN with nearest valid values along each column (hv)
        for j in range(kz_2d.shape[1]):
            col = kz_2d[:, j].copy()
            mask = np.isnan(col)
            if np.any(mask) and not np.all(mask):
                # Interpolate NaN values
                valid_idx = np.where(~mask)[0]
                col[mask] = np.interp(np.where(mask)[0], valid_idx, col[valid_idx])
                kz_2d[:, j] = col
        
        # If still has NaN, replace with edge values
        kz_2d = np.nan_to_num(kz_2d, nan=np.nanmean(kz_2d))
        
        # Create 2D k grid (same k for each hv column)
        k_2d = np.tile(k_coords.reshape(-1, 1), (1, kz_2d.shape[1]))
        
        # Use pcolormesh with 2D coordinates - shows curved iso-hv lines
        im1 = axes[1].pcolormesh(
            kz_2d, k_2d, yz_data.values,
            shading='auto',
            cmap=cmap
        )
        axes[1].set_xlabel('kz (Å⁻¹)')
        axes[1].set_ylabel(x_label)
        axes[1].set_title(yz_title)
    else:
        # Standard rectangular grid
        im1 = axes[1].pcolormesh(
            z_coords, k_coords, yz_data.values,
            shading='auto',
            cmap=cmap
        )
        z_label = 'kz (Å⁻¹)' if (use_kz_coords or 'kz' in str(data.coords.get(scan_dim, ''))) else scan_dim
        axes[1].set_xlabel(z_label)
        axes[1].set_ylabel(x_label)
        axes[1].set_title(yz_title)
    plt.colorbar(im1, ax=axes[1], label='Intensity')
    
    # XZ View: Energy vs Scan/kz
    if use_slices:
        xz_data = data.isel({angle_dim: k_idx})
        xz_title = f'{title_prefix}E vs {scan_dim} @ {angle_dim}={k_val:.3f}'
    else:
        xz_data = data.sum(dim=angle_dim)
        xz_title = f'{title_prefix}XZ: {energy_dim} vs {scan_dim}'
    
    if use_kz_coords and 'kz_full' in data.coords:
        # For XZ, use kz at the k position (or k~0 for sum mode)
        kz_full = data.coords['kz_full'].values
        if use_slices:
            kz_1d = np.nanmean(kz_full[:, k_idx, :], axis=0)
        else:
            k_center_idx = np.argmin(np.abs(k_coords))
            kz_1d = np.nanmean(kz_full[:, k_center_idx, :], axis=0)
        
        im2 = axes[2].pcolormesh(
            kz_1d, e_coords, xz_data.values,
            shading='auto', cmap=cmap
        )
        axes[2].set_xlabel('kz (Å⁻¹)')
    else:
        im2 = axes[2].pcolormesh(
            z_coords, e_coords, xz_data.values,
            shading='auto', cmap=cmap
        )
        z_label = 'kz (Å⁻¹)' if (use_kz_coords or 'kz' in str(data.coords.get(scan_dim, ''))) else scan_dim
        axes[2].set_xlabel(z_label)
    
    axes[2].set_ylabel(f'{energy_dim} (eV)')
    axes[2].set_title(xz_title)
    plt.colorbar(im2, ax=axes[2], label='Intensity')
    
    # Draw crosshair lines - in slice mode, auto-draw at slice positions
    if use_slices:
        line_style = {'color': 'white', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.9}
        
        # XY view: show E and k positions from other slices
        axes[0].axhline(y=e_val, **line_style)  # horizontal line at E (YZ slice position)
        axes[0].axvline(x=k_val, **line_style)  # vertical line at k (XZ slice position)
        
        # YZ view: show k and z positions from other slices
        axes[1].axhline(y=k_val, **line_style)  # horizontal line at k (XZ slice position)
        axes[1].axvline(x=z_val, **line_style)  # vertical line at z (XY slice position)
        
        # XZ view: show E and z positions from other slices
        axes[2].axhline(y=e_val, **line_style)  # horizontal line at E (YZ slice position)
        axes[2].axvline(x=z_val, **line_style)  # vertical line at z (XY slice position)
    elif crosshair_pos is not None:
        # Manual crosshair position
        e_pos, k_pos, z_pos = crosshair_pos
        line_style = {'color': 'white', 'linestyle': '--', 'linewidth': 1.5, 'alpha': 0.9}
        
        # XY view (E vs k): show E and k positions
        axes[0].axhline(y=e_pos, **line_style)
        axes[0].axvline(x=k_pos, **line_style)
        
        # YZ view (k vs z): show k and z positions
        axes[1].axhline(y=k_pos, **line_style)
        axes[1].axvline(x=z_pos, **line_style)
        
        # XZ view (E vs z): show E and z positions
        axes[2].axhline(y=e_pos, **line_style)
        axes[2].axvline(x=z_pos, **line_style)
    
    # Add interactive contrast sliders below plots
    if interactive:
        # Get global data range from all views
        data_min = min(np.nanmin(xy_data.values), np.nanmin(yz_data.values), np.nanmin(xz_data.values))
        data_max = max(np.nanmax(xy_data.values), np.nanmax(yz_data.values), np.nanmax(xz_data.values))
        
        ax_vmin = fig.add_axes([0.15, 0.06, 0.70, 0.02])
        ax_vmax = fig.add_axes([0.15, 0.02, 0.70, 0.02])
        
        slider_vmin = Slider(ax_vmin, 'Min', data_min, data_max, valinit=data_min)
        slider_vmax = Slider(ax_vmax, 'Max', data_min, data_max, valinit=data_max)
        
        def update(val):
            vmin = slider_vmin.val
            vmax = slider_vmax.val
            im0.set_clim(vmin=vmin, vmax=vmax)
            im1.set_clim(vmin=vmin, vmax=vmax)
            im2.set_clim(vmin=vmin, vmax=vmax)
            fig.canvas.draw_idle()
        
        slider_vmin.on_changed(update)
        slider_vmax.on_changed(update)
        
        # Store sliders in figure to prevent garbage collection
        fig._sliders = (slider_vmin, slider_vmax)
    
    if not interactive:
        plt.tight_layout()
    return fig


def plot_2d_data(data, title="", figsize=(14, 7), cmap='bone_r', show_crosshair=True, 
                 interactive=True, show_enhancement=True):
    """
    Plot 2D data with optional interactive enhancement preview.
    
    Parameters
    ----------
    data : xarray.DataArray
        2D data
    title : str
        Plot title
    figsize : tuple
        Figure size (default: (14, 7) for side-by-side view with controls)
    cmap : str
        Colormap name (default: 'bone_r')
    show_crosshair : bool
        If True, show white dashed lines at Eb=0 and angle/k=0 (default: True)
    interactive : bool
        If True, show sliders to control colormap min/max (default: True)
    show_enhancement : bool
        If True, show enhancement dropdown and side-by-side view (default: True)
        
    Note
    ----
    For interactive sliders to work in Jupyter notebooks, use:
        %matplotlib widget
    or:
        %matplotlib tk
        
    Enhancement Features:
        - Select enhancement method from dropdown (BG Sub, CLAHE, Curvature, etc.)
        - Enhanced image appears on the right ONLY after selecting a method
        - Original image stays on the left (unchanged)
        - Colormap dropdown controls both images
        - Invert toggle to reverse colormap
        - Separate contrast sliders for original and enhanced images
    """
    from matplotlib.widgets import Slider, RadioButtons, CheckButtons, RangeSlider
    
    # Import enhancement functions
    from processing.enhancement import (
        clahe, histogram_equalize, gamma_correction,
        curvature_luo, curvature_second_derivative
    )
    from processing.background import shirley_background
    
    dims = list(data.dims)
    if len(dims) != 2:
        # Try to squeeze if dimensions are 1
        data = data.squeeze()
        dims = list(data.dims)
        if len(dims) != 2:
            raise ValueError(f"Data must be 2D, but has dimensions: {dims}")
    
    y_dim, x_dim = dims[0], dims[1]
    
    # Store original data (never modify)
    original_values = data.values.copy()
    x_coords = data.coords[x_dim].values
    y_coords = data.coords[y_dim].values
    
    # Get data range for sliders
    data_min = float(np.nanmin(original_values))
    data_max = float(np.nanmax(original_values))
    
    # Set axis labels
    x_label = f'{x_dim}'
    y_label = f'{y_dim}'
    if x_dim in ['theta', 'phi', 'angle']:
        x_label += ' (deg)'
    elif 'k' in x_dim:
        x_label += ' (Å⁻¹)'
    if y_dim in ['energy', 'Eb', 'binding_energy']:
        y_label += ' (eV)'
    
    # Create figure layout
    if show_enhancement and interactive:
        # Side-by-side layout with controls
        fig = plt.figure(figsize=figsize)
        
        # Original image (left side)
        ax_orig = fig.add_axes([0.06, 0.32, 0.38, 0.60])
        
        # Enhanced image (right side, initially hidden)
        ax_enh = fig.add_axes([0.52, 0.32, 0.38, 0.60])
        ax_enh.set_visible(False)  # Hidden until enhancement selected
        
        # === LEFT SIDE: Original image contrast RangeSlider ===
        ax_orig_contrast = fig.add_axes([0.15, 0.22, 0.2, 0.03])
        
        # === RIGHT SIDE: Enhanced image contrast RangeSlider ===
        ax_enh_contrast = fig.add_axes([0.6, 0.22, 0.2, 0.03])
        ax_enh_contrast.set_visible(False)
        
        # Parameter slider (right side below enhanced image, for CLAHE etc.)
        ax_param = fig.add_axes([0.6, 0.19, 0.2, 0.03])
        ax_param.set_visible(False)
        
        # Enhancement radio buttons (below enhanced image)
        ax_radio = fig.add_axes([0.52, 0.02, 0.38, 0.14])
        ax_radio.set_facecolor('#f8f8f8')
        
        # Colormap dropdown (bottom left)
        ax_cmap = fig.add_axes([0.06, 0.02, 0.12, 0.14])
        ax_cmap.set_facecolor('#f0f0f0')
        
        # Invert toggle (bottom left, next to colormap)
        ax_invert = fig.add_axes([0.20, 0.08, 0.10, 0.06])
        ax_invert.set_facecolor('#f0f0f0')
        
    elif interactive:
        fig, ax_orig = plt.subplots(figsize=(figsize[0]/2, figsize[1]))
        plt.subplots_adjust(bottom=0.25)
        ax_enh = None
    else:
        fig, ax_orig = plt.subplots(figsize=(figsize[0]/2, figsize[1]))
        ax_enh = None
    
    # Plot original image (no colorbar)
    im_orig = ax_orig.pcolormesh(
        x_coords, y_coords, original_values,
        shading='auto', cmap=cmap,
        vmin=data_min, vmax=data_max
    )
    ax_orig.set_title(title if title else 'Original')
    ax_orig.set_xlabel(x_label)
    ax_orig.set_ylabel(y_label)
    
    # Draw crosshair on original
    if show_crosshair:
        if y_coords.min() <= 0 <= y_coords.max():
            ax_orig.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.8)
        if x_coords.min() <= 0 <= x_coords.max():
            ax_orig.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.8)
    
    if show_enhancement and interactive:
        # Pre-create enhanced image (will be updated when method selected)
        im_enh = ax_enh.pcolormesh(
            x_coords, y_coords, original_values,
            shading='auto', cmap=cmap,
            vmin=data_min, vmax=data_max
        )
        ax_enh.set_title('Enhanced (select method →)')
        ax_enh.set_xlabel(x_label)
        ax_enh.set_ylabel(y_label)
        
        # Draw crosshair on enhanced
        if show_crosshair:
            if y_coords.min() <= 0 <= y_coords.max():
                ax_enh.axhline(y=0, color='white', linestyle='--', linewidth=1, alpha=0.8)
            if x_coords.min() <= 0 <= x_coords.max():
                ax_enh.axvline(x=0, color='white', linestyle='--', linewidth=1, alpha=0.8)
        
        # Enhancement options (English labels, on the right)
        enhancement_labels = [
            'None', 'BG Sub', 'CLAHE', 'Curvature', 
            '2nd Deriv', 'Gamma'
        ]
        radio = RadioButtons(ax_radio, enhancement_labels, active=0, 
                            activecolor='steelblue')
        ax_radio.set_title('Enhancement', fontsize=10, fontweight='bold')
        
        # Make radio button labels
        for label in radio.labels:
            label.set_fontsize(9)
        
        # Colormap options
        cmap_labels = ['bone_r', 'viridis', 'hot', 'coolwarm', 'gray']
        radio_cmap = RadioButtons(ax_cmap, cmap_labels, active=0, activecolor='steelblue')
        ax_cmap.set_title('Colormap', fontsize=9)
        for label in radio_cmap.labels:
            label.set_fontsize(8)
        
        # Invert toggle
        check_invert = CheckButtons(ax_invert, ['Invert'], [False])
        for label in check_invert.labels:
            label.set_fontsize(9)
        
        # Original image contrast RangeSlider
        slider_orig_contrast = RangeSlider(ax_orig_contrast, 'Contrast', data_min, data_max, 
                                           valinit=(data_min, data_max))
        
        # Enhanced image contrast RangeSlider
        slider_enh_contrast = RangeSlider(ax_enh_contrast, 'Contrast', data_min, data_max, 
                                          valinit=(data_min, data_max))
        
        # Parameter slider (hidden until enhancement with params selected)
        slider_param = Slider(ax_param, 'Param', 0.1, 10.0, valinit=2.0)
        
        # Store state
        state = {
            'current_method': 'None',
            'current_cmap': cmap,
            'inverted': False,
            'enhanced_data': None,
        }
        
        def get_current_cmap():
            """Get current colormap, applying invert if needed."""
            cmap_name = state['current_cmap']
            if state['inverted']:
                if cmap_name.endswith('_r'):
                    return cmap_name[:-2]
                else:
                    return cmap_name + '_r'
            return cmap_name
        
        def apply_enhancement(method, param):
            """Apply enhancement to original data and return result."""
            if method == 'None':
                return original_values.copy()
            elif method == 'BG Sub':
                # Shirley background subtraction (column-wise for EDC direction)
                result = original_values.copy()
                for i in range(result.shape[1]):
                    col = result[:, i]
                    bg = shirley_background(col)
                    result[:, i] = col - bg
                result = result - np.nanmin(result)
                return result
            elif method == 'CLAHE':
                return clahe(original_values, clip_limit=param)
            elif method == 'Curvature':
                return curvature_luo(original_values, k_res=param, e_res=param)
            elif method == '2nd Deriv':
                return curvature_second_derivative(original_values, strength=param)
            elif method == 'Gamma':
                return gamma_correction(original_values, gamma=param)
            else:
                return original_values.copy()
        
        def update_enhanced():
            """Update enhanced image based on current method and parameter."""
            method = state['current_method']
            param = slider_param.val
            current_cmap = get_current_cmap()
            
            # Show/hide enhanced panel and its sliders
            if method == 'None':
                ax_enh.set_visible(False)
                ax_enh_contrast.set_visible(False)
                state['enhanced_data'] = None
            else:
                ax_enh.set_visible(True)
                ax_enh_contrast.set_visible(True)
                
                enhanced = apply_enhancement(method, param)
                state['enhanced_data'] = enhanced
                
                # Update enhanced image
                enh_min = float(np.nanmin(enhanced))
                enh_max = float(np.nanmax(enhanced))
                
                im_enh.set_array(enhanced.ravel())
                im_enh.set_cmap(current_cmap)
                
                # Update slider range for enhanced image
                ax_enh_contrast.clear()
                slider_enh_new = RangeSlider(ax_enh_contrast, 'Contrast', enh_min, enh_max,
                                             valinit=(enh_min, enh_max))
                slider_enh_new.on_changed(on_enh_contrast_change)
                fig._widgets['slider_enh_contrast'] = slider_enh_new
                
                im_enh.set_clim(vmin=enh_min, vmax=enh_max)
                ax_enh.set_title(f'Enhanced: {method}')
            
            fig.canvas.draw_idle()
        
        def on_radio_change(label):
            """Handle enhancement method change."""
            state['current_method'] = label
            
            # Configure parameter slider based on method - need to recreate slider for new range
            if label in ['CLAHE', 'Curvature', '2nd Deriv', 'Gamma']:
                ax_param.set_visible(True)
                ax_param.clear()
                
                if label == 'CLAHE':
                    new_slider = Slider(ax_param, 'clip_limit', 0.001, 0.5, valinit=0.01)
                elif label == 'Curvature':
                    new_slider = Slider(ax_param, 'Scale', 0.1, 5.0, valinit=1.0)
                elif label == '2nd Deriv':
                    new_slider = Slider(ax_param, 'Sigma', 0.1, 5.0, valinit=1.0)
                elif label == 'Gamma':
                    new_slider = Slider(ax_param, 'Gamma', 0.1, 3.0, valinit=1.0)
                
                new_slider.on_changed(on_param_change)
                fig._widgets['slider_param'] = new_slider
            else:
                ax_param.set_visible(False)
            
            update_enhanced()
        
        def on_cmap_change(label):
            """Handle colormap change."""
            state['current_cmap'] = label
            current_cmap = get_current_cmap()
            im_orig.set_cmap(current_cmap)
            im_enh.set_cmap(current_cmap)
            fig.canvas.draw_idle()
        
        def on_invert_change(label):
            """Handle invert toggle."""
            state['inverted'] = not state['inverted']
            current_cmap = get_current_cmap()
            im_orig.set_cmap(current_cmap)
            im_enh.set_cmap(current_cmap)
            fig.canvas.draw_idle()
        
        def on_param_change(val):
            """Handle parameter slider change."""
            if state['current_method'] != 'None':
                update_enhanced()
        
        def on_orig_contrast_change(val):
            """Handle contrast RangeSlider change for original image."""
            vmin, vmax = val
            im_orig.set_clim(vmin=vmin, vmax=vmax)
            fig.canvas.draw_idle()
        
        def on_enh_contrast_change(val):
            """Handle contrast RangeSlider change for enhanced image."""
            vmin, vmax = val
            im_enh.set_clim(vmin=vmin, vmax=vmax)
            fig.canvas.draw_idle()
        
        # Connect callbacks
        radio.on_clicked(on_radio_change)
        radio_cmap.on_clicked(on_cmap_change)
        check_invert.on_clicked(on_invert_change)
        slider_param.on_changed(on_param_change)
        slider_orig_contrast.on_changed(on_orig_contrast_change)
        slider_enh_contrast.on_changed(on_enh_contrast_change)
        
        # Store widgets to prevent garbage collection
        fig._widgets = {
            'radio': radio,
            'radio_cmap': radio_cmap,
            'check_invert': check_invert,
            'slider_param': slider_param,
            'slider_orig_contrast': slider_orig_contrast,
            'slider_enh_contrast': slider_enh_contrast,
        }
        
    elif interactive:
        # Simple interactive mode without enhancement
        ax_vmin = plt.axes([0.15, 0.1, 0.65, 0.03])
        ax_vmax = plt.axes([0.15, 0.05, 0.65, 0.03])
        
        slider_vmin = Slider(ax_vmin, 'Min', data_min, data_max, valinit=data_min)
        slider_vmax = Slider(ax_vmax, 'Max', data_min, data_max, valinit=data_max)
        
        def update(val):
            vmin = slider_vmin.val
            vmax = slider_vmax.val
            im_orig.set_clim(vmin=vmin, vmax=vmax)
            fig.canvas.draw_idle()
        
        slider_vmin.on_changed(update)
        slider_vmax.on_changed(update)
        
        fig._sliders = (slider_vmin, slider_vmax)
    
    if not interactive:
        plt.tight_layout()
    
    return fig