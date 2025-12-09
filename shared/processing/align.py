"""
Axis Alignment Module for ARPES Data

This module provides functions to align coordinate axes in ARPES data
by applying offsets and tracking alignment history.

Typical use cases:
- Energy axis: Shift so that E_F = 0
- Angle axis: Shift so that normal emission θ = 0

Usage:
------
    from align import align_axis, align_energy
    
    # Align energy axis
    data_aligned = align_energy(data, E_F=0.05)
    
    # Or use general function
    data_aligned = align_axis(data, 'theta', offset=2.3)
"""

import numpy as np
import xarray as xr
from datetime import datetime


def align_axis(data, axis_name, offset, new_name=None, method='manual'):
    """
    Align an axis by applying an offset.
    
    The new coordinates will be: coord_new = coord_old - offset
    
    Parameters:
    -----------
    data : xarray.DataArray
        Input data with the axis to align
    axis_name : str
        Name of the axis/coordinate to align (e.g., 'Eb', 'theta')
    offset : float
        Offset to subtract from the coordinates
    new_name : str, optional
        New name for the axis. If None, keeps the original name.
    method : str
        Description of alignment method (for metadata)
    
    Returns:
    --------
    xarray.DataArray: New DataArray with aligned coordinates
    
    Example:
    --------
    >>> # Shift energy so E_F (at 0.05 eV) becomes 0
    >>> data_aligned = align_axis(data, 'energy', offset=0.05)
    """
    if axis_name not in data.coords:
        raise ValueError(f"Axis '{axis_name}' not found in data. "
                        f"Available axes: {list(data.coords.keys())}")
    
    # Get original coordinates
    old_coords = data.coords[axis_name].values
    
    # Calculate new coordinates
    new_coords = old_coords - offset
    
    # Create new data with updated coordinates
    if new_name is None:
        new_name = axis_name
    
    # Use assign_coords to create new DataArray
    new_data = data.assign_coords({axis_name: new_coords})
    
    # If renaming, need to rename the dimension
    if new_name != axis_name:
        new_data = new_data.rename({axis_name: new_name})
    
    # Update alignment metadata
    new_data = _update_alignment_metadata(
        new_data, axis_name, offset, new_name, method
    )
    
    return new_data


def _update_alignment_metadata(data, axis_name, offset, new_name, method):
    """Update the alignment history in data attributes."""
    # Get existing alignment info or create new
    alignment = dict(data.attrs.get('alignment', {}))
    
    # Add this alignment
    alignment[axis_name] = {
        'offset': float(offset),
        'original_name': axis_name,
        'new_name': new_name,
        'method': method,
        'timestamp': datetime.now().isoformat()
    }
    
    # Create new attrs dict and update
    new_attrs = dict(data.attrs)
    new_attrs['alignment'] = alignment
    
    # Assign new attributes
    data.attrs = new_attrs
    
    return data


def align_energy(data, E_F, energy_axis='energy'):
    """
    Align energy axis so that the Fermi level is at zero.
    
    Parameters:
    -----------
    data : xarray.DataArray
        Input ARPES data
    E_F : float
        Current Fermi level position in eV
    energy_axis : str
        Name of the energy axis (default: 'energy')
    
    Returns:
    --------
    xarray.DataArray: Data with aligned energy axis (E_F = 0)
    
    Example:
    --------
    >>> # After fitting Fermi edge and finding E_F = 0.05 eV
    >>> data_aligned = align_energy(data, E_F=0.05)
    """
    return align_axis(
        data, 
        axis_name=energy_axis, 
        offset=E_F,
        method=f'fermi_alignment (E_F={E_F:.4f} eV)'
    )


def align_energy_3d(data, E_F_array, scan_dim='scan', energy_axis='energy'):
    """
    Align 3D data where Fermi level varies with scan position.
    
    Resamples each slice to a common energy grid after shifting.
    
    Parameters:
    -----------
    data : xarray.DataArray
        3D input data (Energy, Angle, Scan)
    E_F_array : array-like
        Array of Fermi level positions for each scan index
    scan_dim : str
        Name of the scan dimension
    energy_axis : str
        Name of the energy dimension (default: 'energy')
        
    Returns:
    --------
    xarray.DataArray: Aligned 3D data with E_F = 0
    """
    # Save original dimension order
    original_dims = data.dims
    
    # 1. Define common energy grid (centered at 0)
    eb_values = data.coords[energy_axis].values
    eb_res = np.abs(np.mean(np.diff(eb_values)))
    
    # Estimate new range
    min_shift = np.min(E_F_array)
    max_shift = np.max(E_F_array)
    new_min = eb_values.min() - max_shift
    new_max = eb_values.max() - min_shift
    
    new_eb = np.arange(new_min, new_max, eb_res)
    
    # 2. Process each slice
    aligned_slices = []
    scan_coords = data.coords[scan_dim].values
    
    # Check if tqdm is available for progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(enumerate(scan_coords), total=len(scan_coords), desc="Resampling")
    except ImportError:
        iterator = enumerate(scan_coords)
    
    for i, scan_val in iterator:
        slice_data = data.sel({scan_dim: scan_val})
        
        # Shift energy axis
        shifted_eb = slice_data.coords[energy_axis].values - E_F_array[i]
        
        # Assign new coords temporarily
        slice_data = slice_data.assign_coords({energy_axis: shifted_eb})
        
        # Resample to common grid
        slice_aligned = slice_data.interp({energy_axis: new_eb}, method='linear', kwargs={'fill_value': 0})
        
        aligned_slices.append(slice_aligned)
    
    # 3. Concatenate
    data_aligned = xr.concat(aligned_slices, dim=scan_dim)
    data_aligned = data_aligned.assign_coords({scan_dim: scan_coords})
    
    # 4. Restore original dimension order
    data_aligned = data_aligned.transpose(*original_dims)
    
    # Update metadata
    data_aligned.attrs = dict(data.attrs)
    data_aligned.attrs['E_F_correction'] = '3D_resampled'
    
    return data_aligned
