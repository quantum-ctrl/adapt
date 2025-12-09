"""
Dimension Manipulation Module for ARPES Data

This module provides functions to manipulate data dimensions:
- Reduce 3D data to 2D by collapsing one dimension
- Merge multiple 2D datasets into a single 3D dataset

These operations preserve axis coordinates and metadata where possible.

Usage:
------
    from processing import reduce_to_2d, merge_to_3d
    
    # Reduce 3D to 2D by summing over scan axis
    data_2d = reduce_to_2d(data_3d, axis='scan', method='sum')
    
    # Merge multiple 2D datasets into 3D
    data_3d = merge_to_3d([data1, data2, data3], new_dim_name='scan', 
                          new_dim_values=[0, 1, 2])
"""

import numpy as np
import xarray as xr
from typing import Literal, Sequence
from scipy.ndimage import uniform_filter1d


def normalize_slices(
    data: xr.DataArray,
    norm_dim: str | int = -1,
    roi_dim1: float | tuple[float, float] = 0.9,
    roi_dim2: float | tuple[float, float] = 0.9,
    smooth_window: int = 3,
    method: Literal['sum', 'mean', 'max'] = 'sum',
    keep_attrs: bool = True
) -> xr.DataArray:
    """
    Normalize each slice along a specified dimension using ROI-based intensity.
    
    For 3D data, this function:
    1. For each slice along norm_dim, calculates the integrated intensity
       within an ROI (central portion of the other two dimensions)
    2. Smooths the intensity curve along norm_dim
    3. Normalizes each slice by dividing by its smoothed intensity value
    
    This is useful for correcting intensity variations across scans (e.g.,
    beam intensity changes during measurement).
    
    Parameters:
    -----------
    data : xarray.DataArray
        Input 3D data
    norm_dim : str or int
        The dimension along which to normalize (e.g., 'scan' or dim3).
        Can be dimension name (str) or index (int).
        Default is -1 (last dimension).
    roi_dim1: float or tuple
        Fraction (0-1) of the first remaining dimension to use for ROI,
        OR a tuple of (min, max) coordinate values.
        Default is 0.9 (central 90%).
    roi_dim2: float or tuple
        Fraction (0-1) of the second remaining dimension to use for ROI,
        OR a tuple of (min, max) coordinate values.
        Default is 0.9 (central 90%).
    smooth_window : int
        Window size for smoothing the intensity curve.
        Default is 3. Set to 1 for no smoothing.
    method : str
        Method to calculate slice intensity: 'sum', 'mean', or 'max'.
        Default is 'sum'.
    keep_attrs : bool
        Whether to preserve attributes from the original data.
        Default is True.
    
    Returns:
    --------
    xarray.DataArray: Normalized data with same shape as input
    
    Examples:
    ---------
    >>> # Normalize along scan axis using central 90% ROI
    >>> normalized = normalize_slices(data_3d, norm_dim='scan')
    
    >>> # Normalize along first dimension with custom ROI
    >>> normalized = normalize_slices(data_3d, norm_dim=0, 
    ...                               roi_dim1=0.8, roi_dim2=0.8)
    
    >>> # Normalize with more smoothing
    >>> normalized = normalize_slices(data_3d, norm_dim='scan', 
    ...                               smooth_window=5)
    
    Notes:
    ------
    - The ROI is centered in the data. For example, roi_dim1=0.9 means
      using the central 90% of dim1 (5% trimmed from each edge).
    - Smoothing uses a uniform (boxcar) filter to reduce noise in the
      intensity curve while preserving trends.
    - The output is normalized such that the mean of smoothed intensities
      equals 1 (preserving overall intensity scale).
    
    Raises:
    -------
    ValueError: If data is not 3D
    """
    if data.ndim != 3:
        raise ValueError(f"Input must be 3D data, got {data.ndim}D")
    
    # Convert axis index to name
    if isinstance(norm_dim, int):
        if norm_dim < 0:
            norm_dim = data.ndim + norm_dim
        if norm_dim < 0 or norm_dim >= data.ndim:
            raise ValueError(f"norm_dim index {norm_dim} out of range for {data.ndim}D data")
        norm_dim_name = data.dims[norm_dim]
    else:
        norm_dim_name = norm_dim
        if norm_dim_name not in data.dims:
            raise ValueError(f"norm_dim '{norm_dim_name}' not found. Available: {data.dims}")
    
    # Get the other two dimensions
    other_dims = [d for d in data.dims if d != norm_dim_name]
    if len(other_dims) != 2:
        raise ValueError(f"Expected 2 other dimensions, got {len(other_dims)}")
    
    dim1_name, dim2_name = other_dims
    
    # Calculate ROI bounds for each dimension
    def get_roi_slice(dim_name: str, roi_spec: float | tuple[float, float]) -> slice:
        """Get slice object for the ROI portion of a dimension."""
        coords = data.coords[dim_name].values
        n_points = len(coords)
        
        if isinstance(roi_spec, (int, float)):
            # Fraction case (central ROI)
            roi_fraction = float(roi_spec)
            # Calculate number of points to trim from each side
            trim_fraction = (1.0 - roi_fraction) / 2.0
            start_idx = int(n_points * trim_fraction)
            end_idx = n_points - start_idx
            
            # Ensure we have at least 1 point
            if end_idx <= start_idx:
                start_idx = 0
                end_idx = n_points
        else:
            # Tuple case (coordinate range)
            min_val, max_val = roi_spec
            # Handle potentially unsorted or decreasing coordinates
            range_min = min(min_val, max_val)
            range_max = max(min_val, max_val)
            
            mask = (coords >= range_min) & (coords <= range_max)
            if not np.any(mask):
                return slice(0, 0)
            
            indices = np.where(mask)[0]
            start_idx = indices[0]
            end_idx = indices[-1] + 1
        
        return slice(start_idx, end_idx)
    
    roi_slice1 = get_roi_slice(dim1_name, roi_dim1)
    roi_slice2 = get_roi_slice(dim2_name, roi_dim2)
    
    # Calculate intensity for each slice along norm_dim
    n_slices = data.sizes[norm_dim_name]
    intensities = np.zeros(n_slices)
    
    # Build indexer for the ROI
    norm_dim_idx = data.dims.index(norm_dim_name)
    dim1_idx = data.dims.index(dim1_name)
    dim2_idx = data.dims.index(dim2_name)
    
    for i in range(n_slices):
        # Build the indexer
        indexer = [slice(None)] * 3
        indexer[norm_dim_idx] = i
        indexer[dim1_idx] = roi_slice1
        indexer[dim2_idx] = roi_slice2
        
        # Get the ROI data for this slice
        roi_data = data.values[tuple(indexer)]
        
        # Calculate intensity using specified method
        if method == 'sum':
            intensities[i] = np.nansum(roi_data)
        elif method == 'mean':
            intensities[i] = np.nanmean(roi_data)
        elif method == 'max':
            intensities[i] = np.nanmax(roi_data)
        else:
            raise ValueError(f"Unknown method '{method}'. Available: ['sum', 'mean', 'max']")
    
    # Smooth the intensity curve
    if smooth_window > 1:
        smoothed_intensities = uniform_filter1d(intensities, size=smooth_window, mode='nearest')
    else:
        smoothed_intensities = intensities.copy()
    
    # Normalize to preserve overall intensity scale (mean = 1)
    mean_intensity = np.mean(smoothed_intensities)
    if mean_intensity == 0:
        raise ValueError("Mean intensity is zero, cannot normalize")
    
    norm_factors = smoothed_intensities / mean_intensity
    
    # Avoid division by zero
    norm_factors[norm_factors == 0] = 1.0
    
    # Apply normalization to each slice
    result_values = data.values.copy()
    for i in range(n_slices):
        indexer = [slice(None)] * 3
        indexer[norm_dim_idx] = i
        result_values[tuple(indexer)] = result_values[tuple(indexer)] / norm_factors[i]
    
    # Create result DataArray
    result = xr.DataArray(
        result_values,
        dims=data.dims,
        coords=data.coords,
        attrs=data.attrs if keep_attrs else {}
    )
    
    # Add normalization metadata
    if keep_attrs:
        result.attrs['normalization'] = {
            'norm_dim': norm_dim_name,
            'roi_dim1': roi_dim1,
            'roi_dim2': roi_dim2,
            'smooth_window': smooth_window,
            'method': method,
            'intensity_range': [float(intensities.min()), float(intensities.max())],
            'norm_factor_range': [float(norm_factors.min()), float(norm_factors.max())],
        }
    
    return result


def reduce_to_2d(
    data: xr.DataArray,
    axis: str | int = -1,
    method: Literal['sum', 'mean', 'max', 'min', 'median'] = 'sum',
    keep_attrs: bool = True
) -> xr.DataArray:
    """
    Reduce 3D data to 2D by collapsing one dimension.
    
    Parameters:
    -----------
    data : xarray.DataArray
        Input 3D data (e.g., with dimensions Eb, theta, scan)
    axis : str or int
        The dimension to collapse. Can be dimension name (str) or index (int).
        Default is -1 (last dimension, typically 'scan').
    method : str
        Reduction method: 'sum', 'mean', 'max', 'min', or 'median'
        Default is 'sum'.
    keep_attrs : bool
        Whether to preserve attributes from the original data.
        Default is True.
    
    Returns:
    --------
    xarray.DataArray: 2D data with the specified dimension collapsed
    
    Examples:
    ---------
    >>> # Sum over scan axis
    >>> data_2d = reduce_to_2d(data_3d, axis='scan', method='sum')
    
    >>> # Average over energy axis
    >>> data_2d = reduce_to_2d(data_3d, axis='Eb', method='mean')
    
    >>> # Use axis index (0 = first dimension)
    >>> data_2d = reduce_to_2d(data_3d, axis=0, method='mean')
    
    Raises:
    -------
    ValueError: If data is not 3D or axis not found
    """
    if data.ndim != 3:
        raise ValueError(f"Input must be 3D data, got {data.ndim}D")
    
    # Convert axis index to name
    if isinstance(axis, int):
        if axis < 0:
            axis = data.ndim + axis
        if axis < 0 or axis >= data.ndim:
            raise ValueError(f"Axis index {axis} out of range for {data.ndim}D data")
        axis_name = data.dims[axis]
    else:
        axis_name = axis
        if axis_name not in data.dims:
            raise ValueError(f"Axis '{axis_name}' not found. Available: {data.dims}")
    
    # Get original axis info for metadata
    original_shape = data.shape
    axis_size = data.sizes[axis_name]
    axis_coords = data.coords[axis_name].values
    
    # Apply reduction
    reduction_ops = {
        'sum': lambda x: x.sum(dim=axis_name, keep_attrs=keep_attrs),
        'mean': lambda x: x.mean(dim=axis_name, keep_attrs=keep_attrs),
        'max': lambda x: x.max(dim=axis_name, keep_attrs=keep_attrs),
        'min': lambda x: x.min(dim=axis_name, keep_attrs=keep_attrs),
        'median': lambda x: x.median(dim=axis_name, keep_attrs=keep_attrs),
    }
    
    if method not in reduction_ops:
        raise ValueError(f"Unknown method '{method}'. "
                        f"Available: {list(reduction_ops.keys())}")
    
    result = reduction_ops[method](data)
    
    # Update metadata
    if keep_attrs:
        result.attrs = dict(data.attrs)
        result.attrs['reduction'] = {
            'original_shape': original_shape,
            'reduced_axis': axis_name,
            'reduced_axis_size': axis_size,
            'reduced_axis_range': [float(axis_coords.min()), float(axis_coords.max())],
            'method': method
        }
    
    return result


def merge_to_3d(
    datasets: Sequence[xr.DataArray],
    new_dim_name: str = 'scan',
    new_dim_values: Sequence | None = None,
    new_dim_attrs: dict | None = None
) -> xr.DataArray:
    """
    Merge multiple 2D datasets into a single 3D dataset.
    
    Parameters:
    -----------
    datasets : sequence of xarray.DataArray
        List of 2D DataArrays to merge. All must have compatible dimensions.
    new_dim_name : str
        Name for the new dimension. Default is 'scan'.
    new_dim_values : sequence, optional
        Coordinate values for the new dimension. If None, uses 0, 1, 2, ...
        Length must match number of datasets.
    new_dim_attrs : dict, optional
        Attributes to store for the new dimension (e.g., {'units': 'eV'}).
    
    Returns:
    --------
    xarray.DataArray: 3D data with new dimension added
    
    Examples:
    ---------
    >>> # Merge with default integer indices
    >>> data_3d = merge_to_3d([data1, data2, data3])
    
    >>> # Merge with custom dimension values (e.g., photon energies)
    >>> data_3d = merge_to_3d([data1, data2, data3], 
    ...                       new_dim_name='hv',
    ...                       new_dim_values=[21.2, 40.8, 60.0])
    
    >>> # Merge with metadata
    >>> data_3d = merge_to_3d([data1, data2, data3],
    ...                       new_dim_name='temperature',
    ...                       new_dim_values=[10, 50, 100],
    ...                       new_dim_attrs={'units': 'K'})
    
    Raises:
    -------
    ValueError: If datasets have incompatible dimensions or shapes
    """
    if len(datasets) == 0:
        raise ValueError("At least one dataset is required")
    
    if len(datasets) == 1:
        # Single dataset: just expand dimensions
        data = datasets[0]
        if new_dim_values is None:
            new_dim_values = [0]
        return data.expand_dims({new_dim_name: new_dim_values})
    
    # Validate all datasets are 2D and compatible
    first = datasets[0]
    if first.ndim != 2:
        raise ValueError(f"All datasets must be 2D, got {first.ndim}D")
    
    ref_dims = first.dims
    ref_shape = first.shape
    
    for i, ds in enumerate(datasets[1:], start=1):
        if ds.ndim != 2:
            raise ValueError(f"Dataset {i} is {ds.ndim}D, expected 2D")
        if ds.dims != ref_dims:
            raise ValueError(f"Dataset {i} has dimensions {ds.dims}, "
                           f"expected {ref_dims}")
        if ds.shape != ref_shape:
            raise ValueError(f"Dataset {i} has shape {ds.shape}, "
                           f"expected {ref_shape}")
    
    # Create new dimension values
    if new_dim_values is None:
        new_dim_values = np.arange(len(datasets))
    else:
        new_dim_values = np.asarray(new_dim_values)
        if len(new_dim_values) != len(datasets):
            raise ValueError(f"new_dim_values length ({len(new_dim_values)}) "
                           f"must match number of datasets ({len(datasets)})")
    
    # Stack datasets along new dimension
    result = xr.concat(datasets, dim=new_dim_name)
    result = result.assign_coords({new_dim_name: new_dim_values})
    
    # Merge attributes from all datasets
    merged_attrs = {}
    for ds in datasets:
        merged_attrs.update(ds.attrs)
    
    # Add merge metadata
    merged_attrs['merge'] = {
        'n_datasets': len(datasets),
        'new_dimension': new_dim_name,
        'new_dimension_values': new_dim_values.tolist() if hasattr(new_dim_values, 'tolist') else list(new_dim_values)
    }
    
    if new_dim_attrs:
        merged_attrs['merge']['new_dimension_attrs'] = new_dim_attrs
    
    result.attrs = merged_attrs
    
    return result


def interpolate_to_common_grid(
    datasets: Sequence[xr.DataArray],
    grid_dim: str | None = None,
    n_points: int | None = None,
    method: str = 'linear'
) -> list[xr.DataArray]:
    """
    Interpolate multiple datasets to a common coordinate grid.
    
    Useful before merging datasets that have slightly different coordinate ranges.
    
    Parameters:
    -----------
    datasets : sequence of xarray.DataArray
        List of DataArrays to interpolate
    grid_dim : str, optional
        Dimension to interpolate. If None, interpolates all common dimensions.
    n_points : int, optional
        Number of points in the new grid. If None, uses the maximum from all datasets.
    method : str
        Interpolation method ('linear', 'nearest', 'cubic'). Default is 'linear'.
    
    Returns:
    --------
    list of xarray.DataArray: Interpolated datasets on common grid
    
    Example:
    --------
    >>> # Align energy grids before merging
    >>> aligned = interpolate_to_common_grid([data1, data2], grid_dim='Eb')
    >>> data_3d = merge_to_3d(aligned)
    """
    if len(datasets) == 0:
        return []
    
    # Find common dimensions
    common_dims = set(datasets[0].dims)
    for ds in datasets[1:]:
        common_dims &= set(ds.dims)
    
    if grid_dim is not None:
        if grid_dim not in common_dims:
            raise ValueError(f"Dimension '{grid_dim}' not found in all datasets")
        dims_to_interpolate = [grid_dim]
    else:
        dims_to_interpolate = list(common_dims)
    
    # Determine common grid for each dimension
    common_grids = {}
    for dim in dims_to_interpolate:
        all_coords = [ds.coords[dim].values for ds in datasets]
        
        # Find overlapping range
        min_val = max(arr.min() for arr in all_coords)
        max_val = min(arr.max() for arr in all_coords)
        
        # Determine number of points
        if n_points is None:
            n_pts = max(len(arr) for arr in all_coords)
        else:
            n_pts = n_points
        
        common_grids[dim] = np.linspace(min_val, max_val, n_pts)
    
    # Interpolate each dataset
    result = []
    for ds in datasets:
        interpolated = ds.interp(common_grids, method=method)
        result.append(interpolated)
    
    return result


def crop(
    data: xr.DataArray,
    dim1: tuple[float, float] | None = None,
    dim2: tuple[float, float] | None = None,
    dim3: tuple[float, float] | None = None,
    energy: tuple[float, float] | None = None,
    angle: tuple[float, float] | None = None,
    scan: tuple[float, float] | None = None,
    keep_attrs: bool = True
) -> xr.DataArray:
    """
    Crop data along specified dimensions using coordinate ranges.
    
    Parameters:
    -----------
    data : xarray.DataArray
        Input data (1D, 2D, or 3D)
    dim1 : tuple of (min, max), optional
        Range for the first dimension (index 0). 
        If None, no cropping is applied to this dimension.
    dim2 : tuple of (min, max), optional
        Range for the second dimension (index 1).
        If None, no cropping is applied to this dimension.
    dim3 : tuple of (min, max), optional
        Range for the third dimension (index 2).
        If None, no cropping is applied to this dimension.
    energy : tuple of (min, max), optional
        Range for 'energy' dimension by name (regardless of position).
    angle : tuple of (min, max), optional
        Range for 'angle' dimension by name (regardless of position).
    scan : tuple of (min, max), optional
        Range for 'scan' dimension by name (regardless of position).
    keep_attrs : bool
        Whether to preserve attributes from the original data.
        Default is True.
    
    Returns:
    --------
    xarray.DataArray: Cropped data
    
    Examples:
    ---------
    >>> # Crop using named dimension (recommended)
    >>> cropped = crop(data_3d, energy=(-3, 1))
    
    >>> # Crop multiple named dimensions
    >>> cropped = crop(data_3d, energy=(-3, 1), angle=(-10, 10))
    
    >>> # Crop using index (depends on dimension order!)
    >>> cropped = crop(data_2d, dim1=(-1, 0.5))
    
    >>> # Crop 3D data by index: energy from -2 to 1, angle from -10 to 10
    >>> cropped = crop(data_3d, dim1=(-2, 1), dim2=(-10, 10), dim3=(0, 5))
    
    Notes:
    ------
    - Named arguments (energy, angle, scan) are recommended as they work 
      regardless of dimension order.
    - Index-based arguments (dim1, dim2, dim3) depend on dimension order.
    - Cropping uses coordinate values (not indices), so the ranges are inclusive
      based on the nearest coordinate values.
    - If a range extends beyond the data bounds, only the available data is returned.
    - For dimensions not specified, the full range is preserved.
    
    Raises:
    -------
    ValueError: If a dimension range is specified for a non-existent dimension
    """
    # Build the selection dictionary
    sel_dict = {}
    n_dims = data.ndim
    
    # First, handle index-based ranges (dim1, dim2, dim3)
    ranges = [dim1, dim2, dim3]
    for i, rng in enumerate(ranges):
        if rng is not None:
            if i >= n_dims:
                raise ValueError(
                    f"dim{i+1} specified but data only has {n_dims} dimensions"
                )
            dim_name = data.dims[i]
            min_val, max_val = rng
            sel_dict[dim_name] = slice(min_val, max_val)
    
    # Then, handle named dimension ranges (these override index-based if same dim)
    named_ranges = {'energy': energy, 'angle': angle, 'scan': scan}
    for dim_name, rng in named_ranges.items():
        if rng is not None:
            if dim_name not in data.dims:
                raise ValueError(
                    f"Dimension '{dim_name}' not found in data. Available: {list(data.dims)}"
                )
            min_val, max_val = rng
            sel_dict[dim_name] = slice(min_val, max_val)
    
    # Apply selection
    if sel_dict:
        result = data.sel(sel_dict)
    else:
        result = data.copy()
    
    # Preserve attributes
    if keep_attrs:
        result.attrs = dict(data.attrs)
        # Add crop metadata
        crop_info = {}
        for dim_name, slc in sel_dict.items():
            original_range = [
                float(data.coords[dim_name].values.min()),
                float(data.coords[dim_name].values.max())
            ]
            crop_info[dim_name] = {
                'original_range': original_range,
                'crop_range': [slc.start, slc.stop],
                'original_size': data.sizes[dim_name],
                'cropped_size': result.sizes[dim_name]
            }
        if crop_info:
            result.attrs['crop'] = crop_info
    
    return result
