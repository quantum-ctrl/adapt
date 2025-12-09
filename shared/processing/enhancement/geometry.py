"""
Image Enhancement Module - Geometric Transformation Functions

Provides rotation, flipping, and scaling operations for scientific image preprocessing.
All functions accept NumPy ndarrays and return ndarrays with preserved dtype.
"""

import numpy as np
import cv2
from typing import Literal, Optional, Tuple, Union


def rotate(image: np.ndarray, angle: float, 
           center: Optional[Tuple[float, float]] = None,
           scale: float = 1.0,
           border_mode: str = "constant",
           border_value: float = 0) -> np.ndarray:
    """
    Rotate an image by a specified angle.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as 2D (grayscale) or 3D (color) array.
    angle : float
        Rotation angle in degrees. Positive values rotate counter-clockwise.
    center : tuple of float, optional
        Center of rotation (x, y). Default is image center.
    scale : float, optional
        Scaling factor applied during rotation. Default is 1.0.
    border_mode : str, optional
        Border handling mode. One of "constant", "replicate", "reflect".
        Default is "constant".
    border_value : float, optional
        Value used for constant border mode. Default is 0.
    
    Returns
    -------
    np.ndarray
        Rotated image with the same shape and dtype as input.
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> rotated = rotate(image, 45)  # Rotate 45 degrees counter-clockwise
    """
    original_dtype = image.dtype
    
    h, w = image.shape[:2]
    
    if center is None:
        center = (w / 2, h / 2)
    
    # Border mode mapping
    border_modes = {
        "constant": cv2.BORDER_CONSTANT,
        "replicate": cv2.BORDER_REPLICATE,
        "reflect": cv2.BORDER_REFLECT
    }
    cv_border = border_modes.get(border_mode, cv2.BORDER_CONSTANT)
    
    # Get rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # Handle different dtypes
    if image.dtype in [np.float32, np.float64]:
        result = cv2.warpAffine(
            image.astype(np.float64), rotation_matrix, (w, h),
            borderMode=cv_border, borderValue=border_value
        )
    else:
        result = cv2.warpAffine(
            image, rotation_matrix, (w, h),
            borderMode=cv_border, borderValue=int(border_value)
        )
    
    return result.astype(original_dtype)


def flip(image: np.ndarray, 
         direction: Literal["horizontal", "vertical", "both"] = "horizontal") -> np.ndarray:
    """
    Flip an image horizontally, vertically, or both.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as 2D (grayscale) or 3D (color) array.
    direction : str, optional
        Flip direction. One of:
        - "horizontal": Flip left-right
        - "vertical": Flip top-bottom
        - "both": Flip both directions (180° rotation)
        Default is "horizontal".
    
    Returns
    -------
    np.ndarray
        Flipped image with the same shape and dtype as input.
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> flipped_h = flip(image, "horizontal")
    >>> flipped_v = flip(image, "vertical")
    """
    valid_directions = ["horizontal", "vertical", "both"]
    if direction not in valid_directions:
        raise ValueError(f"direction must be one of {valid_directions}, got '{direction}'")
    
    original_dtype = image.dtype
    
    if direction == "horizontal":
        flip_code = 1  # Flip around y-axis
    elif direction == "vertical":
        flip_code = 0  # Flip around x-axis
    else:  # both
        flip_code = -1  # Flip around both axes
    
    result = cv2.flip(image, flip_code)
    
    return result.astype(original_dtype)


def scale(image: np.ndarray, 
          scale_factor: Optional[float] = None,
          size: Optional[Tuple[int, int]] = None,
          interpolation: str = "linear") -> np.ndarray:
    """
    Scale (resize) an image by a factor or to a specific size.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as 2D (grayscale) or 3D (color) array.
    scale_factor : float, optional
        Scaling factor. Values > 1 enlarge, values < 1 shrink.
        Either scale_factor or size must be provided.
    size : tuple of int, optional
        Target size as (width, height).
        Either scale_factor or size must be provided.
    interpolation : str, optional
        Interpolation method. One of:
        - "nearest": Nearest neighbor (fast, preserves hard edges)
        - "linear": Bilinear interpolation (smooth, default)
        - "cubic": Bicubic interpolation (smoother, slower)
        - "lanczos": Lanczos interpolation (high quality downsampling)
        Default is "linear".
    
    Returns
    -------
    np.ndarray
        Scaled image with the same dtype as input.
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> enlarged = scale(image, scale_factor=2.0)
    >>> resized = scale(image, size=(50, 50))
    """
    if scale_factor is None and size is None:
        raise ValueError("Either scale_factor or size must be provided")
    if scale_factor is not None and size is not None:
        raise ValueError("Only one of scale_factor or size should be provided")
    
    original_dtype = image.dtype
    h, w = image.shape[:2]
    
    # Interpolation mapping
    interp_modes = {
        "nearest": cv2.INTER_NEAREST,
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4
    }
    if interpolation not in interp_modes:
        raise ValueError(f"interpolation must be one of {list(interp_modes.keys())}")
    cv_interp = interp_modes[interpolation]
    
    # Calculate target size
    if scale_factor is not None:
        if scale_factor <= 0:
            raise ValueError("scale_factor must be positive")
        new_w = int(w * scale_factor)
        new_h = int(h * scale_factor)
    else:
        new_w, new_h = size
    
    # Handle different dtypes
    if image.dtype in [np.float32, np.float64]:
        result = cv2.resize(image.astype(np.float64), (new_w, new_h), 
                           interpolation=cv_interp)
    else:
        result = cv2.resize(image, (new_w, new_h), interpolation=cv_interp)
    
    return result.astype(original_dtype)


def get_slice(data, energy_pos=None, angle_pos=None, scan_pos=None,
              energy_dim='energy', angle_dim='angle', scan_dim='scan'):
    """
    Extract 3 orthogonal slices from 3D data at specified axis positions.
    
    This function takes a 3D xarray.DataArray and extracts three 2D slices
    at the specified coordinates. Each slice is returned as a 2D DataArray
    with the same structure and naming conventions as standard 2D ARPES data.
    
    Parameters
    ----------
    data : xarray.DataArray
        3D ARPES data with dimensions (energy, angle, scan)
    energy_pos : float, optional
        Energy coordinate for the YZ slice (angle vs scan).
        If None, uses the center of the energy axis.
    angle_pos : float, optional
        Angle coordinate for the XZ slice (energy vs scan).
        If None, uses the center of the angle axis.
    scan_pos : float, optional
        Scan coordinate for the XY slice (energy vs angle).
        If None, uses the center of the scan axis.
    energy_dim : str
        Name of the energy dimension (default: 'energy')
    angle_dim : str
        Name of the angle dimension (default: 'angle')
    scan_dim : str
        Name of the scan dimension (default: 'scan')
    
    Returns
    -------
    dict
        Dictionary containing three 2D xarray.DataArray slices:
        - 'xy': Energy vs Angle slice at the specified scan position
        - 'xz': Energy vs Scan slice at the specified angle position
        - 'yz': Angle vs Scan slice at the specified energy position
        
        Each slice has:
        - dims: two of (energy, angle, scan) depending on slice type
        - coords: coordinate values for each dimension
        - attrs: copied from the original 3D data with added slice position info
    
    Examples
    --------
    >>> slices = get_slice(data_3d, energy_pos=-0.5, angle_pos=0, scan_pos=10)
    >>> xy_slice = slices['xy']  # 2D data: energy vs angle at scan=10
    >>> xz_slice = slices['xz']  # 2D data: energy vs scan at angle=0
    >>> yz_slice = slices['yz']  # 2D data: angle vs scan at energy=-0.5
    
    Notes
    -----
    The returned slices are compatible with all 2D processing functions,
    such as `plot_2d_data()`, `interactive_plot_2d()`, etc.
    """
    import xarray as xr
    
    # Validate input
    dims = list(data.dims)
    if len(dims) != 3:
        raise ValueError(f"Data must be 3D, got dimensions: {dims}")
    
    # Identify dimension names
    _energy_dim = energy_dim if energy_dim in dims else dims[0]
    _angle_dim = angle_dim if angle_dim in dims else dims[1]
    _scan_dim = scan_dim if scan_dim in dims else dims[2]
    
    # Get coordinate arrays
    energy_vals = data.coords[_energy_dim].values
    angle_vals = data.coords[_angle_dim].values
    scan_vals = data.coords[_scan_dim].values
    
    # Set default positions to center if not provided
    if energy_pos is None:
        energy_pos = energy_vals[len(energy_vals) // 2]
    if angle_pos is None:
        angle_pos = angle_vals[len(angle_vals) // 2]
    if scan_pos is None:
        scan_pos = scan_vals[len(scan_vals) // 2]
    
    # Find nearest indices
    energy_idx = int(np.argmin(np.abs(energy_vals - energy_pos)))
    angle_idx = int(np.argmin(np.abs(angle_vals - angle_pos)))
    scan_idx = int(np.argmin(np.abs(scan_vals - scan_pos)))
    
    # Get actual coordinate values at the indices
    actual_energy = float(energy_vals[energy_idx])
    actual_angle = float(angle_vals[angle_idx])
    actual_scan = float(scan_vals[scan_idx])
    
    # Extract XY slice: Energy vs Angle at fixed Scan
    xy_slice = data.isel({_scan_dim: scan_idx})
    xy_attrs = dict(data.attrs) if data.attrs else {}
    xy_attrs['slice_type'] = 'xy'
    xy_attrs['slice_dim'] = _scan_dim
    xy_attrs['slice_pos'] = actual_scan
    xy_slice = xr.DataArray(
        xy_slice.values,
        dims=[_energy_dim, _angle_dim],
        coords={_energy_dim: energy_vals, _angle_dim: angle_vals},
        attrs=xy_attrs
    )
    
    # Extract XZ slice: Energy vs Scan at fixed Angle
    xz_slice = data.isel({_angle_dim: angle_idx})
    xz_attrs = dict(data.attrs) if data.attrs else {}
    xz_attrs['slice_type'] = 'xz'
    xz_attrs['slice_dim'] = _angle_dim
    xz_attrs['slice_pos'] = actual_angle
    xz_slice = xr.DataArray(
        xz_slice.values,
        dims=[_energy_dim, _scan_dim],
        coords={_energy_dim: energy_vals, _scan_dim: scan_vals},
        attrs=xz_attrs
    )
    
    # Extract YZ slice: Angle vs Scan at fixed Energy
    yz_slice = data.isel({_energy_dim: energy_idx})
    yz_attrs = dict(data.attrs) if data.attrs else {}
    yz_attrs['slice_type'] = 'yz'
    yz_attrs['slice_dim'] = _energy_dim
    yz_attrs['slice_pos'] = actual_energy
    yz_slice = xr.DataArray(
        yz_slice.values,
        dims=[_angle_dim, _scan_dim],
        coords={_angle_dim: angle_vals, _scan_dim: scan_vals},
        attrs=yz_attrs
    )
    
    return {
        'xy': xy_slice,
        'xz': xz_slice,
        'yz': yz_slice,
        'position': {
            _energy_dim: actual_energy,
            _angle_dim: actual_angle,
            _scan_dim: actual_scan
        }
    }


__all__ = [
    'rotate',
    'flip',
    'scale',
    'get_slice'
]
