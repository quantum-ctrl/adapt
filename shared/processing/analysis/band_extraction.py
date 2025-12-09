"""
Band Extraction and Vector Export Module

Automatically detect and extract band structures from ARPES data,
and export them as vector graphics (SVG/PDF).

This module provides:
- Data preprocessing with curvature enhancement
- Band detection using multiple methods (ridge, edge, peak-based)
- Vector export (SVG/PDF) for publication-ready figures

Example Usage
-------------
>>> import xarray as xr
>>> from processing.analysis.band_extraction import extract_bands, export_to_svg
>>>
>>> # Load data via loaders
>>> data = load_adress_data("path/to/data.h5")
>>>
>>> # Extract bands
>>> bands = extract_bands(data, method="ridge")
>>>
>>> # Export to SVG
>>> export_to_svg(bands, "output.svg", data.shape)
"""

import numpy as np
import xarray as xr
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage import filters, feature, morphology
from skimage.measure import find_contours
from typing import List, Dict, Any, Optional, Tuple, Union, Literal
from dataclasses import dataclass
import warnings

# Import from sibling modules
try:
    from .peaks import (
        extract_band_dispersion, detect_peaks_1d,
        extract_mdc_peaks, extract_edc_peaks, fit_peak_lorentzian
    )
except ImportError:
    from peaks import (
        extract_band_dispersion, detect_peaks_1d,
        extract_mdc_peaks, extract_edc_peaks, fit_peak_lorentzian
    )

try:
    from ..enhancement.curvature import curvature_luo, curvature_second_derivative, auto_curvature
except ImportError:
    try:
        from processing.enhancement.curvature import curvature_luo, curvature_second_derivative, auto_curvature
    except ImportError:
        curvature_luo = None
        curvature_second_derivative = None
        auto_curvature = None


# =============================================================================
# Band Profile Extraction Result
# =============================================================================

@dataclass
class BandResult:
    """
    Container for band extraction results with positions and uncertainties.
    
    Attributes
    ----------
    positions : np.ndarray
        2D array of peak positions with shape (n_profiles, n_bands).
        NaN values indicate no peak was found for that profile/band.
    uncertainties : np.ndarray
        2D array of position uncertainties with same shape as positions.
    profile_coords : np.ndarray
        1D array of coordinate values where profiles were extracted
        (e.g., energy values for MDC extraction).
    slice_coords : np.ndarray
        1D array of coordinate values along each profile slice
        (e.g., momentum values for MDC extraction).
    direction : str
        "horizontal" (MDC) or "vertical" (EDC).
    dim_names : tuple of str
        Names of the two dimensions (dim0, dim1).
    n_bands : int
        Number of bands tracked.
    profile_indices : np.ndarray
        Indices of the extracted profiles in the original data.
    fit_quality : np.ndarray
        2D array of fit quality scores (R-squared) for each peak fit.
    """
    positions: np.ndarray
    uncertainties: np.ndarray
    profile_coords: np.ndarray
    slice_coords: np.ndarray
    direction: str
    dim_names: Tuple[str, str]
    n_bands: int
    profile_indices: np.ndarray
    fit_quality: np.ndarray


def extract_band_profiles(
    data: Union[xr.DataArray, np.ndarray],
    direction: Literal["horizontal", "vertical"] = "horizontal",
    reverse: bool = False,
    n_profiles: int = 10,
    n_bands: int = 1,
    threshold: float = 0.05,
    min_distance: int = 3,
    fit_window: int = 10
) -> BandResult:
    """
    Extract band profiles from ARPES data along MDC or EDC direction.
    
    This function extracts evenly-spaced profiles (MDC or EDC) from 2D ARPES
    data, fits peaks using Lorentzian model, and tracks band positions with
    uncertainties.
    
    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        Input 2D ARPES data. For xr.DataArray, expects dims like
        ['energy', 'angle'] or similar. Shape should be (n_dim0, n_dim1).
    direction : {"horizontal", "vertical"}, optional
        Profile extraction direction:
        - "horizontal": MDC extraction (slice along dim1 at fixed dim0).
          For typical ARPES data with dims (energy, angle), this gives
          intensity vs angle at fixed energy.
        - "vertical": EDC extraction (slice along dim0 at fixed dim1).
          For typical ARPES data, this gives intensity vs energy at fixed angle.
        Default: "horizontal".
    reverse : bool, optional
        If False, profiles are extracted from positive to negative coordinate
        direction (e.g., high energy to low energy for MDC).
        If True, profiles are extracted from negative to positive direction.
        Default: False.
    n_profiles : int, optional
        Number of evenly-spaced profiles to extract (间隔). Default: 10.
    n_bands : int, optional
        Number of bands to track per profile. The n_bands most prominent
        peaks are selected based on intensity. Default: 1.
    threshold : float, optional
        Peak detection prominence threshold (0-1). Default: 0.05.
    min_distance : int, optional
        Minimum distance between peaks in pixels. Default: 3.
    fit_window : int, optional
        Half-width of fitting window around each peak (pixels). Default: 10.
        
    Returns
    -------
    BandResult
        Dataclass containing:
        - positions: (n_profiles, n_bands) array of fitted peak positions
        - uncertainties: (n_profiles, n_bands) array of position errors
        - profile_coords: coordinates where profiles were extracted
        - slice_coords: coordinates along the profile slices
        - direction, dim_names, n_bands, profile_indices, fit_quality
        
    Examples
    --------
    >>> from loaders import load_ibw_data
    >>> data = load_ibw_data("path/to/data.ibw")
    >>> result = extract_band_profiles(data, direction="horizontal", n_profiles=15)
    >>> print(f"Positions shape: {result.positions.shape}")
    >>> print(f"Mean uncertainty: {np.nanmean(result.uncertainties):.4f}")
    
    Notes
    -----
    - For MDC (horizontal): profiles are taken at different energies, showing
      intensity vs momentum. Good for tracking dispersion E(k).
    - For EDC (vertical): profiles are taken at different momenta, showing
      intensity vs energy. Good for analyzing spectral features.
    - Peak positions use Lorentzian fitting for sub-pixel accuracy.
    - NaN values in output indicate no peak was found or fit failed.
    """
    # Extract data and coordinates
    if isinstance(data, xr.DataArray):
        image = data.values.astype(np.float64)
        dims = list(data.dims)
        coord0 = data.coords[dims[0]].values.astype(np.float64)
        coord1 = data.coords[dims[1]].values.astype(np.float64)
        dim_names = (dims[0], dims[1])
    else:
        image = np.asarray(data, dtype=np.float64)
        coord0 = np.arange(image.shape[0], dtype=np.float64)
        coord1 = np.arange(image.shape[1], dtype=np.float64)
        dim_names = ('dim0', 'dim1')
    
    if image.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {image.shape}")
    
    n_dim0, n_dim1 = image.shape
    
    # Determine profile direction and setup
    if direction == "horizontal":
        # MDC: slice along dim1 (e.g., angle) at fixed dim0 (e.g., energy)
        n_slices = n_dim0
        slice_coords = coord1
        profile_coord_values = coord0
        extract_func = extract_mdc_peaks
        extract_kwargs_base = {'k_coords': coord1}
        pos_key = 'k_fitted'
        pos_fallback = 'k'
        err_key = 'k_err'
        index_key = 'energy_index'
    else:  # vertical
        # EDC: slice along dim0 (e.g., energy) at fixed dim1 (e.g., angle)
        n_slices = n_dim1
        slice_coords = coord0
        profile_coord_values = coord1
        extract_func = extract_edc_peaks
        extract_kwargs_base = {'energy_coords': coord0}
        pos_key = 'energy_fitted'
        pos_fallback = 'energy'
        err_key = 'energy_err'
        index_key = 'k_index'
    
    # Calculate profile indices (evenly spaced)
    if reverse:
        # Negative to positive: start from 0
        profile_indices = np.linspace(0, n_slices - 1, n_profiles, dtype=int)
    else:
        # Positive to negative: start from end (default)
        profile_indices = np.linspace(n_slices - 1, 0, n_profiles, dtype=int)
    
    # Initialize output arrays
    positions = np.full((n_profiles, n_bands), np.nan)
    uncertainties = np.full((n_profiles, n_bands), np.nan)
    fit_quality = np.full((n_profiles, n_bands), np.nan)
    profile_coords = profile_coord_values[profile_indices]
    
    # Extract and fit peaks for each profile
    for i, idx in enumerate(profile_indices):
        # Extract peaks using the appropriate function
        if direction == "horizontal":
            peaks = extract_func(
                image, energy_index=int(idx),
                threshold=threshold, min_distance=min_distance,
                fit_peaks=True, fit_window=fit_window,
                **extract_kwargs_base
            )
        else:
            peaks = extract_func(
                image, k_index=int(idx),
                threshold=threshold, min_distance=min_distance,
                fit_peaks=True, fit_window=fit_window,
                **extract_kwargs_base
            )
        
        if not peaks:
            continue
        
        # Sort peaks by intensity (most prominent first)
        peaks_sorted = sorted(peaks, key=lambda p: p.get('intensity', 0), reverse=True)
        
        # Extract the n_bands most prominent peaks
        for j, peak in enumerate(peaks_sorted[:n_bands]):
            # Get fitted position if available, otherwise use raw position
            if pos_key in peak:
                positions[i, j] = peak[pos_key]
                uncertainties[i, j] = peak.get(err_key, np.nan)
            elif pos_fallback in peak:
                positions[i, j] = peak[pos_fallback]
                uncertainties[i, j] = np.nan
            
            # Get fit quality if available
            if 'fit_result' in peak and peak['fit_result'].get('success', False):
                fit_quality[i, j] = peak['fit_result'].get('fit_quality', np.nan)
    
    return BandResult(
        positions=positions,
        uncertainties=uncertainties,
        profile_coords=profile_coords,
        slice_coords=slice_coords,
        direction=direction,
        dim_names=dim_names,
        n_bands=n_bands,
        profile_indices=profile_indices,
        fit_quality=fit_quality
    )


# =============================================================================
# Data Preprocessing
# =============================================================================

def preprocess_data(
    data: Union[xr.DataArray, np.ndarray],
    denoise: bool = True,
    denoise_sigma: float = 1.0,
    enhance_contrast: bool = True,
    enhancement_method: Literal["curvature", "second_derivative", "auto", "none"] = "auto"
) -> np.ndarray:
    """
    Preprocess ARPES data to enhance band features.
    
    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        Input 2D ARPES data (energy × momentum/angle).
    denoise : bool, optional
        Apply Gaussian denoising. Default: True.
    denoise_sigma : float, optional
        Gaussian smoothing sigma. Default: 1.0.
    enhance_contrast : bool, optional
        Apply curvature enhancement. Default: True.
    enhancement_method : str, optional
        Enhancement method: "curvature", "second_derivative", "auto", or "none".
        Default: "auto".
        
    Returns
    -------
    np.ndarray
        Preprocessed 2D image with enhanced band contrast.
    """
    # Convert to numpy array
    if isinstance(data, xr.DataArray):
        image = data.values.astype(np.float64)
    else:
        image = np.asarray(data, dtype=np.float64)
    
    if image.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {image.shape}")
    
    # Step 1: Denoise
    if denoise and denoise_sigma > 0:
        image = gaussian_filter(image, sigma=denoise_sigma)
    
    # Step 2: Curvature enhancement
    if enhance_contrast and enhancement_method != "none":
        if enhancement_method == "curvature" and curvature_luo is not None:
            image = curvature_luo(image)
        elif enhancement_method == "second_derivative" and curvature_second_derivative is not None:
            image = curvature_second_derivative(image)
        elif enhancement_method == "auto" and auto_curvature is not None:
            image, _ = auto_curvature(image)
        else:
            # Fallback: simple enhancement using Laplacian
            laplacian = ndimage.laplace(image)
            image = np.maximum(-laplacian, 0)
            # Normalize
            if image.max() > 0:
                image = image / image.max()
    
    return image


# =============================================================================
# Band Detection Methods
# =============================================================================

def detect_bands_ridge(
    image: np.ndarray,
    sigmas: Tuple[float, ...] = (1.0, 2.0, 3.0),
    threshold: float = 0.1,
    min_length: int = 10
) -> List[np.ndarray]:
    """
    Detect bands using ridge detection (Meijering filter).
    
    This method is effective for detecting thin, elongated structures
    like bands in ARPES data.
    
    Parameters
    ----------
    image : np.ndarray
        Preprocessed 2D image.
    sigmas : tuple of float, optional
        Scales for ridge detection. Default: (1.0, 2.0, 3.0).
    threshold : float, optional
        Threshold for ridge response (0-1). Default: 0.1.
    min_length : int, optional
        Minimum contour length in pixels. Default: 10.
        
    Returns
    -------
    list of np.ndarray
        List of band curves, each as (N, 2) array of (row, col) coordinates.
    """
    # Apply Meijering filter for ridge detection
    ridge_response = filters.meijering(image, sigmas=sigmas, black_ridges=False)
    
    # Normalize response
    if ridge_response.max() > 0:
        ridge_response = ridge_response / ridge_response.max()
    
    # Threshold to binary
    binary = ridge_response > threshold
    
    # Skeletonize to get thin lines
    skeleton = morphology.skeletonize(binary)
    
    # Find contours
    contours = find_contours(skeleton.astype(float), 0.5)
    
    # Filter by length
    bands = [c for c in contours if len(c) >= min_length]
    
    return bands


def detect_bands_edge(
    image: np.ndarray,
    sigma: float = 2.0,
    low_threshold: float = 0.1,
    high_threshold: float = 0.3,
    min_length: int = 10
) -> List[np.ndarray]:
    """
    Detect bands using Canny edge detection and contour extraction.
    
    Parameters
    ----------
    image : np.ndarray
        Preprocessed 2D image.
    sigma : float, optional
        Gaussian smoothing sigma for Canny. Default: 2.0.
    low_threshold : float, optional
        Canny low threshold (0-1). Default: 0.1.
    high_threshold : float, optional
        Canny high threshold (0-1). Default: 0.3.
    min_length : int, optional
        Minimum contour length. Default: 10.
        
    Returns
    -------
    list of np.ndarray
        List of band curves as (N, 2) arrays.
    """
    # Normalize image
    img_norm = image.copy()
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()
    
    # Canny edge detection
    edges = feature.canny(
        img_norm, 
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
    
    # Find contours
    contours = find_contours(edges.astype(float), 0.5)
    
    # Filter by length
    bands = [c for c in contours if len(c) >= min_length]
    
    return bands


def detect_bands_peak(
    data: Union[xr.DataArray, np.ndarray],
    threshold: float = 0.05,
    min_distance: int = 3,
    min_track_length: int = 10,
    axis: str = "MDC"
) -> List[np.ndarray]:
    """
    Detect bands using peak tracking from peaks.py.
    
    This method uses MDC or EDC peak fitting and tracking to extract
    band dispersions with high accuracy.
    
    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        Input 2D ARPES data.
    threshold : float, optional
        Peak detection threshold. Default: 0.05.
    min_distance : int, optional
        Minimum peak separation. Default: 3.
    min_track_length : int, optional
        Minimum number of points in a band. Default: 10.
    axis : str, optional
        "MDC" or "EDC" extraction direction. Default: "MDC".
        
    Returns
    -------
    list of np.ndarray
        List of band curves as (N, 2) arrays in (row, col) format.
    """
    # Get coordinates
    if isinstance(data, xr.DataArray):
        image = data.values.astype(np.float64)
        dims = list(data.dims)
        energy_coords = data.coords[dims[0]].values
        k_coords = data.coords[dims[1]].values
    else:
        image = np.asarray(data, dtype=np.float64)
        energy_coords = np.arange(image.shape[0], dtype=float)
        k_coords = np.arange(image.shape[1], dtype=float)
    
    # Extract band dispersions
    dispersions = extract_band_dispersion(
        image,
        axis=axis,
        energy_coords=energy_coords,
        k_coords=k_coords,
        threshold=threshold,
        min_distance=min_distance,
        min_track_length=min_track_length,
        fit_peaks=True
    )
    
    # Convert to contour format (row, col) = (energy_idx, k_idx)
    bands = []
    for disp in dispersions:
        positions = disp['positions']  # (energy, k) pairs
        
        # Convert to pixel coordinates
        rows = np.interp(positions[:, 0], energy_coords, np.arange(len(energy_coords)))
        cols = np.interp(positions[:, 1], k_coords, np.arange(len(k_coords)))
        
        band_curve = np.column_stack([rows, cols])
        bands.append(band_curve)
    
    return bands


def extract_bands(
    data: Union[xr.DataArray, np.ndarray],
    method: Literal["ridge", "edge", "peak", "auto"] = "auto",
    preprocess: bool = True,
    enhancement_method: Literal["curvature", "second_derivative", "auto", "none"] = "auto",
    threshold: float = 0.1,
    min_length: int = 10,
    **kwargs
) -> Tuple[List[np.ndarray], Dict[str, Any]]:
    """
    Extract bands from ARPES data using the specified method.
    
    This is the main interface for band extraction.
    
    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        Input 2D ARPES data (energy × momentum/angle).
    method : str, optional
        Detection method: "ridge", "edge", "peak", or "auto".
        - "ridge": Meijering ridge filter (best for continuous bands)
        - "edge": Canny edge detection (fast, works for clear bands)
        - "peak": MDC/EDC peak fitting (most accurate, slower)
        - "auto": Automatically select based on data characteristics
        Default: "auto".
    preprocess : bool, optional
        Apply preprocessing (denoise + curvature). Default: True.
    enhancement_method : str, optional
        Curvature enhancement method. Default: "auto".
    threshold : float, optional
        Detection threshold. Default: 0.1.
    min_length : int, optional
        Minimum band length. Default: 10.
    **kwargs
        Additional method-specific parameters.
        
    Returns
    -------
    bands : list of np.ndarray
        List of detected bands. Each band is an (N, 2) array of
        (row, col) pixel coordinates.
    metadata : dict
        Extraction metadata including:
        - 'method': Detection method used
        - 'n_bands': Number of bands detected
        - 'image_shape': Original image shape
        - 'coords': Coordinate arrays if available
    """
    # Get image and coordinates
    if isinstance(data, xr.DataArray):
        image = data.values.astype(np.float64)
        dims = list(data.dims)
        coords = {
            'dim0': dims[0],
            'dim1': dims[1],
            'coord0': data.coords[dims[0]].values,
            'coord1': data.coords[dims[1]].values
        }
    else:
        image = np.asarray(data, dtype=np.float64)
        coords = {
            'dim0': 'dim0',
            'dim1': 'dim1',
            'coord0': np.arange(image.shape[0]),
            'coord1': np.arange(image.shape[1])
        }
    
    if image.ndim != 2:
        raise ValueError(f"Expected 2D data, got shape {image.shape}")
    
    original_shape = image.shape
    
    # Preprocess if requested
    if preprocess:
        processed = preprocess_data(
            image,
            denoise=True,
            enhance_contrast=True,
            enhancement_method=enhancement_method
        )
    else:
        processed = image
    
    # Auto-select method based on data
    if method == "auto":
        # Use peak method for xarray data (has coordinates)
        # Use ridge for plain arrays
        if isinstance(data, xr.DataArray):
            method = "peak"
        else:
            method = "ridge"
    
    # Detect bands
    if method == "ridge":
        bands = detect_bands_ridge(
            processed,
            threshold=threshold,
            min_length=min_length,
            **{k: v for k, v in kwargs.items() if k in ['sigmas']}
        )
    elif method == "edge":
        bands = detect_bands_edge(
            processed,
            min_length=min_length,
            **{k: v for k, v in kwargs.items() if k in ['sigma', 'low_threshold', 'high_threshold']}
        )
    elif method == "peak":
        bands = detect_bands_peak(
            data,  # Use original data for peak fitting
            threshold=threshold,
            min_track_length=min_length,
            **{k: v for k, v in kwargs.items() if k in ['min_distance', 'axis']}
        )
    else:
        raise ValueError(f"Unknown method: {method}")
    
    metadata = {
        'method': method,
        'n_bands': len(bands),
        'image_shape': original_shape,
        'coords': coords,
        'preprocessed': preprocess,
        'enhancement_method': enhancement_method
    }
    
    return bands, metadata


# =============================================================================
# Vector Export Functions
# =============================================================================

def _bands_to_svg_paths(
    bands: List[np.ndarray],
    image_shape: Tuple[int, int],
    stroke_width: float = 1.0,
    stroke_color: str = "black",
    smooth: bool = True
) -> List[str]:
    """Convert band curves to SVG path elements."""
    svg_paths = []
    height, width = image_shape
    
    for i, band in enumerate(bands):
        if len(band) < 2:
            continue
        
        # Flip y-axis for SVG coordinate system (origin at top-left)
        points = band.copy()
        points[:, 0] = height - points[:, 0]
        
        # Build SVG path
        if smooth and len(points) >= 4:
            # Use cubic Bezier curves for smooth appearance
            path_data = f"M {points[0, 1]:.2f},{points[0, 0]:.2f}"
            
            # Simple smoothing using Catmull-Rom to Bezier conversion
            for j in range(1, len(points) - 1):
                p0 = points[max(0, j-1)]
                p1 = points[j]
                p2 = points[min(len(points)-1, j+1)]
                
                # Control points
                c1 = p1 + (p1 - p0) * 0.2
                c2 = p1 - (p2 - p1) * 0.2
                
                path_data += f" S {c2[1]:.2f},{c2[0]:.2f} {p1[1]:.2f},{p1[0]:.2f}"
            
            # Final point
            path_data += f" L {points[-1, 1]:.2f},{points[-1, 0]:.2f}"
        else:
            # Simple polyline
            path_data = f"M {points[0, 1]:.2f},{points[0, 0]:.2f}"
            for p in points[1:]:
                path_data += f" L {p[1]:.2f},{p[0]:.2f}"
        
        svg_path = f'<path id="band_{i}" d="{path_data}" fill="none" stroke="{stroke_color}" stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round"/>'
        svg_paths.append(svg_path)
    
    return svg_paths


def export_to_svg(
    bands: List[np.ndarray],
    output_path: str,
    image_shape: Tuple[int, int],
    stroke_width: float = 1.5,
    stroke_color: str = "black",
    colors: Optional[List[str]] = None,
    include_background: bool = False,
    background_image: Optional[np.ndarray] = None,
    smooth: bool = True,
    viewbox_padding: float = 0.05
) -> str:
    """
    Export extracted bands to SVG format.
    
    Parameters
    ----------
    bands : list of np.ndarray
        List of band curves from extract_bands().
    output_path : str
        Output SVG file path.
    image_shape : tuple
        Original image shape (height, width).
    stroke_width : float, optional
        Line width in pixels. Default: 1.5.
    stroke_color : str, optional
        Default line color. Default: "black".
    colors : list of str, optional
        Individual colors for each band. Overrides stroke_color.
    include_background : bool, optional
        Include a faint background image. Default: False.
    background_image : np.ndarray, optional
        Image to use as background.
    smooth : bool, optional
        Use smooth curves. Default: True.
    viewbox_padding : float, optional
        Padding around viewbox as fraction. Default: 0.05.
        
    Returns
    -------
    str
        Path to the created SVG file.
    """
    height, width = image_shape
    
    # Calculate viewbox with padding
    pad_x = width * viewbox_padding
    pad_y = height * viewbox_padding
    viewbox = f"{-pad_x:.1f} {-pad_y:.1f} {width + 2*pad_x:.1f} {height + 2*pad_y:.1f}"
    
    # SVG header
    svg_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" 
     viewBox="{viewbox}"
     width="{width}" height="{height}">
  <title>ARPES Band Structure</title>
  <desc>Automatically extracted band structure from ARPES data</desc>
  
  <g id="bands">
'''
    
    # Add background if requested
    if include_background and background_image is not None:
        # Encode image as base64 PNG
        import io
        import base64
        from PIL import Image
        
        # Normalize and convert to uint8
        bg = background_image.copy()
        if bg.max() > 0:
            bg = (bg / bg.max() * 255).astype(np.uint8)
        else:
            bg = np.zeros_like(bg, dtype=np.uint8)
        
        # Flip for SVG coordinates
        bg = np.flipud(bg)
        
        # Create PIL image and encode
        img = Image.fromarray(bg, mode='L')
        buffer = io.BytesIO()
        img.save(buffer, format='PNG')
        img_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        svg_content += f'''    <image href="data:image/png;base64,{img_data}" 
           x="0" y="0" width="{width}" height="{height}" 
           opacity="0.3" preserveAspectRatio="none"/>
'''
    
    # Add bands
    for i, band in enumerate(bands):
        if len(band) < 2:
            continue
        
        # Get color for this band
        if colors and i < len(colors):
            color = colors[i]
        else:
            color = stroke_color
        
        # Flip y-axis
        points = band.copy()
        points[:, 0] = height - points[:, 0]
        
        # Build path
        if smooth and len(points) >= 4:
            path_data = f"M {points[0, 1]:.2f},{points[0, 0]:.2f}"
            for j in range(1, len(points)):
                path_data += f" L {points[j, 1]:.2f},{points[j, 0]:.2f}"
        else:
            path_data = f"M {points[0, 1]:.2f},{points[0, 0]:.2f}"
            for p in points[1:]:
                path_data += f" L {p[1]:.2f},{p[0]:.2f}"
        
        svg_content += f'''    <path id="band_{i}" d="{path_data}" 
          fill="none" stroke="{color}" stroke-width="{stroke_width}" 
          stroke-linecap="round" stroke-linejoin="round"/>
'''
    
    svg_content += '''  </g>
</svg>
'''
    
    # Write file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(svg_content)
    
    return output_path


def export_to_pdf(
    bands: List[np.ndarray],
    output_path: str,
    image_shape: Tuple[int, int],
    stroke_width: float = 1.5,
    stroke_color: str = "black",
    colors: Optional[List[str]] = None,
    include_background: bool = False,
    background_image: Optional[np.ndarray] = None,
    figsize: Optional[Tuple[float, float]] = None,
    dpi: int = 150
) -> str:
    """
    Export extracted bands to PDF format using matplotlib.
    
    Parameters
    ----------
    bands : list of np.ndarray
        List of band curves from extract_bands().
    output_path : str
        Output PDF file path.
    image_shape : tuple
        Original image shape (height, width).
    stroke_width : float, optional
        Line width in points. Default: 1.5.
    stroke_color : str, optional
        Default line color. Default: "black".
    colors : list of str, optional
        Individual colors for each band.
    include_background : bool, optional
        Include background image. Default: False.
    background_image : np.ndarray, optional
        Image to use as background.
    figsize : tuple, optional
        Figure size in inches. Default: auto-calculated.
    dpi : int, optional
        Resolution for rasterized elements. Default: 150.
        
    Returns
    -------
    str
        Path to the created PDF file.
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    
    height, width = image_shape
    
    # Auto-calculate figure size
    if figsize is None:
        max_size = 10  # inches
        aspect = width / height
        if aspect > 1:
            figsize = (max_size, max_size / aspect)
        else:
            figsize = (max_size * aspect, max_size)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add background
    if include_background and background_image is not None:
        ax.imshow(background_image, origin='lower', cmap='gray', alpha=0.3,
                  extent=[0, width, 0, height])
    
    # Plot bands
    for i, band in enumerate(bands):
        if len(band) < 2:
            continue
        
        if colors and i < len(colors):
            color = colors[i]
        else:
            color = stroke_color
        
        ax.plot(band[:, 1], band[:, 0], color=color, linewidth=stroke_width,
                solid_capstyle='round', solid_joinstyle='round')
    
    # Save
    plt.tight_layout(pad=0)
    fig.savefig(output_path, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    
    return output_path


def overlay_bands_on_data(
    data: Union[xr.DataArray, np.ndarray],
    band_result: Optional[BandResult] = None,
    bands: Optional[List[np.ndarray]] = None,
    show_uncertainty: bool = True,
    uncertainty_alpha: float = 0.3,
    uncertainty_style: Literal["band", "errorbar"] = "band",
    colors: Optional[List[str]] = None,
    linewidth: float = 1.5,
    marker: str = 'o',
    markersize: float = 4,
    cmap: str = 'terrain',
    figsize: Tuple[float, float] = (10, 8),
    title: str = "Extracted Bands"
) -> 'matplotlib.figure.Figure':
    """
    Visualize extracted bands overlaid on the original ARPES data.
    
    Supports both the new BandResult format with uncertainty visualization
    and the legacy bands list format for backwards compatibility.
    
    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        Original 2D ARPES data.
    band_result : BandResult, optional
        Result from extract_band_profiles() containing peak positions and
        uncertainties. If provided, displays bands with uncertainty regions.
    bands : list of np.ndarray, optional
        Legacy format: extracted band curves as (N, 2) arrays of (row, col)
        pixel coordinates. Used for backwards compatibility.
    show_uncertainty : bool, optional
        Whether to display uncertainty regions when using band_result.
        Default: True.
    uncertainty_alpha : float, optional
        Transparency for uncertainty bands (0-1). Default: 0.3.
    uncertainty_style : {"band", "errorbar"}, optional
        Style for displaying uncertainties:
        - "band": Shaded region around the line
        - "errorbar": Error bars at each point
        Default: "band".
    colors : list of str, optional
        Colors for each band. Default: auto color cycle.
    linewidth : float, optional
        Line width. Default: 1.5.
    marker : str, optional
        Marker style for data points. Default: 'o'.
    markersize : float, optional
        Marker size. Default: 4.
    cmap : str, optional
        Colormap for data. Default: 'terrain'.
    figsize : tuple, optional
        Figure size in inches. Default: (10, 8).
    title : str, optional
        Figure title. Default: "Extracted Bands".
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
        
    Examples
    --------
    >>> # Using new BandResult format
    >>> result = extract_band_profiles(data, n_profiles=10, n_bands=2)
    >>> fig = overlay_bands_on_data(data, band_result=result)
    
    >>> # Using legacy bands format
    >>> bands, meta = extract_bands(data, method="peak")
    >>> fig = overlay_bands_on_data(data, bands=bands)
    """
    import matplotlib.pyplot as plt
    
    # Get image and coordinates
    if isinstance(data, xr.DataArray):
        image = data.values
        dims = list(data.dims)
        extent = [
            data.coords[dims[1]].values[0],
            data.coords[dims[1]].values[-1],
            data.coords[dims[0]].values[0],
            data.coords[dims[0]].values[-1]
        ]
        xlabel = dims[1]
        ylabel = dims[0]
    else:
        image = np.asarray(data)
        extent = [0, image.shape[1], 0, image.shape[0]]
        xlabel = 'x'
        ylabel = 'y'
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot data
    im = ax.imshow(image, origin='lower', aspect='auto', 
                   extent=extent, cmap=cmap)
    plt.colorbar(im, ax=ax, label='Intensity')
    
    # Default colors
    if colors is None:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = [c['color'] for c in prop_cycle]
    
    # Plot bands based on input type
    if band_result is not None:
        # New BandResult format with uncertainty
        positions = band_result.positions
        uncertainties = band_result.uncertainties
        profile_coords = band_result.profile_coords
        
        for band_idx in range(band_result.n_bands):
            color = colors[band_idx % len(colors)]
            pos = positions[:, band_idx]
            unc = uncertainties[:, band_idx]
            
            # Filter out NaN values
            valid = ~np.isnan(pos)
            if not np.any(valid):
                continue
            
            x_valid = pos[valid]
            y_valid = profile_coords[valid]
            unc_valid = unc[valid] if show_uncertainty else None
            
            # Determine x/y based on direction
            if band_result.direction == "horizontal":
                # MDC: x-axis is slice coordinate (momentum), y-axis is profile coordinate (energy)
                plot_x = x_valid
                plot_y = y_valid
                unc_x = unc_valid
                unc_y = None
            else:
                # EDC: x-axis is profile coordinate (momentum), y-axis is slice coordinate (energy)
                plot_x = y_valid
                plot_y = x_valid
                unc_x = None
                unc_y = unc_valid
            
            # Plot the band
            ax.plot(plot_x, plot_y, color=color, linewidth=linewidth,
                   marker=marker, markersize=markersize, 
                   label=f'Band {band_idx}')
            
            # Plot uncertainty
            if show_uncertainty and unc_valid is not None and np.any(~np.isnan(unc_valid)):
                unc_valid_filled = np.nan_to_num(unc_valid, nan=0.0)
                
                if uncertainty_style == "band":
                    if band_result.direction == "horizontal":
                        ax.fill_betweenx(
                            plot_y,
                            plot_x - unc_valid_filled,
                            plot_x + unc_valid_filled,
                            alpha=uncertainty_alpha,
                            color=color
                        )
                    else:
                        ax.fill_between(
                            plot_x,
                            plot_y - unc_valid_filled,
                            plot_y + unc_valid_filled,
                            alpha=uncertainty_alpha,
                            color=color
                        )
                else:  # errorbar
                    if band_result.direction == "horizontal":
                        ax.errorbar(plot_x, plot_y, xerr=unc_valid_filled,
                                   fmt='none', ecolor=color, alpha=0.7,
                                   capsize=2)
                    else:
                        ax.errorbar(plot_x, plot_y, yerr=unc_valid_filled,
                                   fmt='none', ecolor=color, alpha=0.7,
                                   capsize=2)
    
    elif bands is not None:
        # Legacy bands format
        for i, band in enumerate(bands):
            if len(band) < 2:
                continue
            
            color = colors[i % len(colors)]
            
            # Convert pixel coordinates to data coordinates
            if isinstance(data, xr.DataArray):
                coord0 = data.coords[dims[0]].values
                coord1 = data.coords[dims[1]].values
                x = np.interp(band[:, 1], np.arange(len(coord1)), coord1)
                y = np.interp(band[:, 0], np.arange(len(coord0)), coord0)
            else:
                x = band[:, 1]
                y = band[:, 0]
            
            ax.plot(x, y, color=color, linewidth=linewidth, label=f'Band {i}')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Legend
    n_bands_total = band_result.n_bands if band_result else (len(bands) if bands else 0)
    if 0 < n_bands_total <= 10:
        ax.legend(loc='upper right')
    
    plt.tight_layout()
    return fig


# =============================================================================
# Convenience Function
# =============================================================================

def extract_and_export(
    data: Union[xr.DataArray, np.ndarray],
    output_path: str,
    output_format: Literal["svg", "pdf", "both"] = "svg",
    method: Literal["ridge", "edge", "peak", "auto"] = "auto",
    **kwargs
) -> Tuple[List[np.ndarray], Dict[str, Any], List[str]]:
    """
    One-step extraction and export of bands from ARPES data.
    
    Parameters
    ----------
    data : xr.DataArray or np.ndarray
        Input 2D ARPES data.
    output_path : str
        Output file path (without extension).
    output_format : str, optional
        "svg", "pdf", or "both". Default: "svg".
    method : str, optional
        Detection method. Default: "auto".
    **kwargs
        Additional parameters for extract_bands and export functions.
        
    Returns
    -------
    bands : list of np.ndarray
        Extracted band curves.
    metadata : dict
        Extraction metadata.
    output_files : list of str
        Paths to created output files.
        
    Examples
    --------
    >>> bands, meta, files = extract_and_export(
    ...     data, "output/bands", output_format="both"
    ... )
    >>> print(f"Extracted {meta['n_bands']} bands")
    >>> print(f"Created files: {files}")
    """
    # Extract bands
    bands, metadata = extract_bands(data, method=method, **kwargs)
    
    if len(bands) == 0:
        warnings.warn("No bands detected in the data")
        return bands, metadata, []
    
    output_files = []
    image_shape = metadata['image_shape']
    
    # Export based on format
    if output_format in ("svg", "both"):
        svg_path = output_path + ".svg"
        export_to_svg(bands, svg_path, image_shape, **kwargs)
        output_files.append(svg_path)
    
    if output_format in ("pdf", "both"):
        pdf_path = output_path + ".pdf"
        export_to_pdf(bands, pdf_path, image_shape, **kwargs)
        output_files.append(pdf_path)
    
    return bands, metadata, output_files


# =============================================================================
# Module exports
# =============================================================================

__all__ = [
    'BandResult',
    'extract_band_profiles',
    'preprocess_data',
    'detect_bands_ridge',
    'detect_bands_edge',
    'detect_bands_peak',
    'extract_bands',
    'export_to_svg',
    'export_to_pdf',
    'overlay_bands_on_data',
    'extract_and_export'
]
