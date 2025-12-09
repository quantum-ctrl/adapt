"""
Image Enhancement Module - Intensity Transformation Functions

Provides brightness, contrast, and histogram operations for scientific image processing.
All functions accept NumPy ndarrays and return ndarrays with preserved dtype.

This module uses scikit-image for high dynamic range (HDR) support, avoiding
the precision loss that occurs when converting to uint8.
"""

import numpy as np
from typing import Tuple
from skimage import exposure


def _normalize_to_float(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """
    Normalize image to 0-1 float range for processing.
    
    Returns:
        tuple: (normalized_image, original_min, original_max)
    """
    original_min = float(np.nanmin(image))
    original_max = float(np.nanmax(image))
    
    if original_max > original_min:
        normalized = (image.astype(np.float64) - original_min) / (original_max - original_min)
    else:
        normalized = np.zeros_like(image, dtype=np.float64)
    
    return normalized, original_min, original_max


def _restore_range(image: np.ndarray, original_dtype: np.dtype,
                   original_min: float, original_max: float) -> np.ndarray:
    """Restore image to original value range and dtype."""
    result = image * (original_max - original_min) + original_min
    return result.astype(original_dtype)


def histogram_equalize(image: np.ndarray, nbins: int = 256) -> np.ndarray:
    """
    Apply global histogram equalization to an image.
    
    Histogram equalization spreads out intensity values to enhance contrast,
    particularly useful for images with narrow intensity distributions.
    
    Uses pure numpy for high dynamic range support and compatibility.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as 2D (grayscale) array.
    nbins : int, optional
        Number of bins for histogram. Default is 256.
    
    Returns
    -------
    np.ndarray
        Equalized image with the same shape and dtype as input.
    
    Examples
    --------
    >>> import numpy as np
    >>> low_contrast = np.random.rand(100, 100).astype(np.float32) * 0.3 + 0.3
    >>> enhanced = histogram_equalize(low_contrast)
    """
    original_dtype = image.dtype
    normalized, orig_min, orig_max = _normalize_to_float(image)
    
    # Flatten the image for histogram computation
    flat = normalized.ravel()
    
    # Compute histogram
    hist, bin_edges = np.histogram(flat, bins=nbins, range=(0, 1))
    
    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf = cdf / cdf[-1]  # Normalize to 0-1
    
    # Create lookup table: map each bin to its CDF value
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Map input values to equalized values using interpolation
    result = np.interp(flat, bin_centers, cdf)
    result = result.reshape(normalized.shape)
    
    return _restore_range(result, original_dtype, orig_min, orig_max)


def clahe(image: np.ndarray, clip_limit: float = 0.01, 
          tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE).
    
    CLAHE divides the image into small tiles and applies histogram equalization
    to each tile with a contrast limit, preventing over-amplification of noise.
    Ideal for scientific images with varying local contrast.
    
    Uses scikit-image for high dynamic range support - works directly on
    float data without lossy uint8 conversion.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as 2D (grayscale) or 3D (color) array.
    clip_limit : float, optional
        Threshold for contrast limiting (0-1). Higher values allow more contrast.
        Default is 0.01.
    tile_grid_size : tuple of int, optional
        Size of grid for histogram equalization. Default is (8, 8).
    
    Returns
    -------
    np.ndarray
        CLAHE-enhanced image with the same shape and dtype as input.
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> enhanced = clahe(image, clip_limit=0.01)
    """
    if clip_limit <= 0:
        raise ValueError("clip_limit must be positive")
    
    original_dtype = image.dtype
    normalized, orig_min, orig_max = _normalize_to_float(image)
    
    # skimage's equalize_adapthist expects kernel_size, not tile_grid_size
    # Convert tile_grid_size to kernel_size (size of each tile in pixels)
    kernel_size = (
        max(1, image.shape[0] // tile_grid_size[0]),
        max(1, image.shape[1] // tile_grid_size[1])
    )
    
    if len(normalized.shape) == 3:
        # For color images, convert to LAB and apply CLAHE to L channel
        from skimage import color
        lab = color.rgb2lab(normalized)
        lab[:, :, 0] = exposure.equalize_adapthist(
            lab[:, :, 0] / 100.0, 
            kernel_size=kernel_size,
            clip_limit=clip_limit
        ) * 100.0
        result = color.lab2rgb(lab)
    else:
        result = exposure.equalize_adapthist(
            normalized, 
            kernel_size=kernel_size,
            clip_limit=clip_limit
        )
    
    return _restore_range(result, original_dtype, orig_min, orig_max)


def adjust_contrast(image: np.ndarray, factor: float = 1.0) -> np.ndarray:
    """
    Adjust image contrast by a scaling factor.
    
    Contrast adjustment scales pixel values around the mean intensity.
    Factor > 1 increases contrast, factor < 1 decreases contrast.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as 2D (grayscale) or 3D (color) array.
    factor : float, optional
        Contrast scaling factor. Default is 1.0 (no change).
        - factor > 1: Increase contrast
        - factor < 1: Decrease contrast
        - factor = 0: Uniform gray image
    
    Returns
    -------
    np.ndarray
        Contrast-adjusted image with the same shape and dtype as input.
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> high_contrast = adjust_contrast(image, factor=1.5)
    >>> low_contrast = adjust_contrast(image, factor=0.5)
    """
    if factor < 0:
        raise ValueError("factor must be non-negative")
    
    original_dtype = image.dtype
    normalized, orig_min, orig_max = _normalize_to_float(image)
    
    # Adjust contrast around mean (in normalized 0-1 space)
    mean = np.mean(normalized)
    result = mean + factor * (normalized - mean)
    
    # Clip to valid range
    result = np.clip(result, 0, 1)
    
    return _restore_range(result, original_dtype, orig_min, orig_max)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Apply gamma correction to an image.
    
    Gamma correction applies a non-linear transformation to adjust brightness.
    Gamma < 1 brightens the image, gamma > 1 darkens it. Commonly used to
    correct for display characteristics or enhance visibility of features.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as 2D (grayscale) or 3D (color) array.
    gamma : float, optional
        Gamma value for correction. Default is 1.0 (no change).
        - gamma < 1: Brighten image (expand dark regions)
        - gamma > 1: Darken image (compress dark regions)
    
    Returns
    -------
    np.ndarray
        Gamma-corrected image with the same shape and dtype as input.
    
    Examples
    --------
    >>> import numpy as np
    >>> dark_image = np.random.rand(100, 100).astype(np.float32) ** 2
    >>> brightened = gamma_correction(dark_image, gamma=0.5)
    """
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    
    original_dtype = image.dtype
    normalized, orig_min, orig_max = _normalize_to_float(image)
    
    # Apply gamma correction directly on normalized float data
    inv_gamma = 1.0 / gamma
    result = np.power(normalized, inv_gamma)
    
    return _restore_range(result, original_dtype, orig_min, orig_max)


__all__ = [
    'histogram_equalize',
    'clahe',
    'adjust_contrast',
    'gamma_correction'
]
