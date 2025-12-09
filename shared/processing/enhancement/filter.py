"""
Image Enhancement Module - Spatial Filtering Functions

Provides edge enhancement and sharpening algorithms for scientific image processing.
All functions accept NumPy ndarrays and return ndarrays with preserved dtype.
"""

import numpy as np
import cv2
from typing import Literal


def _ensure_uint8(image: np.ndarray) -> tuple[np.ndarray, bool, float, float]:
    """
    Convert image to uint8 for OpenCV processing.
    
    Returns:
        tuple: (converted_image, was_float, original_min, original_max)
    """
    was_float = image.dtype in [np.float32, np.float64]
    
    if was_float:
        original_min, original_max = image.min(), image.max()
        if original_max > original_min:
            normalized = (image - original_min) / (original_max - original_min)
        else:
            normalized = np.zeros_like(image)
        return (normalized * 255).astype(np.uint8), True, original_min, original_max
    elif image.dtype == np.uint16:
        return (image / 256).astype(np.uint8), False, 0, 0
    else:
        return image.astype(np.uint8), False, 0, 0


def _restore_dtype(image: np.ndarray, original_dtype: np.dtype, 
                   was_float: bool, original_min: float, original_max: float) -> np.ndarray:
    """Restore image to original dtype after processing."""
    if was_float:
        result = image.astype(np.float64) / 255.0
        result = result * (original_max - original_min) + original_min
        return result.astype(original_dtype)
    elif original_dtype == np.uint16:
        return (image.astype(np.uint16) * 256)
    else:
        return image.astype(original_dtype)


def sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Apply unsharp masking to sharpen an image.
    
    Sharpening enhances edges and fine details by subtracting a blurred
    version of the image from the original.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as 2D (grayscale) or 3D (color) array.
    strength : float, optional
        Sharpening strength. Values > 1 increase sharpening effect.
        Default is 1.0.
    
    Returns
    -------
    np.ndarray
        Sharpened image with the same shape and dtype as input.
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> sharpened = sharpen(image, strength=1.5)
    """
    original_dtype = image.dtype
    image_uint8, was_float, orig_min, orig_max = _ensure_uint8(image)
    
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image_uint8, (0, 0), 3)
    
    # Unsharp masking: sharpened = original + strength * (original - blurred)
    sharpened = cv2.addWeighted(
        image_uint8, 1.0 + strength,
        blurred, -strength,
        0
    )
    
    return _restore_dtype(sharpened, original_dtype, was_float, orig_min, orig_max)


def edge_enhance(image: np.ndarray, 
                 method: Literal["sobel", "canny", "laplacian"] = "sobel") -> np.ndarray:
    """
    Apply edge detection/enhancement to an image.
    
    Edge enhancement highlights boundaries and structural features in images,
    useful for analyzing features in scientific microscopy data.
    
    Parameters
    ----------
    image : np.ndarray
        Input image as 2D (grayscale) or 3D (color) array.
        Color images will be converted to grayscale for edge detection.
    method : str, optional
        Edge detection method. One of:
        - "sobel": Sobel operator (gradient-based, good for directional edges)
        - "canny": Canny edge detector (multi-stage, precise edge detection)
        - "laplacian": Laplacian operator (second-order derivative, isotropic)
        Default is "sobel".
    
    Returns
    -------
    np.ndarray
        Edge-enhanced image. For Canny, returns binary edge map.
        For Sobel and Laplacian, returns gradient magnitude.
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> edges = edge_enhance(image, method="canny")
    """
    valid_methods = ["sobel", "canny", "laplacian"]
    if method not in valid_methods:
        raise ValueError(f"method must be one of {valid_methods}, got '{method}'")
    
    original_dtype = image.dtype
    image_uint8, was_float, orig_min, orig_max = _ensure_uint8(image)
    
    # Convert to grayscale if color image
    if len(image_uint8.shape) == 3:
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_uint8
    
    if method == "sobel":
        # Compute Sobel gradients in x and y directions
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        # Compute gradient magnitude
        magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        # Normalize to 0-255 range
        result = np.clip(magnitude / magnitude.max() * 255, 0, 255).astype(np.uint8)
        
    elif method == "canny":
        # Canny edge detection with automatic thresholds
        median_val = np.median(gray)
        lower = int(max(0, 0.7 * median_val))
        upper = int(min(255, 1.3 * median_val))
        result = cv2.Canny(gray, lower, upper)
        
    elif method == "laplacian":
        # Laplacian edge detection
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        # Take absolute value and normalize
        result = np.abs(laplacian)
        if result.max() > 0:
            result = (result / result.max() * 255).astype(np.uint8)
        else:
            result = result.astype(np.uint8)
    
    return _restore_dtype(result, original_dtype, was_float, orig_min, orig_max)


__all__ = [
    'sharpen',
    'edge_enhance'
]
