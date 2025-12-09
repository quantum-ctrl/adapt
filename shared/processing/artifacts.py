"""
CCD Artifact Removal Module for ARPES Data

This module provides functions to remove sensor-specific artifacts that are not
physical ARPES signals:
- Hot pixels (isolated bright spikes)
- Dead pixels (zero or constant-value pixels)
- Horizontal stripes (scanning streaks)
- Vertical stripes (column artifacts)
- Salt-and-pepper noise (impulse noise)

All functions operate on raw NumPy arrays in the intensity domain and preserve
the original data type. They support both 2D (H×W) and 3D (H×W×E) arrays.

Usage:
    import numpy as np
    from processing.artifacts import remove_hot_pixels, destripe_horizontal
    
    # Remove hot pixels
    cleaned = remove_hot_pixels(image, threshold=5.0)
    
    # Remove horizontal stripes
    destriped = destripe_horizontal(image, strength=0.1)
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import median_filter, uniform_filter1d
from typing import Tuple, Union, Optional


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_input(image: NDArray) -> None:
    """Validate input array dimensions and dtype."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(image)}")
    if image.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")
    if image.dtype not in (np.uint16, np.float32, np.float64):
        # Convert to float64 for safe processing
        pass  # Will be handled in individual functions


def _apply_to_slices(
    image: NDArray, 
    func, 
    **kwargs
) -> NDArray:
    """
    Apply a 2D function to each slice of a 3D array along the last axis.
    
    Parameters
    ----------
    image : NDArray
        Input 2D or 3D array
    func : callable
        Function that operates on 2D arrays
    **kwargs : dict
        Additional arguments passed to func
    
    Returns
    -------
    NDArray
        Processed array with same shape as input
    """
    if image.ndim == 2:
        return func(image, **kwargs)
    elif image.ndim == 3:
        result = np.empty_like(image)
        for i in range(image.shape[2]):
            result[:, :, i] = func(image[:, :, i], **kwargs)
        return result
    else:
        raise ValueError(f"Expected 2D or 3D array, got {image.ndim}D")


def _estimate_noise_std(image: NDArray) -> float:
    """
    Estimate noise standard deviation using Median Absolute Deviation (MAD).
    
    This is a robust estimator less sensitive to outliers than np.std.
    
    Parameters
    ----------
    image : NDArray
        Input 2D array
    
    Returns
    -------
    float
        Estimated noise standard deviation
    """
    # Use difference of adjacent pixels to isolate noise
    diff = np.diff(image.astype(np.float64), axis=0)
    mad = np.median(np.abs(diff - np.median(diff)))
    # Convert MAD to std (for Gaussian: std ≈ 1.4826 * MAD)
    return float(1.4826 * mad / np.sqrt(2))


def _preserve_dtype(func):
    """Decorator to preserve input dtype after processing."""
    def wrapper(image: NDArray, *args, **kwargs) -> NDArray:
        original_dtype = image.dtype
        # Convert to float64 for processing
        working_image = image.astype(np.float64)
        result = func(working_image, *args, **kwargs)
        
        # Handle tuple returns (e.g., return_mask=True)
        if isinstance(result, tuple):
            processed, *rest = result
        else:
            processed = result
            rest = None
        
        # Restore original dtype
        if original_dtype == np.uint16:
            processed = np.clip(processed, 0, 65535).astype(np.uint16)
        elif original_dtype == np.float32:
            processed = processed.astype(np.float32)
        else:
            processed = processed.astype(original_dtype)
        
        if rest is not None:
            return (processed, *rest)
        return processed
    wrapper.__doc__ = func.__doc__
    wrapper.__name__ = func.__name__
    return wrapper


# =============================================================================
# Hot Pixel Removal
# =============================================================================

def _remove_hot_pixels_2d(
    image: NDArray,
    threshold: float = 5.0,
    return_mask: bool = False
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    """Remove hot pixels from a 2D image."""
    # Compute local median
    local_median = median_filter(image, size=3)
    
    # Estimate noise level
    noise_std = _estimate_noise_std(image)
    if noise_std < 1e-10:
        noise_std = np.std(image) * 0.1  # Fallback
    
    # Detect hot pixels: significantly brighter than local median
    deviation = image - local_median
    hot_mask = deviation > threshold * noise_std
    
    # Replace hot pixels with local median
    result = image.copy()
    result[hot_mask] = local_median[hot_mask]
    
    if return_mask:
        return result, hot_mask
    return result


@_preserve_dtype
def remove_hot_pixels(
    image: NDArray,
    threshold: float = 5.0,
    return_mask: bool = False
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    """
    Detect and remove isolated single-pixel spikes (hot pixels).
    
    Hot pixels are identified as pixels that significantly exceed the local
    median intensity. They are replaced with the local median value.
    
    Parameters
    ----------
    image : NDArray
        Input 2D (H×W) or 3D (H×W×E) array
    threshold : float, optional
        Number of standard deviations above local median to classify as hot.
        Default is 5.0.
    return_mask : bool, optional
        If True, also return the detection mask. Default is False.
    
    Returns
    -------
    NDArray
        Cleaned image with hot pixels removed
    NDArray (optional)
        Boolean mask of detected hot pixels (if return_mask=True)
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> image[50, 50] = 100  # Add hot pixel
    >>> cleaned = remove_hot_pixels(image, threshold=5.0)
    """
    _validate_input(image)
    
    if image.ndim == 2:
        return _remove_hot_pixels_2d(image, threshold, return_mask)
    else:
        # For 3D, apply to each slice
        if return_mask:
            results = []
            masks = []
            for i in range(image.shape[2]):
                r, m = _remove_hot_pixels_2d(image[:, :, i], threshold, True)
                results.append(r)
                masks.append(m)
            return np.stack(results, axis=2), np.stack(masks, axis=2)
        else:
            return _apply_to_slices(image, _remove_hot_pixels_2d, 
                                    threshold=threshold, return_mask=False)


# =============================================================================
# Dead Pixel Correction
# =============================================================================

def _remove_dead_pixels_2d(
    image: NDArray,
    return_mask: bool = False
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    """Remove dead pixels from a 2D image."""
    # Detect dead pixels: zero or extremely low constant values
    # Also check for isolated low values (could be stuck pixels)
    
    # Method 1: Absolute zero detection
    zero_mask = image == 0
    
    # Method 2: Values much lower than local neighbors
    local_median = median_filter(image, size=3)
    local_min = np.minimum.reduce([
        np.roll(image, 1, axis=0),
        np.roll(image, -1, axis=0),
        np.roll(image, 1, axis=1),
        np.roll(image, -1, axis=1)
    ])
    
    # Pixel is dead if it's zero AND neighbors are non-zero
    # Or if it's significantly lower than all neighbors
    neighbor_median = local_median
    isolated_low = (image < neighbor_median * 0.01) & (neighbor_median > 0)
    
    dead_mask = zero_mask | isolated_low
    
    # Exclude edges where detection is unreliable
    dead_mask[0, :] = False
    dead_mask[-1, :] = False
    dead_mask[:, 0] = False
    dead_mask[:, -1] = False
    
    # Replace dead pixels with local median (interpolation from neighbors)
    result = image.copy()
    result[dead_mask] = local_median[dead_mask]
    
    if return_mask:
        return result, dead_mask
    return result


@_preserve_dtype
def remove_dead_pixels(
    image: NDArray,
    return_mask: bool = False
) -> Union[NDArray, Tuple[NDArray, NDArray]]:
    """
    Replace zero or constant-value pixels using neighborhood interpolation.
    
    Dead pixels are identified as pixels with zero value or values significantly
    lower than their neighbors. They are replaced with the local median.
    
    Parameters
    ----------
    image : NDArray
        Input 2D (H×W) or 3D (H×W×E) array
    return_mask : bool, optional
        If True, also return the detection mask. Default is False.
    
    Returns
    -------
    NDArray
        Corrected image with dead pixels interpolated
    NDArray (optional)
        Boolean mask of detected dead pixels (if return_mask=True)
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32) + 1
    >>> image[50, 50] = 0  # Add dead pixel
    >>> corrected = remove_dead_pixels(image)
    """
    _validate_input(image)
    
    if image.ndim == 2:
        return _remove_dead_pixels_2d(image, return_mask)
    else:
        if return_mask:
            results = []
            masks = []
            for i in range(image.shape[2]):
                r, m = _remove_dead_pixels_2d(image[:, :, i], True)
                results.append(r)
                masks.append(m)
            return np.stack(results, axis=2), np.stack(masks, axis=2)
        else:
            return _apply_to_slices(image, _remove_dead_pixels_2d, 
                                    return_mask=False)


# =============================================================================
# Horizontal Stripe Removal
# =============================================================================

def _destripe_horizontal_2d(
    image: NDArray,
    strength: float = 0.1,
    window: int = 51
) -> NDArray:
    """Remove horizontal stripes from a 2D image."""
    # Compute row-wise mean profile
    row_mean = np.mean(image, axis=1, keepdims=True)
    
    # High-pass filter: subtract smoothed profile to get stripe pattern
    smoothed = uniform_filter1d(row_mean.flatten(), size=window, mode='reflect')
    smoothed = smoothed.reshape(-1, 1)
    
    # Stripe pattern is the difference between actual and smoothed row means
    stripe_pattern = row_mean - smoothed
    
    # Subtract stripe pattern with controlled strength
    result = image - strength * stripe_pattern
    
    return result


@_preserve_dtype
def destripe_horizontal(
    image: NDArray,
    strength: float = 0.1,
    window: int = 51
) -> NDArray:
    """
    Remove horizontal scanning streaks via row-wise high-pass filtering.
    
    Horizontal stripes are detected by computing row means and subtracting
    a smoothed version to isolate the stripe pattern.
    
    Parameters
    ----------
    image : NDArray
        Input 2D (H×W) or 3D (H×W×E) array
    strength : float, optional
        Correction strength in range [0, 1]. Higher values remove more
        stripe artifacts but may affect real features. Default is 0.1.
    window : int, optional
        Smoothing window size for high-pass filter. Should be odd.
        Default is 51.
    
    Returns
    -------
    NDArray
        Image with horizontal stripes removed
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> # Add horizontal stripes
    >>> image[::10, :] += 0.5
    >>> destriped = destripe_horizontal(image, strength=0.5)
    """
    _validate_input(image)
    
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in range [0, 1]")
    
    return _apply_to_slices(image, _destripe_horizontal_2d, 
                            strength=strength, window=window)


# =============================================================================
# Vertical Stripe Removal
# =============================================================================

def _destripe_vertical_2d(
    image: NDArray,
    strength: float = 0.1,
    window: int = 51
) -> NDArray:
    """Remove vertical stripes from a 2D image."""
    # Compute column-wise mean profile
    col_mean = np.mean(image, axis=0, keepdims=True)
    
    # High-pass filter: subtract smoothed profile to get stripe pattern
    smoothed = uniform_filter1d(col_mean.flatten(), size=window, mode='reflect')
    smoothed = smoothed.reshape(1, -1)
    
    # Stripe pattern is the difference between actual and smoothed column means
    stripe_pattern = col_mean - smoothed
    
    # Subtract stripe pattern with controlled strength
    result = image - strength * stripe_pattern
    
    return result


@_preserve_dtype
def destripe_vertical(
    image: NDArray,
    strength: float = 0.1,
    window: int = 51
) -> NDArray:
    """
    Remove vertical column artifacts via column-wise high-pass filtering.
    
    Vertical stripes are detected by computing column means and subtracting
    a smoothed version to isolate the stripe pattern.
    
    Parameters
    ----------
    image : NDArray
        Input 2D (H×W) or 3D (H×W×E) array
    strength : float, optional
        Correction strength in range [0, 1]. Higher values remove more
        stripe artifacts but may affect real features. Default is 0.1.
    window : int, optional
        Smoothing window size for high-pass filter. Should be odd.
        Default is 51.
    
    Returns
    -------
    NDArray
        Image with vertical stripes removed
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> # Add vertical stripes
    >>> image[:, ::10] += 0.5
    >>> destriped = destripe_vertical(image, strength=0.5)
    """
    _validate_input(image)
    
    if strength < 0 or strength > 1:
        raise ValueError("strength must be in range [0, 1]")
    
    return _apply_to_slices(image, _destripe_vertical_2d, 
                            strength=strength, window=window)


# =============================================================================
# Salt-and-Pepper Noise Removal
# =============================================================================

def _remove_salt_pepper_2d(
    image: NDArray,
    kernel_size: int = 3
) -> NDArray:
    """Remove salt-and-pepper noise from a 2D image."""
    # Use adaptive median filtering to preserve edges better
    # Standard median filter as baseline
    filtered = median_filter(image, size=kernel_size)
    
    # Only replace pixels that are extreme outliers (salt or pepper)
    # This helps preserve band edges
    local_min = median_filter(image, size=kernel_size, mode='nearest')
    local_max = local_min.copy()
    
    # Compute local statistics
    local_median = filtered
    noise_std = _estimate_noise_std(image)
    
    # Identify salt (very bright) and pepper (very dark) pixels
    threshold = 4.0 * max(noise_std, 1e-6)
    salt_mask = (image - local_median) > threshold
    pepper_mask = (local_median - image) > threshold
    noise_mask = salt_mask | pepper_mask
    
    # Replace only noisy pixels, keep others unchanged
    result = image.copy()
    result[noise_mask] = local_median[noise_mask]
    
    return result


@_preserve_dtype
def remove_salt_pepper_noise(
    image: NDArray,
    kernel_size: int = 3
) -> NDArray:
    """
    Apply selective median filtering to remove salt-and-pepper noise.
    
    This function uses an adaptive approach that only modifies pixels
    identified as impulse noise, preserving band edges and real features.
    
    Parameters
    ----------
    image : NDArray
        Input 2D (H×W) or 3D (H×W×E) array
    kernel_size : int, optional
        Size of the median filter kernel. Must be odd. Default is 3.
    
    Returns
    -------
    NDArray
        Image with salt-and-pepper noise removed
    
    Examples
    --------
    >>> import numpy as np
    >>> image = np.random.rand(100, 100).astype(np.float32)
    >>> # Add salt-and-pepper noise
    >>> noise_mask = np.random.random((100, 100)) < 0.05
    >>> image[noise_mask] = np.random.choice([0, 1], noise_mask.sum())
    >>> cleaned = remove_salt_pepper_noise(image, kernel_size=3)
    """
    _validate_input(image)
    
    if kernel_size < 1:
        raise ValueError("kernel_size must be positive")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    
    return _apply_to_slices(image, _remove_salt_pepper_2d, 
                            kernel_size=kernel_size)


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'remove_hot_pixels',
    'remove_dead_pixels',
    'destripe_horizontal',
    'destripe_vertical',
    'remove_salt_pepper_noise'
]


# =============================================================================
# Test Section
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("CCD Artifact Removal Module - Quick Tests")
    print("=" * 60)
    
    # Create synthetic test image: Gaussian band
    print("\n1. Creating synthetic ARPES-like image...")
    
    np.random.seed(42)
    height, width = 200, 150
    
    # Create a parabolic band structure
    x = np.linspace(-2, 2, width)
    y = np.linspace(0, 5, height)
    X, Y = np.meshgrid(x, y)
    
    # Parabolic dispersion: E = E0 + k^2 / 2m*
    band_center = 2.5 + 0.3 * X**2
    band_width = 0.3
    image = np.exp(-((Y - band_center) / band_width)**2)
    
    # Add some noise
    image += 0.05 * np.random.randn(height, width)
    image = np.clip(image, 0, None)
    
    # Add artifacts
    print("2. Adding synthetic artifacts...")
    
    # Hot pixels
    n_hot = 20
    hot_y = np.random.randint(0, height, n_hot)
    hot_x = np.random.randint(0, width, n_hot)
    image_hot = image.copy()
    image_hot[hot_y, hot_x] = image.max() * 5
    
    # Dead pixels
    n_dead = 15
    dead_y = np.random.randint(0, height, n_dead)
    dead_x = np.random.randint(0, width, n_dead)
    image_dead = image_hot.copy()
    image_dead[dead_y, dead_x] = 0
    
    # Horizontal stripes
    image_striped = image_dead.copy()
    stripe_rows = np.random.randint(0, height, 10)
    image_striped[stripe_rows, :] += 0.3
    
    # Salt-and-pepper
    image_noisy = image_striped.copy()
    sp_mask = np.random.random((height, width)) < 0.02
    image_noisy[sp_mask] = np.where(
        np.random.random(sp_mask.sum()) > 0.5,
        image.max() * 2,
        0
    )
    
    # Test each function
    print("\n3. Testing artifact removal functions...")
    
    # Test hot pixel removal
    cleaned_hot, hot_mask = remove_hot_pixels(image_noisy, threshold=5.0, 
                                               return_mask=True)
    print(f"   - Hot pixels detected: {hot_mask.sum()}")
    
    # Test dead pixel removal
    cleaned_dead, dead_mask = remove_dead_pixels(cleaned_hot, return_mask=True)
    print(f"   - Dead pixels detected: {dead_mask.sum()}")
    
    # Test horizontal destriping
    cleaned_h = destripe_horizontal(cleaned_dead, strength=0.5)
    print(f"   - Horizontal destriping applied")
    
    # Test vertical destriping
    cleaned_v = destripe_vertical(cleaned_h, strength=0.1)
    print(f"   - Vertical destriping applied")
    
    # Test salt-pepper removal
    cleaned_sp = remove_salt_pepper_noise(cleaned_v, kernel_size=3)
    print(f"   - Salt-pepper noise removed")
    
    # Test 3D array
    print("\n4. Testing 3D array support...")
    image_3d = np.stack([image_noisy, image_noisy * 0.8, image_noisy * 0.6], axis=2)
    cleaned_3d = remove_hot_pixels(image_3d, threshold=5.0)
    print(f"   - 3D input shape: {image_3d.shape}")
    print(f"   - 3D output shape: {cleaned_3d.shape}")
    
    # Test dtype preservation
    print("\n5. Testing dtype preservation...")
    for dtype in [np.float32, np.float64, np.uint16]:
        test_img = (image_noisy * 1000).astype(dtype)
        result = remove_hot_pixels(test_img)
        print(f"   - {dtype.__name__}: input dtype preserved = {result.dtype == dtype}")
    
    # Visualization
    print("\n6. Attempting visualization...")
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        
        axes[0, 0].imshow(image, aspect='auto', cmap='viridis')
        axes[0, 0].set_title('Original (clean)')
        
        axes[0, 1].imshow(image_noisy, aspect='auto', cmap='viridis')
        axes[0, 1].set_title('With artifacts')
        
        axes[0, 2].imshow(cleaned_sp, aspect='auto', cmap='viridis')
        axes[0, 2].set_title('After cleaning')
        
        # Show difference
        axes[1, 0].imshow(image_noisy - image, aspect='auto', cmap='RdBu_r', 
                          vmin=-0.5, vmax=0.5)
        axes[1, 0].set_title('Artifacts (added noise)')
        
        axes[1, 1].imshow(cleaned_sp - image, aspect='auto', cmap='RdBu_r',
                          vmin=-0.5, vmax=0.5)
        axes[1, 1].set_title('Residual after cleaning')
        
        # Combined mask
        axes[1, 2].imshow(hot_mask.astype(int) + dead_mask.astype(int) * 2, 
                          aspect='auto', cmap='hot')
        axes[1, 2].set_title('Detection masks')
        
        plt.tight_layout()
        plt.savefig('artifacts_test_result.png', dpi=150)
        print("   - Saved visualization to 'artifacts_test_result.png'")
        plt.show()
        
    except ImportError:
        print("   - matplotlib not available, skipping visualization")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
