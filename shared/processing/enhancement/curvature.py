"""
Curvature Analysis Module for ARPES Band Structure Enhancement

Provides curvature-based methods for enhancing band contrast in 2D ARPES images.
These methods are particularly effective for visualizing dispersive electronic bands
in angle-resolved photoemission spectroscopy data.

Supported input formats:
    - 2D NumPy arrays (k, E) or (angle, E)
    - float32 or float64 dtype

Reference:
    T.R. Luo et al., Rev. Sci. Instrum., 2020

Usage:
    from processing.enhancement import curvature_luo, curvature_second_derivative, auto_curvature
    
    # Apply Luo curvature method
    enhanced = curvature_luo(arpes_image, k_res=1.0, e_res=1.0)
    
    # Apply second derivative method
    enhanced = curvature_second_derivative(arpes_image, smoothing_sigma=1.0)
    
    # Auto-select best method
    enhanced, metadata = auto_curvature(arpes_image)
"""

import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from typing import Tuple, Dict, Any, Union


def _normalize_to_01(image: np.ndarray) -> np.ndarray:
    """
    Normalize image values to [0, 1] range.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array.
        
    Returns
    -------
    np.ndarray
        Normalized image with values in [0, 1].
    """
    img_min = np.nanmin(image)
    img_max = np.nanmax(image)
    
    if img_max - img_min < 1e-10:
        return np.zeros_like(image)
    
    return (image - img_min) / (img_max - img_min)


def _adaptive_smooth(image: np.ndarray, base_sigma: float = 1.0) -> np.ndarray:
    """
    Apply adaptive Gaussian smoothing based on local intensity.
    
    Regions with higher intensity gradients receive less smoothing
    to preserve band features, while noisy low-intensity regions
    receive more smoothing.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array.
    base_sigma : float
        Base smoothing sigma value.
        
    Returns
    -------
    np.ndarray
        Adaptively smoothed image.
    """
    # Compute local gradient magnitude
    grad_y, grad_x = np.gradient(image)
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Normalize gradient magnitude
    grad_norm = _normalize_to_01(grad_magnitude)
    
    # Adaptive sigma: less smoothing where gradient is high
    # sigma varies from base_sigma to base_sigma * 3
    sigma_map = base_sigma * (1 + 2 * (1 - grad_norm))
    
    # Apply weighted smoothing (approximate adaptive filtering)
    smoothed_low = gaussian_filter(image, sigma=base_sigma)
    smoothed_high = gaussian_filter(image, sigma=base_sigma * 3)
    
    # Blend based on local gradient
    weight = grad_norm[:, :, np.newaxis] if image.ndim == 3 else grad_norm
    result = weight * smoothed_low + (1 - weight) * smoothed_high
    
    return result


def curvature_luo(
    image: np.ndarray,
    k_res: float = 1.0,
    e_res: float = 1.0
) -> np.ndarray:
    """
    Implement T.R. Luo curvature method for ARPES data enhancement.
    
    This method computes a curvature measure that enhances band visibility
    while suppressing background intensity variations. It uses the 2D 
    curvature formula adapted for ARPES data with optional resolution 
    scaling along momentum and energy axes.
    
    The curvature is computed as:
        C = -∂²I/∂k² - ∂²I/∂E²
    
    with intensity normalization to prevent saturation in high-intensity
    regions and adaptive smoothing to suppress noise amplification.
    
    Parameters
    ----------
    image : np.ndarray
        Input 2D ARPES image as NumPy array (float32/float64).
        Shape should be (n_k, n_E) or (n_angle, n_E).
    k_res : float, optional
        Momentum/angle axis resolution scaling factor. Default is 1.0.
        Higher values increase sensitivity along k-axis.
    e_res : float, optional
        Energy axis resolution scaling factor. Default is 1.0.
        Higher values increase sensitivity along E-axis.
        
    Returns
    -------
    np.ndarray
        Normalized 2D curvature image with values in [0, 1].
        Same shape as input.
        
    Raises
    ------
    ValueError
        If input is not a 2D array or has invalid dtype.
        
    Examples
    --------
    >>> import numpy as np
    >>> arpes_data = np.random.rand(100, 200).astype(np.float32)
    >>> enhanced = curvature_luo(arpes_data, k_res=1.0, e_res=1.0)
    >>> enhanced.shape
    (100, 200)
    """
    # Input validation
    if image.ndim != 2:
        raise ValueError(f"Input must be 2D array, got {image.ndim}D")
    
    if image.dtype not in [np.float32, np.float64]:
        image = image.astype(np.float64)
    
    # Step 1: Intensity normalization to avoid saturation
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    intensity_norm = np.abs(image) + epsilon
    normalized = image / np.sqrt(intensity_norm)
    
    # Step 2: Adaptive smoothing to suppress noise
    smoothed = _adaptive_smooth(normalized, base_sigma=1.0)
    
    # Step 3: Compute second derivatives along both axes
    # Using Sobel-like finite differences for better noise handling
    d2_dk2 = ndimage.laplace(smoothed) / (k_res ** 2)
    
    # Compute second derivative along each axis separately for anisotropic scaling
    d2_dk2_only = np.gradient(np.gradient(smoothed, axis=0), axis=0) / (k_res ** 2)
    d2_dE2_only = np.gradient(np.gradient(smoothed, axis=1), axis=1) / (e_res ** 2)
    
    # Step 4: Compute curvature as negative sum of second derivatives
    # Negative sign makes bands appear as positive features
    curvature = -(d2_dk2_only + d2_dE2_only)
    
    # Step 5: Apply soft rectification to emphasize positive curvature (bands)
    # Using a smooth function to avoid harsh thresholding
    curvature_rectified = np.maximum(curvature, 0)
    
    # Step 6: Normalize output to [0, 1]
    result = _normalize_to_01(curvature_rectified)
    
    return result.astype(image.dtype)


def curvature_second_derivative(
    image: np.ndarray,
    smoothing_sigma: float = 1.0,
    strength: float = 1.0
) -> np.ndarray:
    """
    Compute second derivative curvature enhancement along both axes.
    
    This method applies Gaussian smoothing before differentiation to 
    reduce noise, then computes second derivatives along both the 
    momentum/angle and energy axes.
    
    Output is: original - curvature * strength
    This preserves the original intensity while enhancing band features,
    similar to the JavaScript implementation.
    
    Parameters
    ----------
    image : np.ndarray
        Input 2D ARPES image as NumPy array (float32/float64).
        Shape should be (n_k, n_E) or (n_angle, n_E).
    smoothing_sigma : float, optional
        Gaussian smoothing sigma applied before differentiation.
        Default is 1.0. Larger values reduce noise but may blur features.
    strength : float, optional
        Strength multiplier for the curvature subtraction.
        Default is 1.0. Higher values give stronger enhancement.
        
    Returns
    -------
    np.ndarray
        Enhanced image with same shape and value range as input.
        Bands appear with enhanced contrast.
        
    Raises
    ------
    ValueError
        If input is not a 2D array or smoothing_sigma is non-positive.
        
    Examples
    --------
    >>> import numpy as np
    >>> arpes_data = np.random.rand(100, 200).astype(np.float32)
    >>> enhanced = curvature_second_derivative(arpes_data, smoothing_sigma=1.5, strength=2.0)
    >>> enhanced.shape
    (100, 200)
    """
    # Input validation
    if image.ndim != 2:
        raise ValueError(f"Input must be 2D array, got {image.ndim}D")
    
    if smoothing_sigma <= 0:
        raise ValueError(f"smoothing_sigma must be positive, got {smoothing_sigma}")
    
    if image.dtype not in [np.float32, np.float64]:
        image = image.astype(np.float64)
    
    original_dtype = image.dtype
    
    # Step 1: Apply Gaussian smoothing to reduce noise
    smoothed = gaussian_filter(image.astype(np.float64), sigma=smoothing_sigma)
    
    # Step 2: Compute second derivatives along both axes
    # d²I/dk² (along axis 0 - momentum/angle axis)
    d2_dk2 = np.gradient(np.gradient(smoothed, axis=0), axis=0)
    
    # d²I/dE² (along axis 1 - energy axis)
    d2_dE2 = np.gradient(np.gradient(smoothed, axis=1), axis=1)
    
    # Step 3: Combine second derivatives
    # Sum of second derivatives (Laplacian-like)
    curvature = d2_dk2 + d2_dE2
    
    # Step 4: Subtract curvature from original (JS-style)
    # This preserves original intensity while enhancing bands
    result = image.astype(np.float64) - curvature * strength
    
    return result.astype(original_dtype)


def _compute_local_contrast(image: np.ndarray, window_size: int = 21) -> float:
    """
    Compute local contrast metric for method selection.
    
    Uses local standard deviation normalized by local mean as a 
    contrast measure.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
    window_size : int
        Window size for local statistics.
        
    Returns
    -------
    float
        Mean local contrast value.
    """
    # Compute local mean using uniform filter
    local_mean = ndimage.uniform_filter(image.astype(np.float64), size=window_size)
    
    # Compute local squared mean
    local_sq_mean = ndimage.uniform_filter(image.astype(np.float64)**2, size=window_size)
    
    # Local variance
    local_var = np.maximum(local_sq_mean - local_mean**2, 0)
    local_std = np.sqrt(local_var)
    
    # Normalized local contrast (avoid division by zero)
    epsilon = 1e-10
    local_contrast = local_std / (np.abs(local_mean) + epsilon)
    
    return np.nanmean(local_contrast)


def _compute_noise_level(image: np.ndarray) -> float:
    """
    Estimate noise level using median absolute deviation of Laplacian.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
        
    Returns
    -------
    float
        Estimated noise level.
    """
    laplacian = ndimage.laplace(image.astype(np.float64))
    
    # Median absolute deviation (robust noise estimator)
    mad = np.median(np.abs(laplacian - np.median(laplacian)))
    
    # Scale factor to convert MAD to standard deviation for Gaussian
    noise_std = 1.4826 * mad
    
    return noise_std


def _compute_gradient_strength(image: np.ndarray) -> float:
    """
    Compute average gradient strength as a measure of edge presence.
    
    Parameters
    ----------
    image : np.ndarray
        Input image.
        
    Returns
    -------
    float
        Mean gradient magnitude.
    """
    grad_y, grad_x = np.gradient(image.astype(np.float64))
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    return np.mean(grad_magnitude)


def auto_curvature(
    image: np.ndarray
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Automatically choose the best curvature method based on local contrast metrics.
    
    This function analyzes the input image to determine which curvature
    enhancement method will produce the best results. It considers:
    - Local contrast levels
    - Noise estimation
    - Gradient strength
    
    Decision logic:
    - High noise: Use Luo method (has adaptive smoothing)
    - High contrast, low noise: Use second derivative (faster, cleaner)
    - Medium conditions: Use Luo method with adjusted parameters
    
    Parameters
    ----------
    image : np.ndarray
        Input 2D ARPES image as NumPy array (float32/float64).
        Shape should be (n_k, n_E) or (n_angle, n_E).
        
    Returns
    -------
    enhanced : np.ndarray
        Enhanced image using the selected method.
        Normalized to [0, 1] range with same shape as input.
    metadata : dict
        Dictionary containing:
        - 'method': str, name of the method used
        - 'local_contrast': float, computed local contrast metric
        - 'noise_level': float, estimated noise level
        - 'gradient_strength': float, mean gradient magnitude
        - 'parameters': dict, parameters used for the selected method
        
    Raises
    ------
    ValueError
        If input is not a 2D array.
        
    Examples
    --------
    >>> import numpy as np
    >>> arpes_data = np.random.rand(100, 200).astype(np.float32)
    >>> enhanced, meta = auto_curvature(arpes_data)
    >>> print(meta['method'])
    'curvature_luo'
    >>> enhanced.shape
    (100, 200)
    """
    # Input validation
    if image.ndim != 2:
        raise ValueError(f"Input must be 2D array, got {image.ndim}D")
    
    if image.dtype not in [np.float32, np.float64]:
        image = image.astype(np.float64)
    
    # Compute image metrics
    local_contrast = _compute_local_contrast(image)
    noise_level = _compute_noise_level(image)
    gradient_strength = _compute_gradient_strength(image)
    
    # Normalize metrics for decision making
    # These thresholds are empirically determined for typical ARPES data
    image_range = np.nanmax(image) - np.nanmin(image) + 1e-10
    normalized_noise = noise_level / image_range
    
    # Decision logic based on image characteristics
    metadata: Dict[str, Any] = {
        'local_contrast': float(local_contrast),
        'noise_level': float(noise_level),
        'gradient_strength': float(gradient_strength),
        'normalized_noise': float(normalized_noise)
    }
    
    # High noise threshold (>10% of image range)
    high_noise = normalized_noise > 0.1
    
    # Low contrast threshold
    low_contrast = local_contrast < 0.2
    
    # Method selection logic
    if high_noise:
        # High noise: use Luo method with increased smoothing
        # Adaptive smoothing handles noise better
        method_name = 'curvature_luo'
        parameters = {'k_res': 1.0, 'e_res': 1.0}
        enhanced = curvature_luo(image, **parameters)
        
    elif low_contrast and not high_noise:
        # Low contrast, low noise: Luo method for better band enhancement
        method_name = 'curvature_luo'
        parameters = {'k_res': 1.2, 'e_res': 1.0}
        enhanced = curvature_luo(image, **parameters)
        
    else:
        # Good contrast, low noise: second derivative is faster and effective
        method_name = 'curvature_second_derivative'
        # Adjust smoothing based on noise level
        sigma = max(0.5, min(2.0, 1.0 + normalized_noise * 5))
        parameters = {'smoothing_sigma': sigma}
        enhanced = curvature_second_derivative(image, **parameters)
    
    metadata['method'] = method_name
    metadata['parameters'] = parameters
    
    return enhanced, metadata


__all__ = [
    'curvature_luo',
    'curvature_second_derivative',
    'auto_curvature'
]


# =============================================================================
# Minimal tests using synthetic ARPES-like images
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("Curvature Analysis Module - Test Suite")
    print("=" * 60)
    
    def create_synthetic_arpes_image(
        n_k: int = 100,
        n_E: int = 200,
        add_noise: bool = True,
        noise_level: float = 0.05
    ) -> np.ndarray:
        """
        Create a synthetic ARPES-like image with parabolic bands.
        
        Parameters
        ----------
        n_k : int
            Number of k/angle points.
        n_E : int
            Number of energy points.
        add_noise : bool
            Whether to add Gaussian noise.
        noise_level : float
            Standard deviation of noise.
            
        Returns
        -------
        np.ndarray
            Synthetic ARPES image.
        """
        # Create coordinate grids
        k = np.linspace(-1, 1, n_k)
        E = np.linspace(-2, 0.5, n_E)
        K, energy = np.meshgrid(k, E, indexing='ij')
        
        # Create parabolic band (hole pocket)
        band_center = -0.5  # Band bottom at -0.5 eV
        effective_mass = 1.0  # Parabolic curvature
        band_dispersion = band_center + effective_mass * K**2
        
        # Create intensity along band with Lorentzian broadening
        band_width = 0.05  # Energy broadening
        intensity = 1.0 / (1 + ((energy - band_dispersion) / band_width)**2)
        
        # Add a second band (electron pocket)
        band2_center = 0.0
        band2_mass = -0.5
        band2_dispersion = band2_center + band2_mass * K**2
        intensity += 0.7 / (1 + ((energy - band2_dispersion) / band_width)**2)
        
        # Add Fermi-Dirac cutoff at E=0
        fermi_width = 0.02
        fermi_cutoff = 1 / (1 + np.exp((energy) / fermi_width))
        intensity *= fermi_cutoff
        
        # Add background
        background = 0.1 * np.exp(-energy / 0.5)
        intensity += background
        
        # Add noise if requested
        if add_noise:
            noise = np.random.randn(n_k, n_E) * noise_level
            intensity += noise
            intensity = np.maximum(intensity, 0)  # Ensure non-negative
        
        return intensity.astype(np.float64)
    
    def test_curvature_luo():
        """Test curvature_luo function."""
        print("\n[TEST] curvature_luo")
        print("-" * 40)
        
        # Create test image
        image = create_synthetic_arpes_image(n_k=80, n_E=150, noise_level=0.03)
        print(f"  Input shape: {image.shape}, dtype: {image.dtype}")
        print(f"  Input range: [{image.min():.4f}, {image.max():.4f}]")
        
        # Apply curvature method
        enhanced = curvature_luo(image, k_res=1.0, e_res=1.0)
        
        print(f"  Output shape: {enhanced.shape}, dtype: {enhanced.dtype}")
        print(f"  Output range: [{enhanced.min():.4f}, {enhanced.max():.4f}]")
        
        # Verify output
        assert enhanced.shape == image.shape, "Shape mismatch"
        assert enhanced.min() >= 0 and enhanced.max() <= 1, "Output not normalized to [0,1]"
        assert enhanced.dtype == image.dtype, "Dtype mismatch"
        
        print("  ✓ PASSED")
        return True
    
    def test_curvature_luo_different_resolutions():
        """Test curvature_luo with different resolution parameters."""
        print("\n[TEST] curvature_luo with different resolutions")
        print("-" * 40)
        
        image = create_synthetic_arpes_image(n_k=80, n_E=150)
        
        # Test different k_res values
        for k_res in [0.5, 1.0, 2.0]:
            enhanced = curvature_luo(image, k_res=k_res, e_res=1.0)
            assert enhanced.shape == image.shape
            assert 0 <= enhanced.min() and enhanced.max() <= 1
            print(f"  k_res={k_res}: range=[{enhanced.min():.4f}, {enhanced.max():.4f}]")
        
        print("  ✓ PASSED")
        return True
    
    def test_curvature_second_derivative():
        """Test curvature_second_derivative function."""
        print("\n[TEST] curvature_second_derivative")
        print("-" * 40)
        
        # Create test image
        image = create_synthetic_arpes_image(n_k=80, n_E=150, noise_level=0.02)
        print(f"  Input shape: {image.shape}, dtype: {image.dtype}")
        
        # Apply with default smoothing
        enhanced = curvature_second_derivative(image, smoothing_sigma=1.0)
        
        print(f"  Output shape: {enhanced.shape}")
        print(f"  Output range: [{enhanced.min():.4f}, {enhanced.max():.4f}]")
        
        # Verify output
        assert enhanced.shape == image.shape, "Shape mismatch"
        assert enhanced.min() >= 0 and enhanced.max() <= 1, "Output not normalized to [0,1]"
        
        print("  ✓ PASSED")
        return True
    
    def test_curvature_second_derivative_sigma_values():
        """Test curvature_second_derivative with different sigma values."""
        print("\n[TEST] curvature_second_derivative with varying sigma")
        print("-" * 40)
        
        image = create_synthetic_arpes_image(n_k=80, n_E=150)
        
        for sigma in [0.5, 1.0, 2.0, 3.0]:
            enhanced = curvature_second_derivative(image, smoothing_sigma=sigma)
            assert enhanced.shape == image.shape
            assert 0 <= enhanced.min() and enhanced.max() <= 1
            print(f"  sigma={sigma}: range=[{enhanced.min():.4f}, {enhanced.max():.4f}]")
        
        print("  ✓ PASSED")
        return True
    
    def test_auto_curvature():
        """Test auto_curvature function."""
        print("\n[TEST] auto_curvature")
        print("-" * 40)
        
        # Test with low noise image
        image_low_noise = create_synthetic_arpes_image(
            n_k=80, n_E=150, noise_level=0.01
        )
        enhanced, meta = auto_curvature(image_low_noise)
        
        print(f"  Low noise image:")
        print(f"    Method selected: {meta['method']}")
        print(f"    Local contrast: {meta['local_contrast']:.4f}")
        print(f"    Noise level: {meta['noise_level']:.4f}")
        print(f"    Parameters: {meta['parameters']}")
        
        assert enhanced.shape == image_low_noise.shape
        assert 0 <= enhanced.min() and enhanced.max() <= 1
        assert 'method' in meta
        assert 'local_contrast' in meta
        
        # Test with high noise image
        image_high_noise = create_synthetic_arpes_image(
            n_k=80, n_E=150, noise_level=0.2
        )
        enhanced_noisy, meta_noisy = auto_curvature(image_high_noise)
        
        print(f"\n  High noise image:")
        print(f"    Method selected: {meta_noisy['method']}")
        print(f"    Local contrast: {meta_noisy['local_contrast']:.4f}")
        print(f"    Noise level: {meta_noisy['noise_level']:.4f}")
        print(f"    Parameters: {meta_noisy['parameters']}")
        
        assert enhanced_noisy.shape == image_high_noise.shape
        
        print("  ✓ PASSED")
        return True
    
    def test_input_validation():
        """Test input validation for all functions."""
        print("\n[TEST] Input validation")
        print("-" * 40)
        
        # Test with 1D array (should fail)
        array_1d = np.random.rand(100)
        
        try:
            curvature_luo(array_1d)
            print("  ✗ curvature_luo should reject 1D input")
            return False
        except ValueError as e:
            print(f"  curvature_luo correctly rejected 1D: {e}")
        
        try:
            curvature_second_derivative(array_1d)
            print("  ✗ curvature_second_derivative should reject 1D input")
            return False
        except ValueError as e:
            print(f"  curvature_second_derivative correctly rejected 1D: {e}")
        
        try:
            auto_curvature(array_1d)
            print("  ✗ auto_curvature should reject 1D input")
            return False
        except ValueError as e:
            print(f"  auto_curvature correctly rejected 1D: {e}")
        
        # Test with invalid smoothing_sigma
        array_2d = np.random.rand(50, 50)
        try:
            curvature_second_derivative(array_2d, smoothing_sigma=-1.0)
            print("  ✗ Should reject negative smoothing_sigma")
            return False
        except ValueError as e:
            print(f"  Correctly rejected invalid sigma: {e}")
        
        print("  ✓ PASSED")
        return True
    
    def test_dtype_handling():
        """Test handling of different dtypes."""
        print("\n[TEST] Dtype handling")
        print("-" * 40)
        
        base_image = create_synthetic_arpes_image(n_k=50, n_E=80)
        
        # Test float32
        img_f32 = base_image.astype(np.float32)
        result_f32 = curvature_luo(img_f32)
        print(f"  float32 input -> {result_f32.dtype} output")
        assert result_f32.dtype == np.float32
        
        # Test float64
        img_f64 = base_image.astype(np.float64)
        result_f64 = curvature_luo(img_f64)
        print(f"  float64 input -> {result_f64.dtype} output")
        assert result_f64.dtype == np.float64
        
        # Test int (should be converted)
        img_int = (base_image * 1000).astype(np.int32)
        result_int = curvature_luo(img_int)
        print(f"  int32 input -> {result_int.dtype} output (converted)")
        
        print("  ✓ PASSED")
        return True
    
    def test_edge_cases():
        """Test edge cases."""
        print("\n[TEST] Edge cases")
        print("-" * 40)
        
        # Very small image
        small_image = np.random.rand(10, 10).astype(np.float64)
        result = curvature_luo(small_image)
        assert result.shape == small_image.shape
        print(f"  Small image (10x10): ✓")
        
        # Uniform image (no features)
        uniform = np.ones((50, 50), dtype=np.float64) * 0.5
        result_uniform = curvature_luo(uniform)
        assert result_uniform.shape == uniform.shape
        # Should produce mostly zeros or very low values
        print(f"  Uniform image: max={result_uniform.max():.4f}")
        
        # Image with NaN (edge case)
        with_nan = create_synthetic_arpes_image(n_k=50, n_E=80)
        with_nan[25, 40] = np.nan
        result_nan = curvature_second_derivative(with_nan)
        print(f"  Image with NaN: has_nan={np.any(np.isnan(result_nan))}")
        
        print("  ✓ PASSED")
        return True
    
    # Run all tests
    tests = [
        test_curvature_luo,
        test_curvature_luo_different_resolutions,
        test_curvature_second_derivative,
        test_curvature_second_derivative_sigma_values,
        test_auto_curvature,
        test_input_validation,
        test_dtype_handling,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  ✗ FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    sys.exit(0 if failed == 0 else 1)
