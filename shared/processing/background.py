"""
Background Removal Module for ARPES Data

This module provides standard background subtraction techniques for ARPES spectra:
- Shirley background (iterative, conserves total loss electrons)
- SNIP background (Statistics-sensitive Non-linear Iterative Peak-clipping)
- Polynomial background (robust fitting with outlier rejection)

All functions return the BACKGROUND array (same shape as input), not the
background-subtracted data. This allows users to inspect the background and
decide how to apply the subtraction.

Usage:
    import numpy as np
    from processing.background import shirley_background, snip_background
    
    # Get Shirley background for 1D EDC
    bg = shirley_background(edc, max_iter=20)
    edc_subtracted = edc - bg
    
    # Get SNIP background for 2D spectrogram
    bg_2d = snip_background(spectrum, iterations=24)
"""

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import uniform_filter1d
from typing import Literal, Optional, Union
import warnings


# =============================================================================
# Helper Functions
# =============================================================================

def _validate_spectrum(data: NDArray, min_dim: int = 1) -> None:
    """Validate input spectrum dimensions."""
    if not isinstance(data, np.ndarray):
        raise TypeError(f"Expected numpy array, got {type(data)}")
    if data.ndim < min_dim:
        raise ValueError(f"Expected at least {min_dim}D array, got {data.ndim}D")


def _lls_transform(data: NDArray) -> NDArray:
    """
    Apply Log-Log-Sqrt (LLS) transform for SNIP algorithm.
    
    LLS transform: y = log(log(sqrt(x + 1) + 1) + 1)
    This transform compresses the dynamic range while preserving peaks.
    """
    # Ensure positive values
    data_pos = np.maximum(data, 0)
    # LLS transform
    return np.log(np.log(np.sqrt(data_pos + 1) + 1) + 1)


def _inverse_lls_transform(data: NDArray) -> NDArray:
    """
    Apply inverse Log-Log-Sqrt (LLS) transform.
    
    Inverse: x = (exp(exp(y) - 1) - 1)^2 - 1
    """
    return (np.exp(np.exp(data) - 1) - 1)**2 - 1


# =============================================================================
# Shirley Background
# =============================================================================

def shirley_background(
    edc: NDArray,
    max_iter: int = 20,
    tol: float = 1e-6
) -> NDArray:
    """
    Calculate Shirley background for energy distribution curves.
    
    The Shirley background is an iterative method that models the inelastic
    scattering contribution. It conserves the total number of loss electrons
    by relating the background at each energy to the integrated intensity
    above that energy.
    
    Parameters
    ----------
    edc : NDArray
        1D energy distribution curve or 2D spectrogram.
        For 2D arrays, background is computed for each row (energy axis = 1).
    max_iter : int, optional
        Maximum number of iterations. Default is 20.
    tol : float, optional
        Convergence tolerance (relative change). Default is 1e-6.
    
    Returns
    -------
    NDArray
        Shirley background with same shape as input.
    
    Notes
    -----
    The algorithm assumes:
    - Energy axis goes from high binding energy (left) to low binding energy (right)
    - Background at endpoints is approximately correct
    
    For typical ARPES data where energy axis may be reversed, ensure correct
    endpoint selection.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create synthetic EDC with Shirley-like background
    >>> energy = np.linspace(-2, 2, 200)
    >>> peak = np.exp(-energy**2 / 0.1)
    >>> background = 0.5 * (1 - np.tanh(energy))
    >>> edc = peak + background
    >>> bg = shirley_background(edc)
    """
    _validate_spectrum(edc, min_dim=1)
    
    if edc.ndim == 1:
        return _shirley_1d(edc, max_iter, tol)
    elif edc.ndim == 2:
        # Apply to each row
        result = np.zeros_like(edc)
        for i in range(edc.shape[0]):
            result[i, :] = _shirley_1d(edc[i, :], max_iter, tol)
        return result
    else:
        raise ValueError(f"Expected 1D or 2D array, got {edc.ndim}D")


def _shirley_1d(edc: NDArray, max_iter: int, tol: float) -> NDArray:
    """Compute Shirley background for 1D EDC."""
    n = len(edc)
    
    # Use float64 for numerical stability
    edc = edc.astype(np.float64)
    
    # Get endpoint values
    # Assuming energy goes from high BE (left) to low BE (right)
    i_left = np.mean(edc[:5])   # Average of first few points
    i_right = np.mean(edc[-5:])  # Average of last few points
    
    # Initial background: linear interpolation
    background = np.linspace(i_left, i_right, n)
    
    # Iterative refinement
    for iteration in range(max_iter):
        # Signal above background
        signal = np.maximum(edc - background, 0)
        
        # Cumulative integral from right to left (loss electrons)
        cumsum_right = np.cumsum(signal[::-1])[::-1]
        total_integral = cumsum_right[0] if cumsum_right[0] > 0 else 1.0
        
        # New background: scales with cumulative loss
        new_background = i_right + (i_left - i_right) * cumsum_right / total_integral
        
        # Check convergence
        change = np.max(np.abs(new_background - background))
        relative_change = change / (np.max(np.abs(background)) + 1e-10)
        
        background = new_background
        
        if relative_change < tol:
            break
    
    return background


# =============================================================================
# SNIP Background
# =============================================================================

def snip_background(
    spectrum: NDArray,
    iterations: int = 24,
    decreasing: bool = True
) -> NDArray:
    """
    Apply SNIP algorithm for background estimation.
    
    SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) is a 
    robust algorithm that iteratively clips peaks to estimate the underlying
    background. Uses LLS transform for better peak preservation.
    
    Parameters
    ----------
    spectrum : NDArray
        1D spectrum or 2D spectrogram. For 2D, SNIP is applied independently
        along the last axis (energy axis).
    iterations : int, optional
        Number of clipping iterations. Higher values produce smoother
        backgrounds but may clip broad features. Default is 24.
    decreasing : bool, optional
        If True, use decreasing window sizes (recommended). Default is True.
    
    Returns
    -------
    NDArray
        Background estimate with same shape as input.
    
    Notes
    -----
    SNIP works best for spectra with well-defined peaks on a slowly varying
    background. The number of iterations controls the width of features that
    are preserved vs clipped.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create spectrum with peaks on polynomial background
    >>> x = np.linspace(0, 10, 500)
    >>> background = 100 + 10 * x - 0.5 * x**2
    >>> peaks = 50 * np.exp(-((x - 3) / 0.3)**2) + 30 * np.exp(-((x - 7) / 0.5)**2)
    >>> spectrum = background + peaks + 5 * np.random.randn(500)
    >>> bg = snip_background(spectrum, iterations=30)
    """
    _validate_spectrum(spectrum, min_dim=1)
    
    if spectrum.ndim == 1:
        return _snip_1d(spectrum, iterations, decreasing)
    elif spectrum.ndim == 2:
        # Apply to each row (along energy axis)
        result = np.zeros_like(spectrum)
        for i in range(spectrum.shape[0]):
            result[i, :] = _snip_1d(spectrum[i, :], iterations, decreasing)
        return result
    elif spectrum.ndim == 3:
        # For 3D, apply to each 2D slice
        result = np.zeros_like(spectrum)
        for i in range(spectrum.shape[2]):
            result[:, :, i] = snip_background(spectrum[:, :, i], iterations, decreasing)
        return result
    else:
        raise ValueError(f"Expected 1D, 2D, or 3D array, got {spectrum.ndim}D")


def _snip_1d(spectrum: NDArray, iterations: int, decreasing: bool) -> NDArray:
    """Apply SNIP algorithm to 1D spectrum."""
    n = len(spectrum)
    
    # Use float64 for numerical stability
    spectrum = spectrum.astype(np.float64)
    
    # Apply LLS transform
    y = _lls_transform(spectrum)
    
    # Working copy
    working = y.copy()
    
    if decreasing:
        # Decreasing window sizes (recommended)
        window_sizes = range(iterations, 0, -1)
    else:
        # Constant small window
        window_sizes = [1] * iterations
    
    for p in window_sizes:
        # For each point, take minimum of current value and average of neighbors
        for i in range(p, n - p):
            neighbor_avg = 0.5 * (working[i - p] + working[i + p])
            working[i] = min(working[i], neighbor_avg)
    
    # Inverse LLS transform
    background = _inverse_lls_transform(working)
    
    # Ensure non-negative
    background = np.maximum(background, 0)
    
    return background


# =============================================================================
# Polynomial Background
# =============================================================================

def poly_background(
    data: NDArray,
    order: int = 3,
    axis: Literal['energy', 'kx', 'ky', 'hv'] = 'energy',
    robust: bool = True,
    outlier_threshold: float = 2.5
) -> NDArray:
    """
    Fit polynomial background with optional outlier rejection.
    
    Uses robust regression (iterative reweighted least squares with Huber
    loss) to ignore peaks when fitting the background.
    
    Parameters
    ----------
    data : NDArray
        1D, 2D, or 3D array. Background is computed along specified axis.
    order : int, optional
        Polynomial order. Default is 3.
    axis : {'energy', 'kx', 'ky', 'hv'}, optional
        Axis along which to fit polynomial. This affects interpretation
        for multi-dimensional data. Default is 'energy'.
        - 'energy': axis 0 for 1D, axis 1 for 2D, axis 0 for 3D
        - 'kx'/'ky': spatial axes
        - 'hv': photon energy for 3D data
    robust : bool, optional
        If True, use robust regression with outlier rejection. Default is True.
    outlier_threshold : float, optional
        Threshold (in standard deviations) for outlier rejection. Default is 2.5.
    
    Returns
    -------
    NDArray
        Polynomial background with same shape as input.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Create EDC with polynomial background
    >>> energy = np.linspace(-5, 1, 300)
    >>> background = 100 + 20 * energy + 5 * energy**2
    >>> peak = 200 * np.exp(-((energy + 0.5) / 0.2)**2)
    >>> edc = background + peak
    >>> bg = poly_background(edc, order=2)
    """
    _validate_spectrum(data, min_dim=1)
    
    # Determine which axis to use based on dimensionality and axis parameter
    if data.ndim == 1:
        return _poly_background_1d(data, order, robust, outlier_threshold)
    elif data.ndim == 2:
        # For 2D: axis 1 is typically energy
        fit_axis = 1 if axis == 'energy' else 0
        return _poly_background_2d(data, order, fit_axis, robust, outlier_threshold)
    elif data.ndim == 3:
        # For 3D: determine axis mapping
        axis_map = {'energy': 0, 'kx': 1, 'ky': 2, 'hv': 2}
        fit_axis = axis_map.get(axis, 0)
        return _poly_background_3d(data, order, fit_axis, robust, outlier_threshold)
    else:
        raise ValueError(f"Expected 1D, 2D, or 3D array, got {data.ndim}D")


def _poly_background_1d(
    data: NDArray,
    order: int,
    robust: bool,
    outlier_threshold: float
) -> NDArray:
    """Fit polynomial background to 1D data."""
    n = len(data)
    x = np.arange(n, dtype=np.float64)
    y = data.astype(np.float64)
    
    if robust:
        # Iterative reweighted least squares
        weights = np.ones(n)
        
        for _ in range(5):  # Usually converges in 3-4 iterations
            # Weighted polynomial fit
            coeffs = np.polyfit(x, y, order, w=weights)
            background = np.polyval(coeffs, x)
            
            # Compute residuals
            residuals = y - background
            mad = np.median(np.abs(residuals - np.median(residuals)))
            std_robust = 1.4826 * mad  # MAD to std conversion
            
            # Update weights using Huber-like function
            if std_robust > 1e-10:
                normalized_resid = np.abs(residuals) / std_robust
                # Downweight outliers
                weights = np.where(
                    normalized_resid < outlier_threshold,
                    1.0,
                    outlier_threshold / (normalized_resid + 1e-10)
                )
    else:
        # Simple polynomial fit
        coeffs = np.polyfit(x, y, order)
        background = np.polyval(coeffs, x)
    
    return background


def _poly_background_2d(
    data: NDArray,
    order: int,
    fit_axis: int,
    robust: bool,
    outlier_threshold: float
) -> NDArray:
    """Fit polynomial background to 2D data along specified axis."""
    result = np.zeros_like(data, dtype=np.float64)
    
    if fit_axis == 0:
        # Fit along first axis (rows)
        for j in range(data.shape[1]):
            result[:, j] = _poly_background_1d(
                data[:, j], order, robust, outlier_threshold
            )
    else:
        # Fit along second axis (columns)
        for i in range(data.shape[0]):
            result[i, :] = _poly_background_1d(
                data[i, :], order, robust, outlier_threshold
            )
    
    return result


def _poly_background_3d(
    data: NDArray,
    order: int,
    fit_axis: int,
    robust: bool,
    outlier_threshold: float
) -> NDArray:
    """Fit polynomial background to 3D data along specified axis."""
    result = np.zeros_like(data, dtype=np.float64)
    
    if fit_axis == 0:
        # Fit along first axis
        for j in range(data.shape[1]):
            for k in range(data.shape[2]):
                result[:, j, k] = _poly_background_1d(
                    data[:, j, k], order, robust, outlier_threshold
                )
    elif fit_axis == 1:
        # Fit along second axis
        for i in range(data.shape[0]):
            for k in range(data.shape[2]):
                result[i, :, k] = _poly_background_1d(
                    data[i, :, k], order, robust, outlier_threshold
                )
    else:
        # Fit along third axis
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                result[i, j, :] = _poly_background_1d(
                    data[i, j, :], order, robust, outlier_threshold
                )
    
    return result


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    'shirley_background',
    'snip_background',
    'poly_background'
]


# =============================================================================
# Test Section
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Background Removal Module - Quick Tests")
    print("=" * 60)
    
    np.random.seed(42)
    
    # ---------------------------------------------------------------------
    # Test 1: Shirley Background
    # ---------------------------------------------------------------------
    print("\n1. Testing Shirley Background...")
    
    # Create synthetic EDC with Fermi edge + Shirley-like background
    n_points = 200
    energy = np.linspace(-2, 1, n_points)
    
    # Fermi edge (step function convolved with Gaussian)
    from scipy.special import erfc
    fermi_edge = 0.5 * erfc((energy - 0) / 0.05)
    
    # Shirley-like background (increases below Fermi level)
    true_bg_shirley = 0.3 * np.cumsum(fermi_edge[::-1])[::-1]
    true_bg_shirley = true_bg_shirley / true_bg_shirley.max() * 0.5
    
    edc = fermi_edge + true_bg_shirley + 0.02 * np.random.randn(n_points)
    
    bg_shirley = shirley_background(edc, max_iter=30)
    print(f"   - Input shape: {edc.shape}")
    print(f"   - Background shape: {bg_shirley.shape}")
    print(f"   - Background range: [{bg_shirley.min():.3f}, {bg_shirley.max():.3f}]")
    
    # ---------------------------------------------------------------------
    # Test 2: SNIP Background
    # ---------------------------------------------------------------------
    print("\n2. Testing SNIP Background...")
    
    # Create spectrum with multiple peaks on polynomial background
    x = np.linspace(0, 10, 500)
    true_bg_snip = 50 + 10 * x - x**2 + 0.1 * x**3
    peaks = (80 * np.exp(-((x - 2) / 0.3)**2) + 
             60 * np.exp(-((x - 5) / 0.5)**2) +
             40 * np.exp(-((x - 8) / 0.4)**2))
    spectrum = true_bg_snip + peaks + 3 * np.random.randn(500)
    spectrum = np.maximum(spectrum, 0)
    
    bg_snip = snip_background(spectrum, iterations=24)
    print(f"   - Input shape: {spectrum.shape}")
    print(f"   - Background shape: {bg_snip.shape}")
    print(f"   - RMS error vs true: {np.sqrt(np.mean((bg_snip - true_bg_snip)**2)):.2f}")
    
    # Test 2D SNIP
    spectrum_2d = np.vstack([spectrum * (1 + 0.1 * i) for i in range(10)])
    bg_snip_2d = snip_background(spectrum_2d, iterations=24)
    print(f"   - 2D input shape: {spectrum_2d.shape}")
    print(f"   - 2D background shape: {bg_snip_2d.shape}")
    
    # ---------------------------------------------------------------------
    # Test 3: Polynomial Background
    # ---------------------------------------------------------------------
    print("\n3. Testing Polynomial Background...")
    
    # Create EDC with known polynomial background
    energy_poly = np.linspace(-5, 1, 300)
    true_bg_poly = 100 + 20 * energy_poly + 5 * energy_poly**2
    peak = 200 * np.exp(-((energy_poly + 0.5) / 0.2)**2)
    edc_poly = true_bg_poly + peak + 5 * np.random.randn(300)
    
    bg_poly = poly_background(edc_poly, order=2, robust=True)
    bg_poly_simple = poly_background(edc_poly, order=2, robust=False)
    
    print(f"   - Input shape: {edc_poly.shape}")
    print(f"   - Robust poly RMS error: {np.sqrt(np.mean((bg_poly - true_bg_poly)**2)):.2f}")
    print(f"   - Simple poly RMS error: {np.sqrt(np.mean((bg_poly_simple - true_bg_poly)**2)):.2f}")
    
    # Test 2D polynomial
    edc_2d = np.vstack([edc_poly * (1 + 0.05 * i) for i in range(8)])
    bg_poly_2d = poly_background(edc_2d, order=2, axis='energy')
    print(f"   - 2D input shape: {edc_2d.shape}")
    print(f"   - 2D background shape: {bg_poly_2d.shape}")
    
    # ---------------------------------------------------------------------
    # Visualization
    # ---------------------------------------------------------------------
    print("\n4. Attempting visualization...")
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        
        # Shirley
        axes[0, 0].plot(energy, edc, 'b-', label='EDC', alpha=0.7)
        axes[0, 0].plot(energy, bg_shirley, 'r--', label='Shirley BG', lw=2)
        axes[0, 0].plot(energy, true_bg_shirley, 'g:', label='True BG', lw=2)
        axes[0, 0].set_xlabel('Energy (eV)')
        axes[0, 0].set_ylabel('Intensity')
        axes[0, 0].set_title('Shirley Background')
        axes[0, 0].legend()
        
        axes[1, 0].plot(energy, edc - bg_shirley, 'b-', label='Subtracted')
        axes[1, 0].plot(energy, fermi_edge, 'g--', label='True signal', alpha=0.7)
        axes[1, 0].set_xlabel('Energy (eV)')
        axes[1, 0].set_ylabel('Intensity')
        axes[1, 0].set_title('After Shirley Subtraction')
        axes[1, 0].legend()
        
        # SNIP
        axes[0, 1].plot(x, spectrum, 'b-', label='Spectrum', alpha=0.7)
        axes[0, 1].plot(x, bg_snip, 'r--', label='SNIP BG', lw=2)
        axes[0, 1].plot(x, true_bg_snip, 'g:', label='True BG', lw=2)
        axes[0, 1].set_xlabel('Channel')
        axes[0, 1].set_ylabel('Intensity')
        axes[0, 1].set_title('SNIP Background')
        axes[0, 1].legend()
        
        axes[1, 1].plot(x, spectrum - bg_snip, 'b-', label='Subtracted')
        axes[1, 1].plot(x, peaks, 'g--', label='True peaks', alpha=0.7)
        axes[1, 1].set_xlabel('Channel')
        axes[1, 1].set_ylabel('Intensity')
        axes[1, 1].set_title('After SNIP Subtraction')
        axes[1, 1].legend()
        
        # Polynomial
        axes[0, 2].plot(energy_poly, edc_poly, 'b-', label='EDC', alpha=0.7)
        axes[0, 2].plot(energy_poly, bg_poly, 'r--', label='Robust poly', lw=2)
        axes[0, 2].plot(energy_poly, true_bg_poly, 'g:', label='True BG', lw=2)
        axes[0, 2].set_xlabel('Energy (eV)')
        axes[0, 2].set_ylabel('Intensity')
        axes[0, 2].set_title('Polynomial Background')
        axes[0, 2].legend()
        
        axes[1, 2].plot(energy_poly, edc_poly - bg_poly, 'b-', label='Subtracted')
        axes[1, 2].plot(energy_poly, peak, 'g--', label='True peak', alpha=0.7)
        axes[1, 2].set_xlabel('Energy (eV)')
        axes[1, 2].set_ylabel('Intensity')
        axes[1, 2].set_title('After Polynomial Subtraction')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig('background_test_result.png', dpi=150)
        print("   - Saved visualization to 'background_test_result.png'")
        plt.show()
        
    except ImportError:
        print("   - matplotlib not available, skipping visualization")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
