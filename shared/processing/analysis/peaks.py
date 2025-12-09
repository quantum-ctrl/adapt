"""
Automatic EDC/MDC Peak Extraction Module

This module provides functions for automatic peak detection, fitting, and band
dispersion extraction from 2D ARPES spectral maps.

Features:
---------
- 1D peak detection with prominence filtering
- Lorentzian peak fitting with uncertainty estimation
- MDC (Momentum Distribution Curve) peak extraction
- EDC (Energy Distribution Curve) peak extraction
- Automatic band dispersion tracking across 2D maps

Usage:
------
    from processing.analysis.peaks import (
        detect_peaks_1d,
        fit_peak_lorentzian,
        extract_mdc_peaks,
        extract_edc_peaks,
        extract_band_dispersion
    )
    
    # Detect peaks in a 1D signal
    indices, amplitudes = detect_peaks_1d(signal)
    
    # Extract band dispersion from 2D ARPES map
    dispersions = extract_band_dispersion(image, axis="MDC")
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from typing import Tuple, List, Dict, Optional, Union, Any


# ==============================================================================
# Constants
# ==============================================================================

MIN_FWHM = 1e-6  # Minimum FWHM to avoid numerical issues
MAX_FIT_ITER = 5000  # Maximum iterations for curve fitting


# ==============================================================================
# Peak Detection
# ==============================================================================

def detect_peaks_1d(
    signal: np.ndarray,
    threshold: float = 0.05,
    min_distance: int = 3,
    smooth_sigma: float = 0.0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect peak indices in a 1D array.
    
    The signal is normalized, smoothed (optionally), and local maxima are 
    identified using prominence-based filtering.
    
    Parameters
    ----------
    signal : np.ndarray
        1D input signal array.
    threshold : float, optional
        Minimum prominence threshold (relative to normalized signal range).
        Peaks with prominence below this value are filtered out.
        Default: 0.05 (5% of signal range).
    min_distance : int, optional
        Minimum distance (in samples) between neighboring peaks.
        Default: 3.
    smooth_sigma : float, optional
        Gaussian smoothing sigma before peak detection. 0 = no smoothing.
        Default: 0.0.
    
    Returns
    -------
    indices : np.ndarray
        Array of peak indices in the original signal.
    amplitudes : np.ndarray
        Array of peak amplitudes (normalized values at peak positions).
    
    Examples
    --------
    >>> signal = np.array([0, 1, 3, 2, 1, 0, 2, 5, 3, 1])
    >>> indices, amplitudes = detect_peaks_1d(signal, threshold=0.1)
    >>> print(indices)
    [2, 7]
    """
    # Input validation
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {signal.shape}")
    
    if len(signal) < 3:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    # Handle NaN values
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Normalize signal to [0, 1]
    sig_min = signal.min()
    sig_max = signal.max()
    sig_range = sig_max - sig_min
    
    if sig_range < 1e-10:
        # Flat signal, no peaks
        return np.array([], dtype=int), np.array([], dtype=float)
    
    signal_norm = (signal - sig_min) / sig_range
    
    # Optional Gaussian smoothing
    if smooth_sigma > 0:
        signal_smooth = gaussian_filter1d(signal_norm, sigma=smooth_sigma)
    else:
        signal_smooth = signal_norm
    
    # Find local maxima with minimum distance constraint
    peaks, properties = find_peaks(
        signal_smooth,
        distance=min_distance,
        prominence=threshold
    )
    
    if len(peaks) == 0:
        return np.array([], dtype=int), np.array([], dtype=float)
    
    # Get amplitudes from normalized signal (not smoothed)
    amplitudes = signal_norm[peaks]
    
    return peaks, amplitudes


# ==============================================================================
# Peak Fitting
# ==============================================================================

def _lorentzian(x: np.ndarray, x0: float, gamma: float, A: float, bg: float) -> np.ndarray:
    """
    Lorentzian function with background.
    
    L(x) = A * (gamma/2)^2 / ((x - x0)^2 + (gamma/2)^2) + bg
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable.
    x0 : float
        Peak center position.
    gamma : float
        Full Width at Half Maximum (FWHM).
    A : float
        Peak amplitude (height above background).
    bg : float
        Constant background level.
    
    Returns
    -------
    np.ndarray
        Lorentzian profile values.
    """
    half_gamma = gamma / 2
    return A * (half_gamma ** 2) / ((x - x0) ** 2 + half_gamma ** 2) + bg


def fit_peak_lorentzian(
    x: np.ndarray,
    y: np.ndarray,
    x0_guess: Optional[float] = None,
    bounds: Optional[Tuple[List[float], List[float]]] = None
) -> Dict[str, Any]:
    """
    Fit a single peak to a Lorentzian model.
    
    Parameters
    ----------
    x : np.ndarray
        1D array of x-coordinates (e.g., momentum or energy).
    y : np.ndarray
        1D array of intensity values.
    x0_guess : float, optional
        Initial guess for peak position. If None, uses the x value at max y.
    bounds : tuple, optional
        Bounds for fitting parameters ((lower,), (upper,)).
        Order: [x0, gamma, A, bg].
        If None, uses automatic bounds based on data.
    
    Returns
    -------
    dict
        Fitting results containing:
        - 'position': Peak center position
        - 'position_err': Uncertainty in position
        - 'fwhm': Full Width at Half Maximum
        - 'fwhm_err': Uncertainty in FWHM
        - 'amplitude': Peak amplitude
        - 'amplitude_err': Uncertainty in amplitude
        - 'background': Background level
        - 'background_err': Uncertainty in background
        - 'fit_quality': R-squared quality score (0-1)
        - 'fit_curve': Fitted curve values
        - 'success': Whether fit converged
        - 'message': Status or error message
    
    Examples
    --------
    >>> x = np.linspace(-5, 5, 101)
    >>> y = 2.0 / ((x - 0.5)**2 + 1) + 0.5 + 0.1 * np.random.randn(len(x))
    >>> result = fit_peak_lorentzian(x, y)
    >>> print(f"Peak position: {result['position']:.3f}")
    """
    # Input validation
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1D arrays of the same shape")
    
    n_points = len(x)
    if n_points < 4:
        return _failed_fit_result(x, "Not enough data points (need at least 4)")
    
    # Handle NaN values
    valid_mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid_mask) < 4:
        return _failed_fit_result(x, "Not enough valid data points after removing NaN")
    
    x_valid = x[valid_mask]
    y_valid = y[valid_mask]
    
    # Initial parameter guesses
    y_min = y_valid.min()
    y_max = y_valid.max()
    bg_guess = y_min
    A_guess = y_max - y_min
    
    if x0_guess is None:
        x0_guess = x_valid[np.argmax(y_valid)]
    
    # Estimate FWHM from half-maximum points
    half_max = (y_max + y_min) / 2
    above_half = y_valid > half_max
    if np.any(above_half):
        x_above = x_valid[above_half]
        gamma_guess = max(x_above.max() - x_above.min(), (x.max() - x.min()) / 10)
    else:
        gamma_guess = (x.max() - x.min()) / 5
    
    gamma_guess = max(gamma_guess, MIN_FWHM)
    
    # Set up bounds
    x_range = x_valid.max() - x_valid.min()
    if bounds is None:
        bounds = (
            [x_valid.min() - x_range * 0.1, MIN_FWHM, 0, y_min - abs(y_min) * 0.5],
            [x_valid.max() + x_range * 0.1, x_range * 2, A_guess * 10, y_max]
        )
    
    p0 = [x0_guess, gamma_guess, A_guess, bg_guess]
    
    # Perform fit
    try:
        popt, pcov = curve_fit(
            _lorentzian,
            x_valid,
            y_valid,
            p0=p0,
            bounds=bounds,
            maxfev=MAX_FIT_ITER
        )
        
        # Extract uncertainties from covariance matrix
        perr = np.sqrt(np.diag(pcov))
        
        # Calculate fit quality (R-squared)
        y_fit = _lorentzian(x_valid, *popt)
        ss_res = np.sum((y_valid - y_fit) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        r_squared = max(0.0, min(1.0, r_squared))  # Clamp to [0, 1]
        
        # Generate full fit curve
        fit_curve = _lorentzian(x, *popt)
        
        return {
            'position': popt[0],
            'position_err': perr[0],
            'fwhm': popt[1],
            'fwhm_err': perr[1],
            'amplitude': popt[2],
            'amplitude_err': perr[2],
            'background': popt[3],
            'background_err': perr[3],
            'fit_quality': r_squared,
            'fit_curve': fit_curve,
            'success': True,
            'message': 'Fit converged successfully'
        }
        
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        return _failed_fit_result(x, f"Fitting failed: {str(e)}")


def _failed_fit_result(x: np.ndarray, message: str) -> Dict[str, Any]:
    """Create a standardized failed fit result dictionary."""
    return {
        'position': np.nan,
        'position_err': np.nan,
        'fwhm': np.nan,
        'fwhm_err': np.nan,
        'amplitude': np.nan,
        'amplitude_err': np.nan,
        'background': np.nan,
        'background_err': np.nan,
        'fit_quality': 0.0,
        'fit_curve': np.full_like(x, np.nan),
        'success': False,
        'message': message
    }


# ==============================================================================
# MDC/EDC Peak Extraction
# ==============================================================================

def extract_mdc_peaks(
    image: np.ndarray,
    energy_index: int,
    k_coords: Optional[np.ndarray] = None,
    threshold: float = 0.1,
    min_distance: int = 3,
    fit_peaks: bool = True,
    fit_window: int = 10,
    k_range: Optional[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """
    Extract peaks from a Momentum Distribution Curve (MDC) at a given energy.
    
    An MDC is a horizontal slice through an ARPES image at fixed energy,
    showing intensity vs momentum.
    
    Parameters
    ----------
    image : np.ndarray
        2D ARPES image with shape (n_energy, n_momentum).
        First axis is energy, second axis is momentum.
    energy_index : int
        Index along the energy axis to extract the MDC.
    k_coords : np.ndarray, optional
        1D array of momentum coordinates. If None, uses pixel indices.
    threshold : float, optional
        Peak detection prominence threshold. Default: 0.1.
    min_distance : int, optional
        Minimum distance between peaks (in pixels). Default: 3.
    fit_peaks : bool, optional
        Whether to fit detected peaks with Lorentzian. Default: True.
    fit_window : int, optional
        Half-width of fitting window around each peak (in pixels). Default: 10.
    k_range : tuple, optional
        (k_min, k_max) valid momentum range. Peaks outside are filtered.
        Default: None (use full range).
    
    Returns
    -------
    list of dict
        List of peak information dictionaries, each containing:
        - 'k_index': Peak index in momentum array
        - 'k': Peak momentum coordinate (if k_coords provided)
        - 'intensity': Peak intensity from raw data
        - 'fit_result': Lorentzian fit results (if fit_peaks=True)
    
    Examples
    --------
    >>> image = np.random.randn(100, 200)  # (energy, momentum)
    >>> peaks = extract_mdc_peaks(image, energy_index=50)
    >>> for p in peaks:
    ...     print(f"Peak at k={p['k']:.3f}")
    """
    # Input validation
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    n_energy, n_momentum = image.shape
    
    if not 0 <= energy_index < n_energy:
        raise IndexError(f"energy_index {energy_index} out of range [0, {n_energy})")
    
    # Generate momentum coordinates if not provided
    if k_coords is None:
        k_coords = np.arange(n_momentum, dtype=float)
    else:
        k_coords = np.asarray(k_coords, dtype=float)
        if len(k_coords) != n_momentum:
            raise ValueError(f"k_coords length {len(k_coords)} != image width {n_momentum}")
    
    # Extract MDC
    mdc = image[energy_index, :]
    
    # Detect peaks
    peak_indices, peak_amplitudes = detect_peaks_1d(mdc, threshold=threshold, min_distance=min_distance)
    
    if len(peak_indices) == 0:
        return []
    
    # Build peak list
    peaks = []
    for i, (idx, amp) in enumerate(zip(peak_indices, peak_amplitudes)):
        k_val = k_coords[idx]
        
        # Filter by k_range if specified
        if k_range is not None:
            k_min, k_max = k_range
            if k_val < k_min or k_val > k_max:
                continue
        
        peak_info = {
            'k_index': int(idx),
            'k': float(k_val),
            'intensity': float(mdc[idx]),
            'normalized_amplitude': float(amp)
        }
        
        # Fit peak if requested
        if fit_peaks:
            # Define fitting window
            win_start = max(0, idx - fit_window)
            win_end = min(n_momentum, idx + fit_window + 1)
            
            x_fit = k_coords[win_start:win_end]
            y_fit = mdc[win_start:win_end]
            
            if len(x_fit) >= 4:
                fit_result = fit_peak_lorentzian(x_fit, y_fit, x0_guess=k_val)
                peak_info['fit_result'] = fit_result
                
                # Update peak position with fitted value if successful
                if fit_result['success']:
                    peak_info['k_fitted'] = fit_result['position']
                    peak_info['k_err'] = fit_result['position_err']
                    peak_info['fwhm'] = fit_result['fwhm']
                    peak_info['fwhm_err'] = fit_result['fwhm_err']
        
        peaks.append(peak_info)
    
    return peaks


def extract_edc_peaks(
    image: np.ndarray,
    k_index: int,
    energy_coords: Optional[np.ndarray] = None,
    threshold: float = 0.05,
    min_distance: int = 3,
    fit_peaks: bool = True,
    fit_window: int = 10,
    energy_range: Optional[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """
    Extract peaks from an Energy Distribution Curve (EDC) at a given momentum.
    
    An EDC is a vertical slice through an ARPES image at fixed momentum,
    showing intensity vs energy.
    
    Parameters
    ----------
    image : np.ndarray
        2D ARPES image with shape (n_energy, n_momentum).
        First axis is energy, second axis is momentum.
    k_index : int
        Index along the momentum axis to extract the EDC.
    energy_coords : np.ndarray, optional
        1D array of energy coordinates. If None, uses pixel indices.
    threshold : float, optional
        Peak detection prominence threshold. Default: 0.05.
    min_distance : int, optional
        Minimum distance between peaks (in pixels). Default: 3.
    fit_peaks : bool, optional
        Whether to fit detected peaks with Lorentzian. Default: True.
    fit_window : int, optional
        Half-width of fitting window around each peak (in pixels). Default: 10.
    energy_range : tuple, optional
        (E_min, E_max) valid energy range. Peaks outside are filtered.
        Default: None (use full range).
    
    Returns
    -------
    list of dict
        List of peak information dictionaries, each containing:
        - 'energy_index': Peak index in energy array
        - 'energy': Peak energy coordinate (if energy_coords provided)
        - 'intensity': Peak intensity from raw data
        - 'fit_result': Lorentzian fit results (if fit_peaks=True)
    
    Examples
    --------
    >>> image = np.random.randn(100, 200)  # (energy, momentum)
    >>> peaks = extract_edc_peaks(image, k_index=100)
    >>> for p in peaks:
    ...     print(f"Peak at E={p['energy']:.3f}")
    """
    # Input validation
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    n_energy, n_momentum = image.shape
    
    if not 0 <= k_index < n_momentum:
        raise IndexError(f"k_index {k_index} out of range [0, {n_momentum})")
    
    # Generate energy coordinates if not provided
    if energy_coords is None:
        energy_coords = np.arange(n_energy, dtype=float)
    else:
        energy_coords = np.asarray(energy_coords, dtype=float)
        if len(energy_coords) != n_energy:
            raise ValueError(f"energy_coords length {len(energy_coords)} != image height {n_energy}")
    
    # Extract EDC
    edc = image[:, k_index]
    
    # Detect peaks
    peak_indices, peak_amplitudes = detect_peaks_1d(edc, threshold=threshold, min_distance=min_distance)
    
    if len(peak_indices) == 0:
        return []
    
    # Build peak list
    peaks = []
    for i, (idx, amp) in enumerate(zip(peak_indices, peak_amplitudes)):
        e_val = energy_coords[idx]
        
        # Filter by energy_range if specified
        if energy_range is not None:
            e_min, e_max = energy_range
            if e_val < e_min or e_val > e_max:
                continue
        
        peak_info = {
            'energy_index': int(idx),
            'energy': float(e_val),
            'intensity': float(edc[idx]),
            'normalized_amplitude': float(amp)
        }
        
        # Fit peak if requested
        if fit_peaks:
            # Define fitting window
            win_start = max(0, idx - fit_window)
            win_end = min(n_energy, idx + fit_window + 1)
            
            x_fit = energy_coords[win_start:win_end]
            y_fit = edc[win_start:win_end]
            
            if len(x_fit) >= 4:
                fit_result = fit_peak_lorentzian(x_fit, y_fit, x0_guess=e_val)
                peak_info['fit_result'] = fit_result
                
                # Update peak position with fitted value if successful
                if fit_result['success']:
                    peak_info['energy_fitted'] = fit_result['position']
                    peak_info['energy_err'] = fit_result['position_err']
                    peak_info['fwhm'] = fit_result['fwhm']
                    peak_info['fwhm_err'] = fit_result['fwhm_err']
        
        peaks.append(peak_info)
    
    return peaks


# ==============================================================================
# Band Dispersion Extraction
# ==============================================================================

def extract_band_dispersion(
    image: np.ndarray,
    axis: str = "MDC",
    energy_coords: Optional[np.ndarray] = None,
    k_coords: Optional[np.ndarray] = None,
    threshold: float = 0.05,
    min_distance: int = 3,
    fit_peaks: bool = True,
    fit_window: int = 10,
    continuity_threshold: float = 5.0,
    intensity_weight: float = 0.3,
    min_track_length: int = 5,
    k_range: Optional[Tuple[float, float]] = None,
    energy_range: Optional[Tuple[float, float]] = None
) -> List[Dict[str, Any]]:
    """
    Automatically extract band dispersions from a 2D ARPES map.
    
    This function detects peaks across all slices (MDC or EDC) and connects
    them into continuous band tracks using spatial continuity and intensity
    weighting.
    
    Parameters
    ----------
    image : np.ndarray
        2D ARPES image with shape (n_energy, n_momentum).
    axis : str, optional
        Direction for peak extraction. 
        "MDC": Extract peaks along momentum at each energy (default).
        "EDC": Extract peaks along energy at each momentum.
    energy_coords : np.ndarray, optional
        1D array of energy coordinates.
    k_coords : np.ndarray, optional
        1D array of momentum coordinates.
    threshold : float, optional
        Peak detection prominence threshold. Default: 0.05.
    min_distance : int, optional
        Minimum peak separation in pixels. Default: 3.
    fit_peaks : bool, optional
        Whether to fit peaks with Lorentzian. Default: True.
    fit_window : int, optional
        Fitting window half-width in pixels. Default: 10.
    continuity_threshold : float, optional
        Maximum allowed distance between consecutive peaks in a track
        (in pixels). Default: 5.0.
    intensity_weight : float, optional
        Weight for intensity matching in track connection (0-1).
        Higher values prefer connecting peaks with similar intensities.
        Default: 0.3.
    min_track_length : int, optional
        Minimum number of points required to form a valid band track.
        Default: 5.
    k_range : tuple, optional
        Valid momentum range (k_min, k_max). Peaks outside are ignored.
    energy_range : tuple, optional
        Valid energy range (E_min, E_max). Peaks outside are ignored.
    
    Returns
    -------
    list of dict
        List of band dispersion tracks, each containing:
        - 'band_id': Unique identifier for the band
        - 'n_points': Number of points in the track
        - 'positions': Array of (energy, k) positions
        - 'positions_err': Array of position uncertainties
        - 'intensities': Array of peak intensities
        - 'fwhm': Array of FWHM values (if fit_peaks=True)
        - 'fwhm_err': Array of FWHM uncertainties
        - 'fit_quality': Array of fit quality scores
        - 'axis': Extraction axis used ("MDC" or "EDC")
    
    Notes
    -----
    The algorithm works in two phases:
    
    1. Peak extraction: Detect and fit peaks in each MDC or EDC slice.
    
    2. Track building: Connect peaks into continuous tracks using a 
       nearest-neighbor approach with intensity weighting. The cost
       function balances spatial proximity and intensity matching:
       
       cost = distance + intensity_weight * |I1 - I2| / I_range
    
    Examples
    --------
    >>> # Synthetic ARPES-like data with parabolic band
    >>> E = np.linspace(-2, 0.5, 100)
    >>> k = np.linspace(-1, 1, 200)
    >>> KK, EE = np.meshgrid(k, E)
    >>> image = np.exp(-((EE - (0.5 * KK**2 - 1.5))**2) / 0.1)
    >>> 
    >>> dispersions = extract_band_dispersion(image, axis="MDC")
    >>> print(f"Found {len(dispersions)} bands")
    """
    # Input validation
    image = np.asarray(image, dtype=float)
    if image.ndim != 2:
        raise ValueError(f"Expected 2D image, got shape {image.shape}")
    
    n_energy, n_momentum = image.shape
    axis = axis.upper()
    
    if axis not in ("MDC", "EDC"):
        raise ValueError(f"axis must be 'MDC' or 'EDC', got '{axis}'")
    
    # Generate coordinates if not provided
    if energy_coords is None:
        energy_coords = np.arange(n_energy, dtype=float)
    else:
        energy_coords = np.asarray(energy_coords, dtype=float)
    
    if k_coords is None:
        k_coords = np.arange(n_momentum, dtype=float)
    else:
        k_coords = np.asarray(k_coords, dtype=float)
    
    # Phase 1: Extract peaks from all slices
    all_peaks = []
    
    if axis == "MDC":
        # Iterate over energy indices
        for e_idx in range(n_energy):
            e_val = energy_coords[e_idx]
            
            # Skip if outside energy range
            if energy_range is not None:
                if e_val < energy_range[0] or e_val > energy_range[1]:
                    continue
            
            peaks = extract_mdc_peaks(
                image, e_idx,
                k_coords=k_coords,
                threshold=threshold,
                min_distance=min_distance,
                fit_peaks=fit_peaks,
                fit_window=fit_window,
                k_range=k_range
            )
            
            for p in peaks:
                # Use fitted position if available
                if 'k_fitted' in p:
                    k_pos = p['k_fitted']
                    k_err = p['k_err']
                else:
                    k_pos = p['k']
                    k_err = 0.0
                
                all_peaks.append({
                    'slice_idx': e_idx,
                    'slice_coord': e_val,
                    'position': k_pos,
                    'position_idx': p['k_index'],
                    'position_err': k_err,
                    'intensity': p['intensity'],
                    'normalized_amp': p['normalized_amplitude'],
                    'fwhm': p.get('fwhm', np.nan),
                    'fwhm_err': p.get('fwhm_err', np.nan),
                    'fit_quality': p.get('fit_result', {}).get('fit_quality', np.nan),
                    'assigned': False
                })
    
    else:  # EDC
        # Iterate over momentum indices
        for k_idx in range(n_momentum):
            k_val = k_coords[k_idx]
            
            # Skip if outside k range
            if k_range is not None:
                if k_val < k_range[0] or k_val > k_range[1]:
                    continue
            
            peaks = extract_edc_peaks(
                image, k_idx,
                energy_coords=energy_coords,
                threshold=threshold,
                min_distance=min_distance,
                fit_peaks=fit_peaks,
                fit_window=fit_window,
                energy_range=energy_range
            )
            
            for p in peaks:
                # Use fitted position if available
                if 'energy_fitted' in p:
                    e_pos = p['energy_fitted']
                    e_err = p['energy_err']
                else:
                    e_pos = p['energy']
                    e_err = 0.0
                
                all_peaks.append({
                    'slice_idx': k_idx,
                    'slice_coord': k_val,
                    'position': e_pos,
                    'position_idx': p['energy_index'],
                    'position_err': e_err,
                    'intensity': p['intensity'],
                    'normalized_amp': p['normalized_amplitude'],
                    'fwhm': p.get('fwhm', np.nan),
                    'fwhm_err': p.get('fwhm_err', np.nan),
                    'fit_quality': p.get('fit_result', {}).get('fit_quality', np.nan),
                    'assigned': False
                })
    
    if len(all_peaks) == 0:
        return []
    
    # Phase 2: Connect peaks into tracks
    tracks = _build_band_tracks(
        all_peaks,
        continuity_threshold=continuity_threshold,
        intensity_weight=intensity_weight,
        min_track_length=min_track_length
    )
    
    # Format output
    dispersions = []
    for band_id, track in enumerate(tracks):
        n_points = len(track)
        
        if axis == "MDC":
            # Positions are (energy, k)
            positions = np.array([[p['slice_coord'], p['position']] for p in track])
            positions_err = np.array([[0, p['position_err']] for p in track])
        else:
            # Positions are (energy, k)
            positions = np.array([[p['position'], p['slice_coord']] for p in track])
            positions_err = np.array([[p['position_err'], 0] for p in track])
        
        intensities = np.array([p['intensity'] for p in track])
        fwhm = np.array([p['fwhm'] for p in track])
        fwhm_err = np.array([p['fwhm_err'] for p in track])
        fit_quality = np.array([p['fit_quality'] for p in track])
        
        dispersions.append({
            'band_id': band_id,
            'n_points': n_points,
            'positions': positions,
            'positions_err': positions_err,
            'intensities': intensities,
            'fwhm': fwhm,
            'fwhm_err': fwhm_err,
            'fit_quality': fit_quality,
            'axis': axis
        })
    
    return dispersions


def _build_band_tracks(
    peaks: List[Dict],
    continuity_threshold: float,
    intensity_weight: float,
    min_track_length: int
) -> List[List[Dict]]:
    """
    Build band tracks by connecting peaks across slices.
    
    Uses a greedy nearest-neighbor algorithm with intensity weighting.
    """
    if len(peaks) == 0:
        return []
    
    # Sort peaks by slice index
    peaks_sorted = sorted(peaks, key=lambda p: p['slice_idx'])
    
    # Group by slice
    slice_groups = {}
    for p in peaks_sorted:
        idx = p['slice_idx']
        if idx not in slice_groups:
            slice_groups[idx] = []
        slice_groups[idx].append(p)
    
    slice_indices = sorted(slice_groups.keys())
    
    if len(slice_indices) < 2:
        # Not enough slices for tracking
        return []
    
    # Calculate intensity range for normalization
    all_intensities = [p['intensity'] for p in peaks]
    intensity_range = max(all_intensities) - min(all_intensities)
    if intensity_range < 1e-10:
        intensity_range = 1.0
    
    tracks = []
    
    # Start tracks from first slice
    for start_peak in slice_groups[slice_indices[0]]:
        if start_peak['assigned']:
            continue
        
        track = [start_peak]
        start_peak['assigned'] = True
        current_peak = start_peak
        
        # Extend track through subsequent slices
        for i in range(1, len(slice_indices)):
            slice_idx = slice_indices[i]
            candidates = [p for p in slice_groups[slice_idx] if not p['assigned']]
            
            if not candidates:
                continue
            
            # Find best matching peak
            best_peak = None
            best_cost = float('inf')
            
            for cand in candidates:
                # Spatial distance (in position index space for consistency)
                dist = abs(cand['position_idx'] - current_peak['position_idx'])
                
                if dist > continuity_threshold:
                    continue
                
                # Intensity difference (normalized)
                intensity_diff = abs(cand['intensity'] - current_peak['intensity']) / intensity_range
                
                # Combined cost
                cost = dist + intensity_weight * intensity_diff
                
                if cost < best_cost:
                    best_cost = cost
                    best_peak = cand
            
            if best_peak is not None:
                best_peak['assigned'] = True
                track.append(best_peak)
                current_peak = best_peak
        
        if len(track) >= min_track_length:
            tracks.append(track)
    
    # Try to start additional tracks from unassigned peaks
    for slice_idx in slice_indices:
        for peak in slice_groups[slice_idx]:
            if peak['assigned']:
                continue
            
            track = [peak]
            peak['assigned'] = True
            current_peak = peak
            
            # Look forward
            current_slice_pos = slice_indices.index(slice_idx)
            for i in range(current_slice_pos + 1, len(slice_indices)):
                next_slice_idx = slice_indices[i]
                candidates = [p for p in slice_groups[next_slice_idx] if not p['assigned']]
                
                if not candidates:
                    continue
                
                best_peak = None
                best_cost = float('inf')
                
                for cand in candidates:
                    dist = abs(cand['position_idx'] - current_peak['position_idx'])
                    if dist > continuity_threshold:
                        continue
                    
                    intensity_diff = abs(cand['intensity'] - current_peak['intensity']) / intensity_range
                    cost = dist + intensity_weight * intensity_diff
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_peak = cand
                
                if best_peak is not None:
                    best_peak['assigned'] = True
                    track.append(best_peak)
                    current_peak = best_peak
            
            if len(track) >= min_track_length:
                tracks.append(track)
    
    return tracks


# ==============================================================================
# Module exports
# ==============================================================================

__all__ = [
    'detect_peaks_1d',
    'fit_peak_lorentzian',
    'extract_mdc_peaks',
    'extract_edc_peaks',
    'extract_band_dispersion',
]
