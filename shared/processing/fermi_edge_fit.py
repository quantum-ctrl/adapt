"""
Automatic Fermi Edge Fitting Module

This module provides functions to automatically fit the Fermi edge within a
specified energy window in ARPES data.

The fitting uses a Fermi-Dirac distribution convolved with a Gaussian:
    I(E) = A * [1 / (1 + exp((E - E_F) / (k_B * T)))] ⊗ G(σ) + B

Where:
    E_F: Fermi level position
    T: Temperature
    σ: Gaussian broadening (energy resolution)
    A: Amplitude
    B: Background

Usage:
------
    from fermi_edge_fit import fit_fermi_edge, plot_fit_results
    
    # Define energy window (in eV)
    energy_window = (-0.3, 0.3)
    
    # Perform the fit
    results = fit_fermi_edge(data, energy_window)
    
    # Plot results
    plot_fit_results(results)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc

# Physical constants
kB = 8.617333262e-5  # Boltzmann constant in eV/K

# Numerical stability constants
MIN_KT = 1e-6        # Minimum thermal energy to avoid division by zero (eV)
MIN_SIGMA = 1e-6     # Minimum Gaussian width to avoid division by zero (eV)

# Default fitting parameters
DEFAULT_T_GUESS = 20.0      # Initial temperature guess (K)
DEFAULT_SIGMA_GUESS = 0.05  # Initial energy resolution guess (eV)


def fermi_dirac_gaussian(E, E_F, T, sigma, A, B):
    """
    Fermi-Dirac distribution convolved with a Gaussian.
    
    This is an analytical approximation valid when thermal broadening and
    Gaussian broadening are comparable.
    
    Parameters:
    -----------
    E : array-like
        Energy values (binding energy)
    E_F : float
        Fermi level position in eV
    T : float
        Temperature in Kelvin
    sigma : float
        Gaussian broadening (energy resolution) in eV
    A : float
        Amplitude
    B : float
        Background offset
    
    Returns:
    --------
    array-like: Fermi edge intensity profile
    """
    kT = kB * T
    
    # Avoid division by zero
    kT = max(kT, MIN_KT)
    sigma = max(sigma, MIN_SIGMA)
    
    # Total effective width combining thermal and instrumental broadening
    # The thermal broadening contributes with a factor of π/sqrt(3) ≈ 1.81
    total_width = np.sqrt(sigma**2 + (np.pi * kT / np.sqrt(3))**2)
    
    # Use complementary error function for the convolution
    x_eff = (E - E_F) / (np.sqrt(2) * total_width)
    
    return A * 0.5 * erfc(x_eff) + B


def fit_fermi_edge(data, energy_window, theta_range=None, T_fixed=None,
                   energy_dim='energy', angle_dim='angle'):
    """
    Automatically fit the Fermi edge within a given energy window.
    
    Parameters:
    -----------
    data : xarray.DataArray
        2D ARPES data with energy and angle dimensions
    energy_window : tuple
        (E_min, E_max) energy range for the fit in eV
    theta_range : tuple, optional
        (theta_min, theta_max) angle range to integrate over.
        If None, uses the full range.
    T_fixed : float, optional
        If specified, fix the temperature to this value (in Kelvin).
        Useful when the sample temperature is known.
    energy_dim : str
        Name of the energy dimension (default: 'energy')
    angle_dim : str
        Name of the angle dimension (default: 'angle')
    
    Returns:
    --------
    dict: Fit results containing:
        - 'E_F': Fermi level position (eV)
        - 'T': Temperature (K)
        - 'sigma': Energy resolution (eV)
        - 'A': Amplitude
        - 'B': Background
        - 'E_F_err', 'T_err', 'sigma_err', 'A_err', 'B_err': Fit errors
        - 'energy': Energy array used for fitting
        - 'edc': Integrated EDC used for fitting
        - 'fit': Fitted curve
        - 'success': Whether the fit was successful
    """
    # Input validation
    if not hasattr(data, 'coords'):
        raise TypeError("data must be an xarray.DataArray")
    
    if len(energy_window) != 2 or energy_window[0] >= energy_window[1]:
        raise ValueError("energy_window must be (E_min, E_max) with E_min < E_max")
    
    if energy_dim not in data.dims:
        raise ValueError(f"energy_dim '{energy_dim}' not found. Available: {list(data.dims)}")
    
    if angle_dim not in data.dims:
        raise ValueError(f"angle_dim '{angle_dim}' not found. Available: {list(data.dims)}")
    
    # Select energy window
    E_min, E_max = energy_window
    data_sel = data.sel({energy_dim: slice(E_min, E_max)})
    
    # Select theta range if specified
    if theta_range is not None:
        theta_min, theta_max = theta_range
        data_sel = data_sel.sel({angle_dim: slice(theta_min, theta_max)})
    
    # Integrate over angle to get EDC
    edc = data_sel.sum(dim=angle_dim).values.astype(float)
    energy = data_sel.coords[energy_dim].values
    
    # Normalize EDC
    edc_min = edc.min()
    edc_max = edc.max()
    if edc_max - edc_min > 0:
        edc_norm = (edc - edc_min) / (edc_max - edc_min)
    else:
        edc_norm = edc - edc_min
    
    # Initial parameter guesses
    E_F_guess = (E_min + E_max) / 2  # Guess Fermi level at center of window
    T_guess = DEFAULT_T_GUESS if T_fixed is None else T_fixed
    sigma_guess = DEFAULT_SIGMA_GUESS
    A_guess = 1.0
    B_guess = 0.0
    
    # Prepare for fitting
    if T_fixed is not None:
        # Fix temperature - use a modified fitting function
        def fit_func(E, E_F, sigma, A, B):
            return fermi_dirac_gaussian(E, E_F, T_fixed, sigma, A, B)
        
        p0 = [E_F_guess, sigma_guess, A_guess, B_guess]
        bounds = (
            [E_min, 0.001, 0.0, -1.0],  # Lower bounds
            [E_max, 0.5, 10.0, 1.0]      # Upper bounds
        )
        param_names = ['E_F', 'sigma', 'A', 'B']
    else:
        fit_func = fermi_dirac_gaussian
        p0 = [E_F_guess, T_guess, sigma_guess, A_guess, B_guess]
        bounds = (
            [E_min, 1.0, 0.001, 0.0, -1.0],  # Lower bounds
            [E_max, 500.0, 0.5, 10.0, 1.0]   # Upper bounds
        )
        param_names = ['E_F', 'T', 'sigma', 'A', 'B']
    
    try:
        popt, pcov = curve_fit(
            fit_func, 
            energy, 
            edc_norm, 
            p0=p0,
            bounds=bounds,
            maxfev=10000
        )
        perr = np.sqrt(np.diag(pcov))
        
        # Build results dictionary
        results = {
            'energy': energy,
            'edc': edc_norm,
            'edc_raw': edc,
            'success': True,
            'energy_window': energy_window,
            'theta_range': theta_range
        }
        
        if T_fixed is not None:
            results['E_F'] = popt[0]
            results['T'] = T_fixed
            results['sigma'] = popt[1]
            results['A'] = popt[2]
            results['B'] = popt[3]
            results['E_F_err'] = perr[0]
            results['T_err'] = 0.0  # Fixed, no error
            results['sigma_err'] = perr[1]
            results['A_err'] = perr[2]
            results['B_err'] = perr[3]
            results['fit'] = fit_func(energy, *popt)
        else:
            results['E_F'] = popt[0]
            results['T'] = popt[1]
            results['sigma'] = popt[2]
            results['A'] = popt[3]
            results['B'] = popt[4]
            results['E_F_err'] = perr[0]
            results['T_err'] = perr[1]
            results['sigma_err'] = perr[2]
            results['A_err'] = perr[3]
            results['B_err'] = perr[4]
            results['fit'] = fermi_dirac_gaussian(energy, *popt)
        
    except (RuntimeError, ValueError, np.linalg.LinAlgError) as e:
        print(f"Fitting failed: {e}")
        results = {
            'success': False,
            'error': str(e),
            'energy': energy,
            'edc': edc_norm,
            'edc_raw': edc,
            'energy_window': energy_window,
            'theta_range': theta_range
        }
    
    return results


def plot_fit_results(results, figsize=(12, 5)):
    """
    Plot the Fermi edge fit results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from fit_fermi_edge()
    figsize : tuple
        Figure size in inches
    
    Returns:
    --------
    matplotlib.figure.Figure: The figure object
    """
    if not results['success']:
        print("Cannot plot: fitting was not successful.")
        if 'error' in results:
            print(f"Error: {results['error']}")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: EDC and fit
    ax1 = axes[0]
    ax1.plot(results['energy'], results['edc'], 'b.', markersize=4, label='Data')
    ax1.plot(results['energy'], results['fit'], 'r-', linewidth=2, label='Fit')
    ax1.axvline(results['E_F'], color='g', linestyle='--', 
                label=f'$E_F$ = {results["E_F"]:.4f} eV')
    ax1.set_xlabel('Binding Energy (eV)', fontsize=12)
    ax1.set_ylabel('Normalized Intensity', fontsize=12)
    ax1.set_title('Fermi Edge Fit', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Residuals
    ax2 = axes[1]
    residuals = results['edc'] - results['fit']
    ax2.plot(results['energy'], residuals, 'k.-', markersize=4)
    ax2.axhline(0, color='r', linestyle='--')
    ax2.set_xlabel('Binding Energy (eV)', fontsize=12)
    ax2.set_ylabel('Residuals', fontsize=12)
    ax2.set_title('Fit Residuals', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_fit_summary(results):
    """
    Print a summary of the fit results.
    
    Parameters:
    -----------
    results : dict
        Results dictionary from fit_fermi_edge()
    """
    if not results['success']:
        print("Fermi edge fitting failed.")
        if 'error' in results:
            print(f"Error: {results['error']}")
        return
    
    print("=" * 50)
    print("Fermi Edge Fit Results")
    print("=" * 50)
    print(f"Energy window: [{results['energy_window'][0]:.3f}, {results['energy_window'][1]:.3f}] eV")
    if results['theta_range'] is not None:
        print(f"Theta range: [{results['theta_range'][0]:.2f}, {results['theta_range'][1]:.2f}]°")
    print("-" * 50)
    print(f"E_F   = {results['E_F']:.4f} ± {results['E_F_err']:.4f} eV")
    print(f"T     = {results['T']:.1f} ± {results['T_err']:.1f} K")
    print(f"σ     = {results['sigma']*1000:.1f} ± {results['sigma_err']*1000:.1f} meV")
    print("=" * 50)


def fit_fermi_edge_3d(data, energy_window, scan_dim='scan', theta_range=None, 
                      T_fixed=None, show_progress=True):
    """
    Fit Fermi edge for each slice along a scan dimension in 3D data.
    
    Parameters:
    -----------
    data : xarray.DataArray
        3D ARPES data with dimensions like 'energy', 'angle', and 'scan'
    energy_window : tuple
        (E_min, E_max) energy range for the fit in eV
    scan_dim : str
        Name of the scan dimension to iterate over (default: 'scan')
    theta_range : tuple, optional
        (theta_min, theta_max) angle range to integrate over
    T_fixed : float, optional
        If specified, fix the temperature to this value (in Kelvin)
    show_progress : bool
        Whether to show progress bar (requires tqdm)
    
    Returns:
    --------
    dict: Batch fit results containing:
        - 'E_F': Array of Fermi levels for each scan
        - 'T': Array of temperatures
        - 'sigma': Array of energy resolutions
        - 'E_F_err', 'T_err', 'sigma_err': Arrays of errors
        - 'scan_coords': Scan coordinate values
        - 'success': Boolean array indicating fit success
        - 'individual_results': List of full results for each scan
    """
    # Get scan coordinates
    scan_coords = data.coords[scan_dim].values
    n_scans = len(scan_coords)
    
    # Initialize result arrays
    E_F_array = np.full(n_scans, np.nan)
    T_array = np.full(n_scans, np.nan)
    sigma_array = np.full(n_scans, np.nan)
    E_F_err_array = np.full(n_scans, np.nan)
    T_err_array = np.full(n_scans, np.nan)
    sigma_err_array = np.full(n_scans, np.nan)
    success_array = np.zeros(n_scans, dtype=bool)
    individual_results = []
    
    # Try to use tqdm for progress bar
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(enumerate(scan_coords), total=n_scans, desc="Fitting Fermi edges")
        except ImportError:
            print("Note: Install 'tqdm' for progress bar. Processing...")
            iterator = enumerate(scan_coords)
    else:
        iterator = enumerate(scan_coords)
    
    # Fit each scan
    for i, scan_val in iterator:
        # Select 2D slice
        data_2d = data.sel({scan_dim: scan_val})
        
        # Perform fit
        result = fit_fermi_edge(data_2d, energy_window, 
                               theta_range=theta_range, T_fixed=T_fixed)
        individual_results.append(result)
        
        if result['success']:
            E_F_array[i] = result['E_F']
            T_array[i] = result['T']
            sigma_array[i] = result['sigma']
            E_F_err_array[i] = result['E_F_err']
            T_err_array[i] = result['T_err']
            sigma_err_array[i] = result['sigma_err']
            success_array[i] = True
    
    # Summary
    n_success = np.sum(success_array)
    print(f"\nFitting complete: {n_success}/{n_scans} successful")
    
    return {
        'E_F': E_F_array,
        'T': T_array,
        'sigma': sigma_array,
        'E_F_err': E_F_err_array,
        'T_err': T_err_array,
        'sigma_err': sigma_err_array,
        'scan_coords': scan_coords,
        'scan_dim': scan_dim,
        'success': success_array,
        'individual_results': individual_results,
        'energy_window': energy_window
    }


def plot_batch_fit_results(batch_results, figsize=(12, 8)):
    """
    Plot the batch Fermi edge fit results.
    
    Parameters:
    -----------
    batch_results : dict
        Results from fit_fermi_edge_3d()
    figsize : tuple
        Figure size
    
    Returns:
    --------
    matplotlib.figure.Figure: The figure object
    """
    scan_coords = batch_results['scan_coords']
    success = batch_results['success']
    scan_dim = batch_results.get('scan_dim', 'scan')
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Plot E_F vs scan
    ax1 = axes[0, 0]
    ax1.errorbar(scan_coords[success], batch_results['E_F'][success],
                 yerr=batch_results['E_F_err'][success],
                 fmt='o-', capsize=3, markersize=4)
    ax1.set_xlabel(scan_dim, fontsize=12)
    ax1.set_ylabel('$E_F$ (eV)', fontsize=12)
    ax1.set_title('Fermi Level vs Scan', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Plot sigma vs scan
    ax2 = axes[0, 1]
    sigma_meV = batch_results['sigma'][success] * 1000
    sigma_err_meV = batch_results['sigma_err'][success] * 1000
    ax2.errorbar(scan_coords[success], sigma_meV,
                 yerr=sigma_err_meV,
                 fmt='s-', capsize=3, markersize=4, color='green')
    ax2.set_xlabel(scan_dim, fontsize=12)
    ax2.set_ylabel('σ (meV)', fontsize=12)
    ax2.set_title('Energy Resolution vs Scan', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Plot T vs scan (if not fixed)
    ax3 = axes[1, 0]
    if not np.all(batch_results['T_err'] == 0):
        ax3.errorbar(scan_coords[success], batch_results['T'][success],
                     yerr=batch_results['T_err'][success],
                     fmt='^-', capsize=3, markersize=4, color='red')
        ax3.set_ylabel('T (K)', fontsize=12)
        ax3.set_title('Temperature vs Scan', fontsize=14)
    else:
        T_fixed = batch_results['T'][success][0]
        ax3.axhline(T_fixed, color='red', linestyle='--')
        ax3.set_ylabel('T (K)', fontsize=12)
        ax3.set_title(f'Temperature (fixed at {T_fixed:.1f} K)', fontsize=14)
    ax3.set_xlabel(scan_dim, fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    E_F_mean = np.nanmean(batch_results['E_F'][success])
    E_F_std = np.nanstd(batch_results['E_F'][success])
    sigma_mean = np.nanmean(batch_results['sigma'][success]) * 1000
    sigma_std = np.nanstd(batch_results['sigma'][success]) * 1000
    
    summary_text = f"""
    Fit Summary
    ═══════════════════════════
    Successful fits: {np.sum(success)}/{len(success)}
    
    E_F:  {E_F_mean:.4f} ± {E_F_std:.4f} eV
    σ:    {sigma_mean:.1f} ± {sigma_std:.1f} meV
    
    Energy window: {batch_results['energy_window']}
    """
    ax4.text(0.1, 0.5, summary_text, fontsize=12, family='monospace',
             verticalalignment='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    return fig


# ==============================================================================
# Example usage (when run as a script)
# ==============================================================================
if __name__ == "__main__":
    import sys
    import os
    
    try:
        import loaders.load_adress_data as loader
    except ImportError:
        print("Error: Cannot import loaders. Please run from project root or install package.")
        sys.exit(1)
    
    # Load example data
    file_path = "../data/adress/S7_013_cut_376eV.h5"
    
    if os.path.exists(file_path):
        data = loader.load(file_path)
        print("Data loaded successfully!")
        print(data)
        
        # Define energy window for Fermi edge fitting
        energy_window = (-0.3, 0.3)  # eV
        
        # Perform the fit
        results = fit_fermi_edge(data, energy_window)
        
        # Print summary
        print_fit_summary(results)
        
        # Plot results
        if results['success']:
            fig = plot_fit_results(results)
            plt.show()
    else:
        print(f"Example data file not found: {file_path}")
        print("Please adjust the file path or run from the notebooks directory.")
