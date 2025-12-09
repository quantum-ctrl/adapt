"""
ARPES angle-to-momentum (k-space) conversion module.

Converts angle-resolved photoemission data from angle space to momentum space,
properly handling the energy-dependent momentum calculation for each binding energy.
"""

import numpy as np
import xarray as xr

# Physical constants and conversion factor
# k (Å⁻¹) = K_FACTOR * sqrt(Ek (eV)) * sin(theta)
# where K_FACTOR = sqrt(2 * m_e) / ħ with proper unit conversions
K_FACTOR = 0.5123  # Å⁻¹ / sqrt(eV), well-known ARPES constant

__all__ = [
    'convert_angle_to_k',
    'convert_2d_angle_to_k', 
    'convert_3d_kxky_to_k',
    'convert_3d_hv_to_k',
    'convert_hv_to_kxkz',
]


def _calc_k_parallel(Ek, theta):
    """
    Calculate parallel momentum k_parallel.
    
    Parameters
    ----------
    Ek : ndarray
        Kinetic energy (eV)
    theta : ndarray
        Emission angle (radians), broadcastable with Ek
        
    Returns
    -------
    ndarray
        k_parallel in Å⁻¹
    """
    return K_FACTOR * np.sqrt(np.maximum(Ek, 0)) * np.sin(theta)


def _calc_kz(Ek, theta, V0):
    """
    Calculate perpendicular momentum kz using the free-electron final state model.
    
    Parameters
    ----------
    Ek : ndarray
        Kinetic energy (eV)
    theta : ndarray
        Emission angle (radians)
    V0 : float
        Inner potential (eV), typically 10-15 eV for most materials
        
    Returns
    -------
    ndarray
        kz in Å⁻¹
    """
    return K_FACTOR * np.sqrt(np.maximum(Ek * np.cos(theta)**2 + V0, 0))


def convert_2d_angle_to_k(data, hv, phi=4.5, interpolate=True, k_points=None, fill_value=0.0):
    """
    Convert 2D angle-energy data to k-E space.
    
    k is calculated separately for each binding energy:
    k(angle, E) = K_FACTOR * sqrt(Ek(E)) * sin(angle)
    where Ek = hv - phi - |E_binding|
    
    Parameters
    ----------
    data : xarray.DataArray
        2D data with dimensions (angle, energy)
        - angle: emission angle in degrees
        - energy: binding energy in eV (negative below Fermi level)
    hv : float
        Photon energy in eV
    phi : float, optional
        Work function in eV, default 4.5 eV
    interpolate : bool, optional
        If True, interpolate data onto a regular k-grid (default True)
        If False, just add k as a 2D coordinate without interpolation
    k_points : int, optional
        Number of k points for interpolation. If None, uses same as angle points.
    fill_value : float, optional
        Value to fill for points outside the interpolation range (default 0.0).
        Use np.nan if you want to preserve NaN for out-of-range values.
        
    Returns
    -------
    xarray.DataArray
        If interpolate=True: Data with dimension 'k' replacing 'angle'
        If interpolate=False: Data with 'k' as a 2D coordinate (varies with energy)
    """
    from scipy.interpolate import interp1d
    
    # Get coordinates
    angles_deg = data.coords['angle'].values
    energies = data.coords['energy'].values
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles_deg)
    
    # Calculate kinetic energy for each binding energy
    # Ek = hv - phi - |E_binding| = hv - phi + E (since E is negative for occupied states)
    Ek = hv - phi + energies  # shape: (N_energy,)
    
    # Calculate k for each (angle, energy) point
    # angles_rad: (N_angle,) -> (N_angle, 1)
    # Ek: (N_energy,) -> (1, N_energy)
    angles_2d = angles_rad.reshape(-1, 1)
    Ek_2d = Ek.reshape(1, -1)
    
    k_values = _calc_k_parallel(Ek_2d, angles_2d)  # shape: (N_angle, N_energy)
    
    if not interpolate:
        # Just add k as a 2D coordinate without interpolation
        result = data.copy()
        result = result.assign_coords(
            k=(('angle', 'energy'), k_values)
        )
        result.attrs['conversion'] = 'angle_to_k'
        result.attrs['hv'] = hv
        result.attrs['phi'] = phi
        result.attrs['interpolated'] = False
        return result
    
    # Interpolate to regular k-grid
    # Use k range at Fermi level (or highest Ek) as reference
    k_min = k_values.min()
    k_max = k_values.max()
    
    if k_points is None:
        k_points = len(angles_deg)
    
    k_grid = np.linspace(k_min, k_max, k_points)
    
    # Get data values - handle dimension order
    if data.dims[0] == 'angle':
        data_values = data.values  # (angle, energy)
    else:
        data_values = data.values.T  # transpose to (angle, energy)
    
    # Interpolate each energy slice from angle-space to k-space
    data_k = np.zeros((len(energies), k_points))  # (energy, k) to match typical (Eb, angle) layout
    
    for i_e, energy in enumerate(energies):
        k_at_energy = k_values[:, i_e]  # k values at this energy
        intensity_at_energy = data_values[:, i_e]  # intensity at this energy
        
        # Sort by k (required for interpolation)
        sort_idx = np.argsort(k_at_energy)
        k_sorted = k_at_energy[sort_idx]
        intensity_sorted = intensity_at_energy[sort_idx]
        
        # Create interpolator and interpolate onto regular k-grid
        # Use linear interpolation, fill NaN outside range
        f = interp1d(k_sorted, intensity_sorted, kind='linear', 
                     bounds_error=False, fill_value=fill_value)
        data_k[i_e, :] = f(k_grid)
    
    # Create new DataArray with dimensions (energy, k) to match original (energy, angle) layout
    result = xr.DataArray(
        data_k,
        dims=['energy', 'k'],
        coords={
            'energy': energies,
            'k': k_grid
        },
        attrs={
            'conversion': 'angle_to_k',
            'hv': hv,
            'phi': phi,
            'interpolated': True,
            'original_angle_range': (angles_deg.min(), angles_deg.max()),
        }
    )
    
    return result


def convert_3d_kxky_to_k(data, hv, phi=4.5, angle_dims=('angle_x', 'angle_y'),
                          energy_dim='energy', interpolate=True, k_points=None, fill_value=0.0):
    """
    Convert 3D kx-ky-E mapping data to momentum space.
    
    kx and ky are calculated separately for each binding energy:
    kx(angle_x, E) = K_FACTOR * sqrt(Ek(E)) * sin(angle_x)
    ky(angle_y, E) = K_FACTOR * sqrt(Ek(E)) * sin(angle_y)
    
    Parameters
    ----------
    data : xarray.DataArray
        3D data with dimensions (angle_x, angle_y, energy)
        - angle_x, angle_y: emission angles in degrees
        - energy: binding energy in eV
    hv : float
        Photon energy in eV
    phi : float, optional
        Work function in eV, default 4.5 eV
    angle_dims : tuple of str, optional
        Names of the two angle dimensions, default ('angle_x', 'angle_y')
    energy_dim : str, optional
        Name of the energy dimension, default 'energy'
    interpolate : bool, optional
        If True, interpolate data onto a regular k-grid (default True)
        If False, just add kx, ky as 2D coordinates without interpolation
    k_points : int or tuple, optional
        Number of k points for interpolation. If None, uses same as angle points.
        Can be int (same for kx and ky) or tuple (kx_points, ky_points).
    fill_value : float, optional
        Value to fill for points outside the interpolation range (default 0.0).
        Use np.nan if you want to preserve NaN for out-of-range values.
        
    Returns
    -------
    xarray.DataArray
        If interpolate=True: Data with dimensions ('kx', 'ky', energy_dim)
        If interpolate=False: Data with kx, ky as 2D coordinates (vary with energy)
    """
    from scipy.interpolate import RegularGridInterpolator
    
    angle_x_name, angle_y_name = angle_dims
    
    # Get coordinates
    angles_x_deg = data.coords[angle_x_name].values
    angles_y_deg = data.coords[angle_y_name].values
    energies = data.coords[energy_dim].values
    
    # Convert angles to radians
    angles_x_rad = np.deg2rad(angles_x_deg)
    angles_y_rad = np.deg2rad(angles_y_deg)
    
    # Calculate kinetic energy
    Ek = hv - phi + energies  # shape: (N_energy,)
    
    # Calculate kx for each (angle_x, energy)
    angles_x_2d = angles_x_rad.reshape(-1, 1)
    Ek_2d = Ek.reshape(1, -1)
    kx_values = _calc_k_parallel(Ek_2d, angles_x_2d)  # shape: (N_angle_x, N_energy)
    
    # Calculate ky for each (angle_y, energy)
    angles_y_2d = angles_y_rad.reshape(-1, 1)
    ky_values = _calc_k_parallel(Ek_2d, angles_y_2d)  # shape: (N_angle_y, N_energy)
    
    if not interpolate:
        # Create new DataArray with kx, ky as 2D coordinates
        result = data.copy()
        result = result.assign_coords(
            kx=((angle_x_name, energy_dim), kx_values),
            ky=((angle_y_name, energy_dim), ky_values)
        )
        result.attrs['conversion'] = 'angle_to_k'
        result.attrs['hv'] = hv
        result.attrs['phi'] = phi
        result.attrs['interpolated'] = False
        return result
    
    # Interpolate to regular k-grid
    # Determine k_points
    if k_points is None:
        kx_points = len(angles_x_deg)
        ky_points = len(angles_y_deg)
    elif isinstance(k_points, tuple):
        kx_points, ky_points = k_points
    else:
        kx_points = ky_points = k_points
    
    # Create regular k-grids (use range at Fermi level / highest Ek)
    kx_min, kx_max = kx_values.min(), kx_values.max()
    ky_min, ky_max = ky_values.min(), ky_values.max()
    
    kx_grid = np.linspace(kx_min, kx_max, kx_points)
    ky_grid = np.linspace(ky_min, ky_max, ky_points)
    
    # Interpolate for each energy slice
    # Output shape: (energy, kx, ky) to match typical (Eb, kx, ky) layout
    data_k = np.zeros((len(energies), kx_points, ky_points))
    
    # Get data values in correct order (angle_x, angle_y, energy)
    dim_order = list(data.dims)
    target_order = [angle_x_name, angle_y_name, energy_dim]
    if dim_order != target_order:
        transpose_idx = [dim_order.index(d) for d in target_order]
        data_values = np.transpose(data.values, transpose_idx)
    else:
        data_values = data.values
    
    for i_e in range(len(energies)):
        kx_at_e = kx_values[:, i_e]
        ky_at_e = ky_values[:, i_e]
        slice_data = data_values[:, :, i_e]
        
        # Create interpolator for this energy slice
        interp = RegularGridInterpolator(
            (kx_at_e, ky_at_e), slice_data,
            method='linear', bounds_error=False, fill_value=fill_value
        )
        
        # Create meshgrid for target k-values
        kx_mesh, ky_mesh = np.meshgrid(kx_grid, ky_grid, indexing='ij')
        points = np.column_stack([kx_mesh.ravel(), ky_mesh.ravel()])
        
        # Interpolate
        data_k[i_e, :, :] = interp(points).reshape(kx_points, ky_points)
    
    # Create new DataArray with kx, ky as dimensions
    result = xr.DataArray(
        data_k,
        dims=[energy_dim, 'kx', 'ky'],
        coords={
            energy_dim: energies,
            'kx': kx_grid,
            'ky': ky_grid
        },
        attrs={
            'conversion': 'angle_to_k',
            'hv': hv,
            'phi': phi,
            'interpolated': True,
            'original_angle_x_range': (angles_x_deg.min(), angles_x_deg.max()),
            'original_angle_y_range': (angles_y_deg.min(), angles_y_deg.max()),
        }
    )
    
    return result


def convert_3d_hv_to_k(data, phi=4.5, V0=12.57, convert_hv_to_kz=False, 
                        angle_dim='angle', energy_dim='energy', hv_dim='hv',
                        interpolate=True, k_points=None, fill_value=0.0):
    """
    Convert 3D angle-energy-hv scan data to momentum space.
    
    Follows the algorithm from MATLAB convert_to_k.m for Eb(kx,kz) type data.
    
    For each (angle, energy, hv) point:
    - Ek(E, hv) = hv - phi + E  (E is negative for occupied states)
    - k(angle, E, hv) = K_FACTOR * sqrt(Ek) * sin(angle)
    - kz(angle, E, hv) = K_FACTOR * sqrt(Ek * cos²(angle) + V0)
    
    Parameters
    ----------
    data : xarray.DataArray
        3D data with dimensions (angle, energy, hv)
        - angle: emission angle in degrees
        - energy: binding energy in eV (negative below Fermi level)
        - hv: photon energy in eV
    phi : float, optional
        Work function in eV, default 4.5 eV
    V0 : float, optional
        Inner potential in eV, default 12.57
    convert_hv_to_kz : bool, optional
        If True, calculate kz coordinate for each (E, k, hv) point
        If False, keep hv as the third dimension coordinate
    angle_dim : str, optional
        Name of the angle dimension, default 'angle'
    energy_dim : str, optional
        Name of the energy dimension, default 'energy'
    hv_dim : str, optional
        Name of the photon energy dimension, default 'hv'
    interpolate : bool, optional
        If True, interpolate data onto a regular k-grid (default True)
        If False, just add k and kz as 3D coordinates without interpolation
    k_points : int, optional
        Number of k points for interpolation. If None, uses same as angle points.
    fill_value : float, optional
        Value to fill for points outside the interpolation range (default 0.0).
        
    Returns
    -------
    xarray.DataArray
        If interpolate=True: Data with dimension 'k' replacing angle_dim
        If interpolate=False: Data with k as 3D coordinate (varies with energy and hv)
        If convert_hv_to_kz=True: includes kz_full as 3D coordinate
    
    Notes
    -----
    Following MATLAB convert_to_k.m, kz is calculated for every (Eb, theta, hv) point:
    
    ```matlab
    for i = 1:size(dataStr.hv, 2)
        dataStr.kz(:,:,i) = Kzz(hv(i), eb(:,:,i), thtM, tht(:,:,i), tlt, v000, ...);
    end
    ```
    
    The kz formula: kz = K_FACTOR * sqrt(Ek * cos²(theta) + V0)
    where Ek = hv - phi - |Eb|
    """
    from scipy.interpolate import interp1d
    
    # Get coordinates
    angles_deg = data.coords[angle_dim].values
    energies = data.coords[energy_dim].values
    hvs = data.coords[hv_dim].values
    
    n_angle = len(angles_deg)
    n_energy = len(energies)
    n_hv = len(hvs)
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles_deg)
    
    # Calculate kinetic energy for each (energy, hv) combination
    # Ek(E, hv) = hv - phi + E (E is negative for occupied states, so this is correct)
    # Shape: (N_angle, N_energy, N_hv) via broadcasting
    Ek_3d = hvs.reshape(1, 1, -1) - phi + energies.reshape(1, -1, 1)
    # Ek_3d shape: (1, N_energy, N_hv) - will broadcast with angles
    
    # Expand to full 3D for explicit calculations
    Ek_full = np.broadcast_to(Ek_3d, (n_angle, n_energy, n_hv)).copy()
    Ek_full = np.maximum(Ek_full, 0)  # Clamp negative kinetic energies
    
    # angles for 3D calculation
    angles_3d = angles_rad.reshape(-1, 1, 1)
    angles_full = np.broadcast_to(angles_3d, (n_angle, n_energy, n_hv))
    
    # Calculate k for each (angle, energy, hv) point
    # k = K_FACTOR * sqrt(Ek) * sin(theta)
    k_values = K_FACTOR * np.sqrt(Ek_full) * np.sin(angles_full)
    # k_values shape: (N_angle, N_energy, N_hv)
    
    # Calculate kz for each (angle, energy, hv) point
    # kz = K_FACTOR * sqrt(Ek * cos²(theta) + V0)
    # This is the key difference from before: kz depends on Eb, not just E=0
    cos_theta_sq = np.cos(angles_full) ** 2
    kz_values = K_FACTOR * np.sqrt(np.maximum(Ek_full * cos_theta_sq + V0, 0))
    # kz_values shape: (N_angle, N_energy, N_hv)
    
    if not interpolate:
        # Create new DataArray with k and kz as 3D coordinates
        result = data.copy()
        result = result.assign_coords(
            k=((angle_dim, energy_dim, hv_dim), k_values),
            kz=((angle_dim, energy_dim, hv_dim), kz_values)
        )
        result.attrs['conversion'] = 'angle_to_k'
        result.attrs['phi'] = phi
        result.attrs['V0'] = V0
        result.attrs['interpolated'] = False
        return result
    
    # Interpolate to regular k-grid
    k_min = k_values.min()
    k_max = k_values.max()
    
    if k_points is None:
        k_points = n_angle
    
    k_grid = np.linspace(k_min, k_max, k_points)
    
    # Get data values in correct order (angle, energy, hv)
    dim_order = list(data.dims)
    target_order = [angle_dim, energy_dim, hv_dim]
    if dim_order != target_order:
        transpose_idx = [dim_order.index(d) for d in target_order]
        data_values = np.transpose(data.values, transpose_idx)
    else:
        data_values = data.values
    
    # Interpolate each (energy, hv) slice from angle-space to k-space
    # Output shape: (N_energy, N_k, N_hv) to match typical (Eb, k, hv) layout
    data_k = np.zeros((n_energy, k_points, n_hv))
    kz_interp = np.zeros((n_energy, k_points, n_hv))  # kz for interpolated grid
    
    for i_e in range(n_energy):
        for i_hv in range(n_hv):
            k_at_slice = k_values[:, i_e, i_hv]
            kz_at_slice = kz_values[:, i_e, i_hv]
            intensity_at_slice = data_values[:, i_e, i_hv]
            
            # Sort by k (required for interpolation)
            sort_idx = np.argsort(k_at_slice)
            k_sorted = k_at_slice[sort_idx]
            kz_sorted = kz_at_slice[sort_idx]
            intensity_sorted = intensity_at_slice[sort_idx]
            
            # Create interpolator for intensity
            f_intensity = interp1d(k_sorted, intensity_sorted, kind='linear',
                                   bounds_error=False, fill_value=fill_value)
            data_k[i_e, :, i_hv] = f_intensity(k_grid)
            
            # Interpolate kz onto regular k-grid
            f_kz = interp1d(k_sorted, kz_sorted, kind='linear',
                           bounds_error=False, fill_value=np.nan)
            kz_interp[i_e, :, i_hv] = f_kz(k_grid)
    
    # Create new DataArray with k as dimension
    if convert_hv_to_kz:
        # For the hv dimension coordinate, use mean kz at k~0 for each hv
        # This follows MATLAB convention where kz is summarized per hv slice
        k_center_idx = np.argmin(np.abs(k_grid))
        # Average over energies for the "representative" kz at each hv
        kz_coord = np.nanmean(kz_interp[:, k_center_idx, :], axis=0)
        
        result = xr.DataArray(
            data_k,
            dims=[energy_dim, 'k', hv_dim],
            coords={
                energy_dim: energies,
                'k': k_grid,
                hv_dim: kz_coord  # Use representative kz as coordinate
            },
            attrs={
                'conversion': 'angle_to_k_kz',
                'phi': phi,
                'V0': V0,
                'interpolated': True,
                'original_angle_range': (angles_deg.min(), angles_deg.max()),
                'original_hv_range': (hvs.min(), hvs.max()),
            }
        )
        
        # Store the full kz(E, k, hv) as a 3D coordinate for precision
        result = result.assign_coords(
            kz_full=((energy_dim, 'k', hv_dim), kz_interp)
        )
    else:
        result = xr.DataArray(
            data_k,
            dims=[energy_dim, 'k', hv_dim],
            coords={
                energy_dim: energies,
                'k': k_grid,
                hv_dim: hvs
            },
            attrs={
                'conversion': 'angle_to_k',
                'phi': phi,
                'interpolated': True,
                'original_angle_range': (angles_deg.min(), angles_deg.max()),
            }
        )
    
    return result


def convert_hv_to_kxkz(data, phi=4.5, V0=12.57, 
                        angle_dim='angle', energy_dim='energy', hv_dim='hv',
                        k_points=None, kz_points=None, fill_value=0.0):
    """
    Convert hv mapping data directly to a regular (k∥, kz) grid.
    
    This function maps each data point (θ, E, hv) to its true (k∥, kz) coordinates
    based on ARPES physics, then interpolates all points onto a uniform (k∥, kz) 
    grid using 2D interpolation.
    
    This is the physically correct approach for hv mapping data where:
    - Different hv values probe different kz
    - At the same hv, different θ values have slightly different kz
    - The result is a proper Fermi surface map in (k∥, kz) space
    
    Parameters
    ----------
    data : xarray.DataArray
        3D hv mapping data with dimensions (angle, energy, hv)
        - angle: emission angle in degrees
        - energy: binding energy in eV (negative below Fermi level)
        - hv: photon energy in eV
    phi : float, optional
        Work function in eV, default 4.5 eV
    V0 : float, optional
        Inner potential in eV, default 12.57 eV
    angle_dim : str, optional
        Name of the angle dimension, default 'angle'
    energy_dim : str, optional
        Name of the energy dimension, default 'energy'
    hv_dim : str, optional
        Name of the photon energy dimension, default 'hv'
    k_points : int, optional
        Number of k∥ points in output grid. If None, uses same as angle points.
    kz_points : int, optional
        Number of kz points in output grid. If None, uses same as hv points.
    fill_value : float, optional
        Value to fill for points outside the interpolation range (default 0.0).
        
    Returns
    -------
    xarray.DataArray
        3D data with dimensions (energy, k, kz) on a regular grid
    
    Notes
    -----
    For each (θ, E, hv) point:
    - Ek = hv - phi + E  (kinetic energy)
    - k∥ = K_FACTOR * sqrt(Ek) * sin(θ)
    - kz = K_FACTOR * sqrt(Ek * cos²θ + V0)
    
    Then all points are interpolated onto a regular (k∥, kz) grid using
    scipy.interpolate.griddata with linear interpolation.
    
    Example
    -------
    >>> # Convert hv mapping to (k∥, kz) space
    >>> data_kxkz = convert_hv_to_kxkz(data, V0=12.57)
    >>> 
    >>> # Plot directly - no special handling needed
    >>> plot_3d_data(data_kxkz)  # XY: E vs k, YZ: k vs kz, XZ: E vs kz
    """
    from scipy.interpolate import griddata
    
    # Get coordinates
    angles_deg = data.coords[angle_dim].values
    energies = data.coords[energy_dim].values
    hvs = data.coords[hv_dim].values
    
    n_angle = len(angles_deg)
    n_energy = len(energies)
    n_hv = len(hvs)
    
    # Convert angles to radians
    angles_rad = np.deg2rad(angles_deg)
    
    # Prepare output grid sizes
    if k_points is None:
        k_points = n_angle
    if kz_points is None:
        kz_points = n_hv
    
    # Get data values in correct order (angle, energy, hv)
    dim_order = list(data.dims)
    target_order = [angle_dim, energy_dim, hv_dim]
    if dim_order != target_order:
        transpose_idx = [dim_order.index(d) for d in target_order]
        data_values = np.transpose(data.values, transpose_idx)
    else:
        data_values = data.values
    
    # Calculate k∥ and kz for ALL points (3D arrays)
    # Broadcast: angles(N_a,1,1) x energies(1,N_e,1) x hvs(1,1,N_hv)
    angles_3d = angles_rad.reshape(-1, 1, 1)
    energies_3d = energies.reshape(1, -1, 1)
    hvs_3d = hvs.reshape(1, 1, -1)
    
    # Kinetic energy for each point
    Ek_3d = hvs_3d - phi + energies_3d
    Ek_3d = np.maximum(Ek_3d, 0)  # Clamp negative values
    
    # k∥ = K_FACTOR * sqrt(Ek) * sin(θ)
    k_3d = K_FACTOR * np.sqrt(Ek_3d) * np.sin(angles_3d)
    
    # kz = K_FACTOR * sqrt(Ek * cos²θ + V0)
    cos_theta_sq = np.cos(angles_3d) ** 2
    kz_3d = K_FACTOR * np.sqrt(np.maximum(Ek_3d * cos_theta_sq + V0, 0))
    
    # Determine output grid range
    k_min, k_max = np.nanmin(k_3d), np.nanmax(k_3d)
    kz_min, kz_max = np.nanmin(kz_3d), np.nanmax(kz_3d)
    
    k_grid = np.linspace(k_min, k_max, k_points)
    kz_grid = np.linspace(kz_min, kz_max, kz_points)
    
    # Create meshgrid for interpolation targets
    k_mesh, kz_mesh = np.meshgrid(k_grid, kz_grid, indexing='ij')
    
    # Interpolate for each energy slice
    data_out = np.zeros((n_energy, k_points, kz_points))
    
    for i_e in range(n_energy):
        # Flatten (angle, hv) for this energy
        k_flat = k_3d[:, i_e, :].ravel()
        kz_flat = kz_3d[:, i_e, :].ravel()
        intensity_flat = data_values[:, i_e, :].ravel()
        
        # Remove NaN/invalid points
        valid = ~(np.isnan(k_flat) | np.isnan(kz_flat) | np.isnan(intensity_flat))
        if np.sum(valid) < 4:
            # Not enough points for interpolation
            data_out[i_e, :, :] = fill_value
            continue
        
        # 2D interpolation onto regular grid
        data_out[i_e, :, :] = griddata(
            (k_flat[valid], kz_flat[valid]),
            intensity_flat[valid],
            (k_mesh, kz_mesh),
            method='linear',
            fill_value=fill_value
        )
    
    # Create output DataArray
    result = xr.DataArray(
        data_out,
        dims=[energy_dim, 'k', 'kz'],
        coords={
            energy_dim: energies,
            'k': k_grid,
            'kz': kz_grid
        },
        attrs={
            'conversion': 'angle_hv_to_kxkz',
            'phi': phi,
            'V0': V0,
            'interpolated': True,
            'original_angle_range': (angles_deg.min(), angles_deg.max()),
            'original_hv_range': (hvs.min(), hvs.max()),
        }
    )
    
    return result


def convert_angle_to_k(data, hv=None, phi=4.5, V0=12.57, convert_hv_to_kz=False,
                        interpolate=True, k_points=None, fill_value=0.0,
                        is_hv_scan=False, hv_dim=None):
    """
    Unified interface for ARPES angle → momentum conversion.
    
    Automatically detects the type of data (2D or 3D) and applies the 
    appropriate conversion. k is calculated separately for each binding 
    energy to account for the energy-dependent momentum.
    
    Parameters
    ----------
    data : xarray.DataArray
        ARPES data with angle and energy dimensions
        - 2D: (angle, energy) - single k-E cut
        - 3D with hv: (angle, energy, hv) - photon energy scan
        - 3D without hv: (angle_x, angle_y, energy) - kx-ky mapping
    hv : float or None
        Photon energy in eV
        - Required for 2D data
        - Required for 3D kx-ky mapping
        - Not needed if 'hv' is a dimension (will be read from coords)
    phi : float, optional
        Work function in eV, default 4.5 eV
    V0 : float, optional
        Inner potential in eV for kz calculation, default 12.57 eV
    convert_hv_to_kz : bool, optional
        For 3D hv-scan data: if True, calculate kz coordinate
    interpolate : bool, optional
        If True, interpolate data onto a regular k-grid (default True)
        If False, just add k as a coordinate without interpolation
    k_points : int, optional
        Number of k points for interpolation. If None, uses same as angle points.
    fill_value : float, optional
        Value to fill for points outside the interpolation range (default 0.0).
        Use np.nan if you want to preserve NaN for out-of-range values.
    is_hv_scan : bool, optional
        If True, treat 3D data as hv mapping even if the hv dimension 
        is not named 'hv'. Default False.
    hv_dim : str, optional
        Name of the hv dimension if not 'hv' (e.g., 'scan'). 
        Only used when is_hv_scan=True.
        
    Returns
    -------
    xarray.DataArray
        If interpolate=True: Data with 'k' as a proper dimension
        If interpolate=False: Data with 'k' as coordinate (varies with energy)
        
    Examples
    --------
    >>> # 2D angle-energy scan
    >>> result = convert_angle_to_k(data_2d, hv=21.2)
    
    >>> # 3D kx-ky mapping
    >>> result = convert_angle_to_k(data_3d, hv=100.0)
    
    >>> # 3D hv scan with 'hv' dimension
    >>> result = convert_angle_to_k(data_hv, convert_hv_to_kz=True)
    
    >>> # 3D hv scan with 'scan' dimension (need is_hv_scan=True)
    >>> result = convert_angle_to_k(data, is_hv_scan=True, hv_dim='scan')
    """
    dims = data.dims
    
    # Case 1: 2D angle-energy scan
    if data.ndim == 2:
        if hv is None:
            raise ValueError("hv (photon energy) is required for 2D angle-energy conversion")
        return convert_2d_angle_to_k(data, hv=hv, phi=phi, 
                                      interpolate=interpolate, k_points=k_points,
                                      fill_value=fill_value)
    
    # Case 2: 3D data
    if data.ndim == 3:
        # Check if this is an hv scan
        is_hv_data = 'hv' in dims or is_hv_scan
        
        if is_hv_data:
            # Determine the actual hv dimension name
            if 'hv' in dims:
                actual_hv_dim = 'hv'
            elif hv_dim is not None:
                actual_hv_dim = hv_dim
            else:
                # Try to guess - use the dimension that's not 'energy' or 'angle'
                non_ea_dims = [d for d in dims if d not in ['energy', 'angle']]
                if len(non_ea_dims) == 1:
                    actual_hv_dim = non_ea_dims[0]
                else:
                    raise ValueError(
                        "Cannot determine hv dimension. Please specify hv_dim parameter."
                    )
            
            return convert_3d_hv_to_k(data, phi=phi, V0=V0, 
                                       convert_hv_to_kz=convert_hv_to_kz,
                                       hv_dim=actual_hv_dim,
                                       interpolate=interpolate, k_points=k_points,
                                       fill_value=fill_value)
        else:
            # kx-ky-E mapping
            if hv is None:
                raise ValueError("hv (photon energy) is required for 3D kx-ky-E mapping")
            # Try to detect angle dimension names
            angle_dims = [d for d in dims if d != 'energy']
            if len(angle_dims) != 2:
                raise ValueError(f"Expected 2 angle dimensions, got: {angle_dims}")
            return convert_3d_kxky_to_k(data, hv=hv, phi=phi, 
                                         angle_dims=tuple(angle_dims),
                                         interpolate=interpolate, k_points=k_points,
                                         fill_value=fill_value)
    
    raise ValueError(f"Unsupported data dimensionality: {data.ndim}D")
