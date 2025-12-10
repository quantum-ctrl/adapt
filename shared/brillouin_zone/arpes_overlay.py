"""
ARPES Overlay Module for Brillouin Zone Visualization

This module provides utilities for overlaying ARPES experimental data
onto Brillouin Zone plots, supporting both kx-ky energy slices and 
kx-kz hv-mapping trajectories.

Integration with existing code:
------------------------------
Uses functions from processing/angle_to_k.py for momentum calculations.

Usage:
------
    from brillouin_zone import generate_bz
    from brillouin_zone.arpes_overlay import overlay_energy_slice, map_arpes_to_bz
    
    # Map ARPES data to BZ coordinates
    k_coords = map_arpes_to_bz(arpes_data, hv=100, phi=4.5)
    
    # Overlay on BZ plot
    fig, ax = plot_bz(bz)
    overlay_energy_slice(ax, arpes_data, eb=-0.5, hv=100)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any, Union

# Optional xarray import - only needed for ARPES data overlay functions
try:
    import xarray as xr
    _HAS_XARRAY = True
except ImportError:
    _HAS_XARRAY = False
    xr = None  # type: ignore

from .bz_geometry import BrillouinZone


# ARPES constant: k = K_FACTOR * sqrt(Ek) * sin(theta)
# where Ek is kinetic energy in eV, theta in radians, result in Å⁻¹
K_FACTOR = 0.5123  # Å⁻¹ / sqrt(eV)


def angle_to_k(theta: np.ndarray, Ek: float) -> np.ndarray:
    """
    Convert emission angle to parallel momentum.
    
    k∥ = (1/ℏ) * sqrt(2m·Ek) * sin(θ)
       ≈ 0.5123 * sqrt(Ek) * sin(θ)  [Å⁻¹]
    
    Parameters
    ----------
    theta : np.ndarray
        Emission angles in degrees
    Ek : float
        Kinetic energy in eV
        
    Returns
    -------
    np.ndarray
        Parallel momentum in Å⁻¹
    """
    return K_FACTOR * np.sqrt(Ek) * np.sin(np.radians(theta))


def calc_kz(Ek: float, theta: float, V0: float = 10.0) -> float:
    """
    Calculate perpendicular momentum kz using free-electron final state model.
    
    kz = (1/ℏ) * sqrt(2m·(Ek·cos²θ + V0))
       ≈ 0.5123 * sqrt(Ek·cos²θ + V0)  [Å⁻¹]
    
    Parameters
    ----------
    Ek : float
        Kinetic energy in eV
    theta : float
        Emission angle in degrees
    V0 : float
        Inner potential in eV (typically 10-15 eV)
        
    Returns
    -------
    float
        Perpendicular momentum kz in Å⁻¹
    """
    theta_rad = np.radians(theta)
    return K_FACTOR * np.sqrt(Ek * np.cos(theta_rad)**2 + V0)


def map_arpes_to_bz(data: xr.DataArray,
                    hv: float,
                    phi: float = 4.5,
                    V0: float = 10.0,
                    angle_dim: str = 'angle') -> Dict[str, np.ndarray]:
    """
    Convert ARPES data coordinates to Brillouin Zone coordinates.
    
    Parameters
    ----------
    data : xr.DataArray
        ARPES data with angle and energy dimensions
    hv : float
        Photon energy in eV
    phi : float
        Work function in eV
    V0 : float
        Inner potential for kz calculation in eV
    angle_dim : str
        Name of the angle dimension
        
    Returns
    -------
    Dict with:
        'kx' : array of kx values (Å⁻¹)
        'ky' : array (zeros if single angle)
        'kz' : array of kz values (Å⁻¹)
        'energy' : binding energy values (eV)
    """
    # Get angle values
    if angle_dim in data.dims:
        angles = data.coords[angle_dim].values
    else:
        raise ValueError(f"Dimension '{angle_dim}' not found in data")
    
    # Get energy values
    if 'energy' in data.dims:
        energies = data.coords['energy'].values
    elif 'Eb' in data.dims:
        energies = data.coords['Eb'].values
    else:
        energies = np.array([0])
    
    # Calculate kinematic quantities for each energy
    # Energy is typically binding energy (negative for occupied states)
    # Kinetic energy: Ek = hv - phi - |Eb|
    result = {
        'energy': energies,
        'angles': angles,
    }
    
    # Create 2D grids for full k-space mapping
    angle_grid, energy_grid = np.meshgrid(angles, energies, indexing='ij')
    Ek_grid = hv - phi - np.abs(energy_grid)
    Ek_grid = np.maximum(Ek_grid, 0.1)  # Avoid negative/zero kinetic energy
    
    result['kx'] = K_FACTOR * np.sqrt(Ek_grid) * np.sin(np.radians(angle_grid))
    result['kz'] = K_FACTOR * np.sqrt(Ek_grid * np.cos(np.radians(angle_grid))**2 + V0)
    result['ky'] = np.zeros_like(result['kx'])
    
    return result


def create_kxky_plane(bz: BrillouinZone,
                      kz: float = 0.0,
                      n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a kx-ky sampling grid at a fixed kz value (energy slice).
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object
    kz : float
        Fixed kz value for the plane (Å⁻¹)
    n_points : int
        Number of points per axis
        
    Returns
    -------
    kx : np.ndarray
        1D array of kx values
    ky : np.ndarray
        1D array of ky values
    mask : np.ndarray
        2D boolean array indicating points inside the BZ
    """
    bbox_min, bbox_max = bz.get_bounding_box()
    
    kx = np.linspace(bbox_min[0], bbox_max[0], n_points)
    ky = np.linspace(bbox_min[1], bbox_max[1], n_points)
    
    # Create mask for points inside BZ (approximate using bounding box)
    # For exact inside/outside, would need to check against the polyhedron
    KX, KY = np.meshgrid(kx, ky)
    mask = np.ones_like(KX, dtype=bool)
    
    return kx, ky, mask


def create_kxkz_trajectory(bz: BrillouinZone,
                           ky: float = 0.0,
                           n_points: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a kx-kz sampling grid at fixed ky (for hv-mapping visualization).
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object
    ky : float
        Fixed ky value (Å⁻¹)
    n_points : int
        Number of points per axis
        
    Returns
    -------
    kx : np.ndarray
        1D array of kx values
    kz : np.ndarray
        1D array of kz values
    """
    bbox_min, bbox_max = bz.get_bounding_box()
    
    kx = np.linspace(bbox_min[0], bbox_max[0], n_points)
    kz = np.linspace(bbox_min[2], bbox_max[2], n_points)
    
    return kx, kz


def overlay_energy_slice(ax: plt.Axes,
                         data: xr.DataArray,
                         eb: float,
                         hv: float,
                         phi: float = 4.5,
                         angle_dim: str = 'angle',
                         cmap: str = 'hot',
                         alpha: float = 0.7,
                         kz: float = 0.0) -> None:
    """
    Overlay an ARPES energy slice onto a BZ plot.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        3D axes from plot_bz_matplotlib
    data : xr.DataArray
        ARPES data (can be 2D or 3D)
    eb : float
        Binding energy for the slice (eV)
    hv : float
        Photon energy (eV)
    phi : float
        Work function (eV)
    angle_dim : str
        Name of angle dimension
    cmap : str
        Colormap for intensity
    alpha : float
        Transparency
    kz : float
        kz position for 2D data display
    """
    # Select energy slice if 2D or higher
    if 'energy' in data.dims:
        slice_data = data.sel(energy=eb, method='nearest')
    else:
        slice_data = data
    
    # Get angles
    if angle_dim in slice_data.dims:
        angles = slice_data.coords[angle_dim].values
    else:
        return
    
    # Calculate k values
    Ek = hv - phi - np.abs(eb)
    if Ek <= 0:
        return
    
    kx = angle_to_k(angles, Ek)
    
    # For 2D data, plot as line at kz
    if slice_data.ndim == 1:
        ky = np.zeros_like(kx)
        kz_arr = np.full_like(kx, kz)
        
        # Normalize intensities for coloring
        intensities = slice_data.values
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-10)
        
        ax.scatter(kx, ky, kz_arr, c=intensities, cmap=cmap, alpha=alpha, s=10)
    
    # For 2D slice (e.g., from Fermi surface mapping)
    elif slice_data.ndim == 2:
        # Assume second dimension is ky angles
        other_dim = [d for d in slice_data.dims if d != angle_dim][0]
        angles_ky = slice_data.coords[other_dim].values
        ky = angle_to_k(angles_ky, Ek)
        
        # Create meshgrid
        KX, KY = np.meshgrid(kx, ky)
        KZ = np.full_like(KX, kz)
        
        # Scatter plot with intensity coloring
        intensities = slice_data.values.flatten()
        intensities = (intensities - intensities.min()) / (intensities.max() - intensities.min() + 1e-10)
        
        ax.scatter(KX.flatten(), KY.flatten(), KZ.flatten(),
                   c=intensities, cmap=cmap, alpha=alpha, s=5)


def overlay_hv_trajectory(ax: plt.Axes,
                          hv_values: np.ndarray,
                          theta: float = 0.0,
                          phi: float = 4.5,
                          V0: float = 10.0,
                          eb: float = 0.0,
                          color: str = 'blue',
                          linewidth: float = 2.0,
                          label: Optional[str] = None) -> None:
    """
    Overlay an hv-mapping trajectory (kz vs hv) on a BZ plot.
    
    This shows the kz values accessed at different photon energies
    for a fixed emission angle and binding energy.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        3D axes from plot_bz_matplotlib
    hv_values : np.ndarray
        Array of photon energies (eV)
    theta : float
        Emission angle (degrees)
    phi : float
        Work function (eV)
    V0 : float
        Inner potential (eV)
    eb : float
        Binding energy (eV)
    color : str
        Line color
    linewidth : float
        Line width
    label : str, optional
        Legend label
    """
    # Calculate kinetic energy for each hv
    Ek = hv_values - phi - np.abs(eb)
    Ek = np.maximum(Ek, 0.1)
    
    # Calculate kx (parallel) and kz (perpendicular)
    kx = K_FACTOR * np.sqrt(Ek) * np.sin(np.radians(theta))
    kz = K_FACTOR * np.sqrt(Ek * np.cos(np.radians(theta))**2 + V0)
    ky = np.zeros_like(kx)
    
    ax.plot(kx, ky, kz, color=color, linewidth=linewidth, label=label)
    
    if label:
        ax.legend()


def _normalize_direction(direction: Union[str, np.ndarray, list]) -> np.ndarray:
    """
    Convert direction specification to a normalized 3D vector.
    
    Parameters
    ----------
    direction : str or array-like
        Direction specification: 'kx', 'ky', 'kz', or a 3D vector [vx, vy, vz]
        
    Returns
    -------
    np.ndarray
        Normalized 3D direction vector
    """
    if isinstance(direction, str):
        direction_map = {
            'kx': np.array([1.0, 0.0, 0.0]),
            'ky': np.array([0.0, 1.0, 0.0]),
            'kz': np.array([0.0, 0.0, 1.0]),
            'x': np.array([1.0, 0.0, 0.0]),
            'y': np.array([0.0, 1.0, 0.0]),
            'z': np.array([0.0, 0.0, 1.0]),
        }
        if direction.lower() not in direction_map:
            raise ValueError(f"Unknown direction '{direction}'. Use 'kx', 'ky', 'kz' or a 3D vector.")
        return direction_map[direction.lower()]
    else:
        vec = np.array(direction, dtype=float)
        if vec.shape != (3,):
            raise ValueError(f"Direction must be a 3D vector, got shape {vec.shape}")
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            raise ValueError("Direction vector cannot be zero")
        return vec / norm


def get_bz_slice(bz: BrillouinZone,
                 horizontal_dir: Union[str, np.ndarray, list] = 'kx',
                 vertical_dir: Union[str, np.ndarray, list] = 'ky',
                 slice_value: float = 0.0,
                 offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    """
    Get the 2D cross-section of the BZ at an arbitrary slice plane.
    
    The slice plane is defined by two in-plane directions (horizontal and vertical)
    and a fixed value along the perpendicular (normal) direction.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object
    horizontal_dir : str or array-like
        Direction for horizontal axis: 'kx', 'ky', 'kz', or a 3D vector [vx, vy, vz].
        Examples: 'kx', [1, 1, 0] (45° in kx-ky plane)
    vertical_dir : str or array-like
        Direction for vertical axis: 'kx', 'ky', 'kz', or a 3D vector.
    slice_value : float
        Fixed value along the normal direction (perpendicular to the plane).
        For example, if horizontal_dir='kx' and vertical_dir='kz', normal is 'ky',
        and slice_value is the fixed ky value.
    offset : tuple
        Offset to apply to the BZ vertices before slicing, useful for plotting
        multiple BZ copies at different reciprocal lattice positions.
        
    Returns
    -------
    np.ndarray
        2D polygon vertices of the BZ slice in (horizontal, vertical) coordinates,
        shape (N, 2). Returns empty array if slice doesn't intersect BZ.
    """
    from scipy.spatial import ConvexHull
    
    # Normalize directions
    h_vec = _normalize_direction(horizontal_dir)
    v_vec = _normalize_direction(vertical_dir)
    
    # Compute normal direction (perpendicular to the slice plane)
    normal = np.cross(h_vec, v_vec)
    norm_mag = np.linalg.norm(normal)
    if norm_mag < 1e-10:
        raise ValueError("horizontal_dir and vertical_dir must not be parallel")
    normal = normal / norm_mag
    
    # Apply offset to vertices
    offset = np.array(offset)
    vertices = bz.vertices + offset
    
    # Find intersection points with the slice plane: normal · (r - r0) = 0
    # where r0 is a point on the plane: r0 = slice_value * normal
    intersection_points = []
    
    for face in bz.faces:
        v0, v1, v2 = vertices[face]
        
        # Check each edge of the triangle
        edges = [(v0, v1), (v1, v2), (v2, v0)]
        
        for p1, p2 in edges:
            # Project onto normal direction
            d1 = np.dot(p1, normal) - slice_value
            d2 = np.dot(p2, normal) - slice_value
            
            # Check if edge crosses the slice plane
            if d1 * d2 < 0:
                # Linear interpolation to find intersection point
                t = d1 / (d1 - d2)
                intersection_3d = p1 + t * (p2 - p1)
                
                # Project onto 2D plane coordinates
                h_coord = np.dot(intersection_3d, h_vec)
                v_coord = np.dot(intersection_3d, v_vec)
                intersection_points.append([h_coord, v_coord])
    
    if len(intersection_points) < 3:
        return np.array([])
    
    # Remove duplicate points (edges shared between triangles produce duplicates)
    points = np.array(intersection_points)
    
    # Use a tolerance-based deduplication
    unique_points = []
    tol = 1e-6
    for pt in points:
        is_duplicate = False
        for upt in unique_points:
            if np.linalg.norm(pt - upt) < tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(pt)
    
    points = np.array(unique_points)
    
    if len(points) < 3:
        return np.array([])
    
    # Order points to form a polygon (convex hull)
    try:
        hull = ConvexHull(points)
        return points[hull.vertices]
    except:
        return points


def get_bz_slice_at_kz(bz: BrillouinZone, kz: float) -> np.ndarray:
    """
    Get the 2D cross-section of the BZ at a given kz value (kx-ky plane).
    
    This is a convenience wrapper around get_bz_slice for the common case
    of slicing at fixed kz.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object
    kz : float
        kz value for the slice (Å⁻¹)
        
    Returns
    -------
    np.ndarray
        2D polygon vertices of the BZ slice, shape (N, 2)
        Returns empty array if kz is outside BZ bounds.
    """
    return get_bz_slice(bz, horizontal_dir='kx', vertical_dir='ky', slice_value=kz)


def plot_bz_slice_2d(bz: BrillouinZone,
                     slice_value: float = 0.0,
                     horizontal_dir: Union[str, np.ndarray, list] = 'kx',
                     vertical_dir: Union[str, np.ndarray, list] = 'ky',
                     ax: Optional[plt.Axes] = None,
                     figsize: Tuple[int, int] = (8, 8),
                     facecolor: str = 'lightblue',
                     edgecolor: str = 'black',
                     alpha: float = 0.3,
                     show_hs_points: bool = True,
                     rotation: float = 0.0,
                     horizontal_range: Optional[Tuple[float, float]] = None,
                     vertical_range: Optional[Tuple[float, float]] = None,
                     # Legacy parameter for backward compatibility
                     kz: Optional[float] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a 2D slice of the Brillouin Zone at an arbitrary plane.
    
    Supports flexible slice plane definition using direction vectors, enabling:
    - kx-ky plane at fixed kz (default, traditional usage)
    - kx-kz plane at fixed ky (for hv-mapping overlay)
    - Custom planes like 45° in kx-ky vs kz
    
    Also supports extended k-range to display multiple BZ copies.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object
    slice_value : float
        Fixed value along the normal direction (perpendicular to the slice plane).
        For kx-ky plane, this is kz. For kx-kz plane, this is ky.
    horizontal_dir : str or array-like
        Direction for horizontal axis: 'kx', 'ky', 'kz', or a 3D vector.
        Examples: 'kx', [1, 1, 0] (45° in kx-ky plane)
    vertical_dir : str or array-like
        Direction for vertical axis: 'kx', 'ky', 'kz', or a 3D vector.
    ax : matplotlib.axes.Axes, optional
        Existing axes. If None, creates new figure.
    figsize : tuple
        Figure size
    facecolor : str
        Fill color for BZ polygons
    edgecolor : str
        Border color for BZ polygons
    alpha : float
        Transparency
    show_hs_points : bool
        Whether to show high-symmetry points in the slice plane
    rotation : float
        Rotation angle in degrees (counterclockwise) to apply to the slice.
    horizontal_range : tuple (min, max), optional
        Range for horizontal axis. If provided, plots multiple BZ copies
        to cover this range. Example: (-2, 2) for kx from -2 to 2 Å⁻¹
    vertical_range : tuple (min, max), optional
        Range for vertical axis. If provided, plots multiple BZ copies
        to cover this range. Example: (10, 15) for kz from 10 to 15 Å⁻¹
    kz : float, optional
        **Deprecated**: Use slice_value instead. Provided for backward compatibility
        when using the default kx-ky plane.
        
    Returns
    -------
    fig, ax
    
    Examples
    --------
    # Traditional kx-ky slice at kz=0
    >>> fig, ax = plot_bz_slice_2d(bz, slice_value=0)
    
    # kx-kz slice at ky=0
    >>> fig, ax = plot_bz_slice_2d(bz, slice_value=0, 
    ...                            horizontal_dir='kx', vertical_dir='kz')
    
    # Extended kx-kz range: kx from -2 to 2, kz from 10 to 15
    >>> fig, ax = plot_bz_slice_2d(bz, slice_value=0,
    ...                            horizontal_dir='kx', vertical_dir='kz',
    ...                            horizontal_range=(-2, 2), vertical_range=(10, 15))
    
    # 45° direction in kx-ky plane vs kz
    >>> fig, ax = plot_bz_slice_2d(bz, slice_value=0,
    ...                            horizontal_dir=[1, 1, 0], vertical_dir='kz')
    """
    # Handle legacy kz parameter
    if kz is not None:
        slice_value = kz
    
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()
    
    # Normalize directions
    h_vec = _normalize_direction(horizontal_dir)
    v_vec = _normalize_direction(vertical_dir)
    
    # Compute normal direction
    normal = np.cross(h_vec, v_vec)
    norm_mag = np.linalg.norm(normal)
    if norm_mag < 1e-10:
        raise ValueError("horizontal_dir and vertical_dir must not be parallel")
    normal = normal / norm_mag
    
    # Create rotation matrix for 2D
    theta_rad = np.radians(rotation)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    rot_matrix = np.array([[cos_t, -sin_t],
                           [sin_t, cos_t]])
    
    # Determine which BZ copies to plot
    # Get reciprocal lattice vectors
    rec_basis = bz.reciprocal_basis  # Should be (3, 3) array with b1, b2, b3 as rows
    
    # Project reciprocal vectors onto the slice plane
    rec_h = np.array([np.dot(rec_basis[i], h_vec) for i in range(3)])
    rec_v = np.array([np.dot(rec_basis[i], v_vec) for i in range(3)])
    rec_n = np.array([np.dot(rec_basis[i], normal) for i in range(3)])
    
    # Determine the range of reciprocal lattice indices needed
    if horizontal_range is None and vertical_range is None:
        # Just plot the first BZ centered at origin
        offsets_3d = [(0.0, 0.0, 0.0)]
    else:
        # Calculate which BZ copies are needed to cover the range
        bbox_min, bbox_max = bz.get_bounding_box()
        
        # Project BZ bounding box onto slice plane
        bz_h_min = np.dot(bbox_min, h_vec)
        bz_h_max = np.dot(bbox_max, h_vec)
        bz_v_min = np.dot(bbox_min, v_vec)
        bz_v_max = np.dot(bbox_max, v_vec)
        bz_h_size = bz_h_max - bz_h_min
        bz_v_size = bz_v_max - bz_v_min
        
        h_min, h_max = horizontal_range if horizontal_range else (bz_h_min, bz_h_max)
        v_min, v_max = vertical_range if vertical_range else (bz_v_min, bz_v_max)
        
        # Calculate the range of reciprocal lattice indices needed
        # based on the requested k-space range
        rec_vec_norms = np.array([np.linalg.norm(rec_basis[i]) for i in range(3)])
        
        # Estimate max index needed for each reciprocal vector
        max_k_range = max(abs(h_min), abs(h_max), abs(v_min), abs(v_max))
        max_index = int(np.ceil(max_k_range / rec_vec_norms.min())) + 2
        
        # Find indices for periodic copies
        offsets_3d = []
        # Search a dynamically determined range of reciprocal lattice indices
        for i in range(-max_index, max_index + 1):
            for j in range(-max_index, max_index + 1):
                for k in range(-max_index, max_index + 1):
                    # Compute the offset in 3D
                    offset_3d = i * rec_basis[0] + j * rec_basis[1] + k * rec_basis[2]
                    
                    # Project offset onto slice plane
                    offset_h = np.dot(offset_3d, h_vec)
                    offset_v = np.dot(offset_3d, v_vec)
                    offset_n = np.dot(offset_3d, normal)
                    
                    # Check if this copy would be visible in the requested range
                    # The BZ centered at this offset should overlap with the view range
                    bz_center_h = offset_h
                    bz_center_v = offset_v
                    
                    # Also need to check if the slice plane value matches
                    # (offset along normal should match the slice_value within BZ extent)
                    bz_n_size = np.dot(bbox_max - bbox_min, np.abs(normal))
                    if abs(offset_n - slice_value) > bz_n_size:
                        continue
                    
                    # Check if BZ at this offset overlaps with view range
                    if (bz_center_h + bz_h_max >= h_min and bz_center_h + bz_h_min <= h_max and
                        bz_center_v + bz_v_max >= v_min and bz_center_v + bz_v_min <= v_max):
                        offsets_3d.append(tuple(offset_3d))
    
    # Plot each BZ copy
    all_hs_points = []  # Collect high-symmetry points for all copies
    
    for offset in offsets_3d:
        offset_arr = np.array(offset)
        
        # Get slice for this BZ copy, adjusting slice_value for the offset
        offset_n = np.dot(offset_arr, normal)
        adjusted_slice_value = slice_value - offset_n
        
        slice_polygon = get_bz_slice(bz, horizontal_dir, vertical_dir, 
                                      adjusted_slice_value, offset=offset)
        
        if len(slice_polygon) > 0:
            # Apply rotation to polygon
            rotated_polygon = slice_polygon @ rot_matrix.T
            # Close the polygon
            polygon = np.vstack([rotated_polygon, rotated_polygon[0]])
            ax.fill(polygon[:, 0], polygon[:, 1], 
                    facecolor=facecolor, edgecolor=edgecolor, 
                    alpha=alpha, linewidth=1.5, zorder=1)
        
        # Collect high-symmetry points
        if show_hs_points:
            for name, point in bz.high_symmetry_points.items():
                offset_point = point + offset_arr
                # Check if point is near the slice plane
                point_n = np.dot(offset_point, normal)
                tol = 0.1 * np.linalg.norm(bz.reciprocal_basis[0])  # 10% of first rec vector
                if abs(point_n - slice_value) < tol:
                    point_h = np.dot(offset_point, h_vec)
                    point_v = np.dot(offset_point, v_vec)
                    all_hs_points.append((name, point_h, point_v))
    
    # Plot high-symmetry points
    if show_hs_points and all_hs_points:
        for name, h_coord, v_coord in all_hs_points:
            # Apply rotation
            rotated_point = rot_matrix @ np.array([h_coord, v_coord])
            
            # Check if within view range
            if horizontal_range:
                if rotated_point[0] < horizontal_range[0] or rotated_point[0] > horizontal_range[1]:
                    continue
            if vertical_range:
                if rotated_point[1] < vertical_range[0] or rotated_point[1] > vertical_range[1]:
                    continue
            
            ax.plot(rotated_point[0], rotated_point[1], 'ro', markersize=6, zorder=2)
            ax.annotate('Γ' if name == 'Gamma' else name,
                        (rotated_point[0], rotated_point[1]),
                        xytext=(3, 3), textcoords='offset points',
                        fontsize=9, fontweight='bold', color='darkred', zorder=3)
    
    # Set axis labels based on direction
    def get_axis_label(direction):
        if isinstance(direction, str):
            if direction.lower() in ['kx', 'x']:
                return '$k_x$ (Å$^{-1}$)'
            elif direction.lower() in ['ky', 'y']:
                return '$k_y$ (Å$^{-1}$)'
            elif direction.lower() in ['kz', 'z']:
                return '$k_z$ (Å$^{-1}$)'
        return '$k$ (Å$^{-1}$)'
    
    ax.set_xlabel(get_axis_label(horizontal_dir), fontsize=12)
    ax.set_ylabel(get_axis_label(vertical_dir), fontsize=12)
    
    # Generate title
    def get_normal_name(h_dir, v_dir):
        h_str = h_dir if isinstance(h_dir, str) else f'[{h_dir[0]:.1f},{h_dir[1]:.1f},{h_dir[2]:.1f}]'
        v_str = v_dir if isinstance(v_dir, str) else f'[{v_dir[0]:.1f},{v_dir[1]:.1f},{v_dir[2]:.1f}]'
        
        # Determine fixed axis
        if isinstance(h_dir, str) and isinstance(v_dir, str):
            h_low, v_low = h_dir.lower(), v_dir.lower()
            if set([h_low, v_low]) == {'kx', 'ky'} or set([h_low, v_low]) == {'x', 'y'}:
                return f'$k_z$ = {slice_value:.2f}'
            elif set([h_low, v_low]) == {'kx', 'kz'} or set([h_low, v_low]) == {'x', 'z'}:
                return f'$k_y$ = {slice_value:.2f}'
            elif set([h_low, v_low]) == {'ky', 'kz'} or set([h_low, v_low]) == {'y', 'z'}:
                return f'$k_x$ = {slice_value:.2f}'
        return f'slice = {slice_value:.2f}'
    
    ax.set_title(f'BZ slice at {get_normal_name(horizontal_dir, vertical_dir)} Å$^{{-1}}$', fontsize=14)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Set axis limits if ranges provided
    if horizontal_range:
        ax.set_xlim(horizontal_range)
    if vertical_range:
        ax.set_ylim(vertical_range)
    
    return fig, ax


def _normalize_cleavage_plane(cleavage_plane: Union[str, np.ndarray, list, Tuple]) -> np.ndarray:
    """
    Convert cleavage plane specification to a normalized surface normal vector.
    
    Parameters
    ----------
    cleavage_plane : str, array-like, or tuple
        Cleavage plane specification:
        - Miller indices as string: '001', '110', '111'
        - Miller indices as tuple or list: (0, 0, 1), [1, 1, 0]
        - Direct normal vector: [nx, ny, nz]
        
    Returns
    -------
    np.ndarray
        Normalized surface normal vector (3D)
    """
    if isinstance(cleavage_plane, str):
        # Parse Miller indices from string like '001', '110', '111'
        cleavage_plane = cleavage_plane.strip()
        if len(cleavage_plane) == 3:
            try:
                h = int(cleavage_plane[0])
                k = int(cleavage_plane[1])
                l = int(cleavage_plane[2])
                vec = np.array([float(h), float(k), float(l)])
            except ValueError:
                raise ValueError(f"Cannot parse cleavage plane string '{cleavage_plane}'")
        else:
            raise ValueError(f"Invalid cleavage plane string '{cleavage_plane}'. Expected 3 digits (e.g., '001', '110', '111')")
    else:
        vec = np.array(cleavage_plane, dtype=float)
        if vec.shape != (3,):
            raise ValueError(f"Cleavage plane must be a 3D vector, got shape {vec.shape}")
    
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        raise ValueError("Cleavage plane normal vector cannot be zero")
    return vec / norm


def calc_arpes_hemisphere(hv: float, 
                          phi: float = 4.5, 
                          V0: float = 10.0, 
                          eb: float = 0.0,
                          theta_range: Tuple[float, float] = (-60, 60),
                          azimuth_range: Tuple[float, float] = (0, 360),
                          n_theta: int = 50,
                          n_azimuth: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the accessible k-space hemisphere (arc surface) for a given photon energy.
    
    In ARPES, for a given hv and binding energy, the emission angle θ traces out
    a hemisphere in k-space. The kz value depends on θ through the free-electron
    final state model.
    
    Parameters
    ----------
    hv : float
        Photon energy in eV
    phi : float
        Work function in eV
    V0 : float
        Inner potential in eV
    eb : float
        Binding energy in eV (negative for occupied states)
    theta_range : tuple (min, max)
        Range of emission angles in degrees
    azimuth_range : tuple (min, max)
        Range of azimuthal angles in degrees (0-360 for full hemisphere)
    n_theta : int
        Number of points along theta direction
    n_azimuth : int
        Number of points along azimuth direction
        
    Returns
    -------
    kx, ky, kz : np.ndarray
        2D arrays of k-space coordinates forming the arc surface mesh
        Each has shape (n_theta, n_azimuth)
    """
    # Calculate kinetic energy
    Ek = hv - phi - np.abs(eb)
    if Ek <= 0:
        raise ValueError(f"Kinetic energy ({Ek:.2f} eV) must be positive. Check hv, phi, and eb values.")
    
    # Create angle grids
    theta = np.linspace(theta_range[0], theta_range[1], n_theta)
    azimuth = np.linspace(azimuth_range[0], azimuth_range[1], n_azimuth)
    THETA, AZIMUTH = np.meshgrid(theta, azimuth, indexing='ij')
    
    # Convert to radians
    theta_rad = np.radians(THETA)
    azimuth_rad = np.radians(AZIMUTH)
    
    # Calculate k-space coordinates
    # k_parallel = K_FACTOR * sqrt(Ek) * sin(theta)
    # kz = K_FACTOR * sqrt(Ek * cos²(theta) + V0)
    k_parallel = K_FACTOR * np.sqrt(Ek) * np.sin(np.abs(theta_rad))
    kz = K_FACTOR * np.sqrt(Ek * np.cos(theta_rad)**2 + V0)
    
    # Project k_parallel onto kx, ky using azimuthal angle
    kx = k_parallel * np.cos(azimuth_rad) * np.sign(np.sin(theta_rad))
    ky = k_parallel * np.sin(azimuth_rad) * np.sign(np.sin(theta_rad))
    
    return kx, ky, kz


def plot_bz_with_arpes_arc(bz: BrillouinZone,
                           hv: float,
                           phi: float = 4.5,
                           V0: float = 10.0,
                           eb: float = 0.0,
                           ax: Optional[plt.Axes] = None,
                           figsize: Tuple[int, int] = (10, 10),
                           bz_facecolor: Optional[str] = 'cyan',
                           bz_edgecolor: str = 'grey',
                           bz_alpha: float = 0.2,
                           bz_fill: bool = True,
                           arc_color: str = 'red',
                           arc_alpha: float = 0.5,
                           theta_range: Tuple[float, float] = (-60, 60),
                           azimuth_range: Tuple[float, float] = (0, 360),
                           show_hs_points: bool = True,
                           bz_range_x: Tuple[int, int] = (0, 1),
                           bz_range_y: Tuple[int, int] = (0, 1),
                           bz_range_z: Tuple[int, int] = (0, 1),
                           n_bz_copies: Optional[int] = None,
                           equal_aspect: bool = True,
                           title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot 3D Brillouin Zone with the ARPES measurement arc surface overlaid.
    
    The arc surface represents all k-space points accessible at the given
    photon energy hv, as the emission angle θ is varied.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object
    hv : float
        Photon energy in eV
    phi : float
        Work function in eV
    V0 : float
        Inner potential in eV
    eb : float
        Binding energy in eV
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes. If None, creates new figure.
    figsize : tuple
        Figure size
    bz_facecolor : str or None
        BZ face color. Set to None or use bz_fill=False for wireframe.
    bz_edgecolor : str
        BZ edge color (default 'grey')
    bz_alpha : float
        BZ transparency
    bz_fill : bool
        Whether to fill BZ faces. If False, shows only wireframe edges.
    arc_color : str
        Color for the arc surface
    arc_alpha : float
        Arc surface transparency
    theta_range : tuple
        Emission angle range in degrees
    azimuth_range : tuple
        Azimuthal angle range in degrees
    show_hs_points : bool
        Whether to show high-symmetry points
    bz_range_x : tuple (min, max)
        Range of BZ indices along kx direction, e.g. (-2, 2) shows BZ -2 to 1
    bz_range_y : tuple (min, max)
        Range of BZ indices along ky direction
    bz_range_z : tuple (min, max)
        Range of BZ indices along kz direction, e.g. (5, 15) shows BZ 5 to 14
    n_bz_copies : int, optional
        **Deprecated**: Use bz_range_z instead. If set, overrides bz_range_z.
    equal_aspect : bool
        If True, set equal aspect ratio for all axes
    title : str, optional
        Plot title
        
    Returns
    -------
    fig, ax
    
    Examples
    --------
    >>> # Show BZ from kx(-2,2), ky(-2,2), kz(5,15)
    >>> fig, ax = plot_bz_with_arpes_arc(
    ...     bz, hv=50,
    ...     bz_range_x=(-2, 2), bz_range_y=(-2, 2), bz_range_z=(5, 15),
    ...     bz_fill=False, bz_edgecolor='grey'
    ... )
    """
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    
    # Handle legacy n_bz_copies parameter
    if n_bz_copies is not None:
        bz_range_z = (0, n_bz_copies)
    
    # Create figure if needed
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
    
    # Get reciprocal lattice vectors for translations
    rec_basis = bz.reciprocal_basis
    a_star = np.linalg.norm(rec_basis[0])  # |a*| for kx translation
    b_star = np.linalg.norm(rec_basis[1])  # |b*| for ky translation
    c_star = np.linalg.norm(rec_basis[2])  # |c*| for kz translation
    
    # Plot multiple BZ copies
    for ix in range(bz_range_x[0], bz_range_x[1]):
        for iy in range(bz_range_y[0], bz_range_y[1]):
            for iz in range(bz_range_z[0], bz_range_z[1]):
                # Offset for this copy
                offset = np.array([ix * a_star, iy * b_star, iz * c_star])
                
                # Create shifted vertices for this BZ copy
                shifted_vertices = bz.vertices + offset
                
                # Plot the BZ copy
                faces_3d = []
                for face in bz.faces:
                    triangle = shifted_vertices[face]
                    faces_3d.append(triangle)
                
                if bz_fill:
                    mesh = Poly3DCollection(faces_3d, alpha=bz_alpha,
                                            facecolor=bz_facecolor, edgecolor=bz_edgecolor,
                                            linewidth=0.5)
                    ax.add_collection3d(mesh)
                else:
                    # Wireframe mode: use Line3DCollection with only boundary edges
                    # In a triangulated convex polyhedron, true edges appear exactly twice
                    # (shared by exactly 2 triangular faces). Internal triangulation 
                    # diagonals would also appear twice, but we can filter by edge length
                    # or just accept that for BZ this is the best we can do.
                    from mpl_toolkits.mplot3d.art3d import Line3DCollection
                    from collections import Counter
                    
                    edge_count = Counter()
                    for face in bz.faces:
                        v0, v1, v2 = face
                        edge_count[tuple(sorted([v0, v1]))] += 1
                        edge_count[tuple(sorted([v1, v2]))] += 1
                        edge_count[tuple(sorted([v2, v0]))] += 1
                    
                    # Draw edges that appear exactly twice (shared by 2 triangles = true polyhedron edges)
                    edge_lines = []
                    for (e0, e1), count in edge_count.items():
                        if count == 2:  # True BZ boundary edges appear exactly twice
                            p0 = shifted_vertices[e0]
                            p1 = shifted_vertices[e1]
                            edge_lines.append([p0, p1])
                    
                    lc = Line3DCollection(edge_lines, colors=bz_edgecolor, linewidths=0.8)
                    ax.add_collection3d(lc)
                
                # Plot high-symmetry points for this copy
                if show_hs_points:
                    for name, point in bz.high_symmetry_points.items():
                        shifted_point = point + offset
                        ax.scatter(*shifted_point, c='red', s=30, zorder=3)
                        display_name = 'Γ' if name == 'Gamma' else name
                        ax.text(shifted_point[0], shifted_point[1], shifted_point[2],
                               f'  {display_name}', fontsize=7, color='darkred')
    
    # Calculate and plot arc surface
    try:
        kx, ky, kz = calc_arpes_hemisphere(hv, phi, V0, eb, theta_range, azimuth_range)
        
        # Plot as surface mesh
        ax.plot_surface(kx, ky, kz, color=arc_color, alpha=arc_alpha, 
                       linewidth=0, antialiased=True)
        
        # Also plot the outline at theta=0 (normal emission line)
        Ek = hv - phi - np.abs(eb)
        kz_ne = K_FACTOR * np.sqrt(Ek + V0)
        ax.scatter([0], [0], [kz_ne], c='yellow', s=100, marker='*', 
                  label=f'Normal emission (kz={kz_ne:.2f})', zorder=5)
        
    except ValueError as e:
        print(f"Warning: Cannot plot arc surface - {e}")
    
    # Set axis labels
    ax.set_xlabel('$k_x$ (Å$^{-1}$)')
    ax.set_ylabel('$k_y$ (Å$^{-1}$)')
    ax.set_zlabel('$k_z$ (Å$^{-1}$)')
    
    # Set equal aspect ratio
    if equal_aspect:
        # Get axis limits
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        
        # Calculate ranges and find max
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)
        
        # Set equal limits
        x_mid = np.mean(x_limits)
        y_mid = np.mean(y_limits)
        z_mid = np.mean(z_limits)
        ax.set_xlim3d([x_mid - max_range/2, x_mid + max_range/2])
        ax.set_ylim3d([y_mid - max_range/2, y_mid + max_range/2])
        ax.set_zlim3d([z_mid - max_range/2, z_mid + max_range/2])
    
    # Set title
    if title:
        ax.set_title(title)
    else:
        Ek = hv - phi - np.abs(eb)
        ax.set_title(f'BZ with ARPES arc surface\nhv={hv} eV, Ek={Ek:.1f} eV, V0={V0} eV', fontsize=12)
    
    ax.legend()
    return fig, ax



def calc_hv_for_kpoint(kx: float, ky: float, kz: float,
                       phi: float = 4.5,
                       V0: float = 10.0,
                       eb: float = 0.0,
                       cleavage_plane: Union[str, np.ndarray, list, Tuple] = '001') -> float:
    """
    Calculate the photon energy required to reach a specific k-space point.
    
    Uses the free-electron final state model to solve for hv given the target
    kx, ky, kz coordinates.
    
    For a (001) surface:
    - k∥ = sqrt(kx² + ky²) = K_FACTOR * sqrt(Ek) * sin(θ)
    - kz = K_FACTOR * sqrt(Ek * cos²(θ) + V0)
    
    Solving these equations:
    - From k∥² + kz² = K_FACTOR² * (Ek + V0)
    - Ek = (k∥² + kz²)/K_FACTOR² - V0
    - hv = Ek + phi + |eb|
    
    Parameters
    ----------
    kx, ky, kz : float
        Target k-space coordinates in Å⁻¹
    phi : float
        Work function in eV
    V0 : float
        Inner potential in eV
    eb : float
        Binding energy in eV
    cleavage_plane : str, array-like, or tuple
        Cleavage plane specification (e.g., '001', '110', '111' or vector)
        
    Returns
    -------
    float
        Required photon energy in eV
        
    Raises
    ------
    ValueError
        If the k-point is not physically accessible (Ek would be negative)
    """
    # Normalize cleavage plane
    normal = _normalize_cleavage_plane(cleavage_plane)
    
    # For general cleavage planes, we need to rotate k-space
    # For simplicity, we'll implement the (001) case explicitly
    # and use rotation for others
    
    if np.allclose(normal, [0, 0, 1]):
        # (001) surface - standard case
        k_parallel = np.sqrt(kx**2 + ky**2)
        kz_perp = kz
    elif np.allclose(normal, [1, 1, 0] / np.sqrt(2)):
        # (110) surface
        # k_parallel is in the plane perpendicular to [110]
        # Need to project properly
        k_parallel = np.sqrt((kx - ky)**2 / 2 + kz**2)
        kz_perp = (kx + ky) / np.sqrt(2)
    elif np.allclose(normal, [1, 1, 1] / np.sqrt(3)):
        # (111) surface
        k_parallel = np.sqrt((kx - ky)**2 / 2 + (kx + ky - 2*kz)**2 / 6)
        kz_perp = (kx + ky + kz) / np.sqrt(3)
    else:
        # General case: rotate to put normal along z
        # Build rotation matrix
        z_axis = np.array([0, 0, 1])
        if np.allclose(normal, z_axis):
            k_parallel = np.sqrt(kx**2 + ky**2)
            kz_perp = kz
        else:
            # Rodrigues rotation
            v = np.cross(normal, z_axis)
            s = np.linalg.norm(v)
            c = np.dot(normal, z_axis)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2 + 1e-10)
            k_rotated = R @ np.array([kx, ky, kz])
            k_parallel = np.sqrt(k_rotated[0]**2 + k_rotated[1]**2)
            kz_perp = k_rotated[2]
    
    # Solve for kinetic energy
    # k∥² + kz² = K_FACTOR² * (Ek + V0)
    # Ek = (k∥² + kz²)/K_FACTOR² - V0
    Ek = (k_parallel**2 + kz_perp**2) / K_FACTOR**2 - V0
    
    if Ek < 0:
        raise ValueError(f"k-point ({kx:.3f}, {ky:.3f}, {kz:.3f}) is not accessible. "
                        f"Calculated Ek={Ek:.2f} eV is negative. "
                        f"This point may be inside the inner potential sphere.")
    
    # Calculate photon energy
    hv = Ek + phi + np.abs(eb)
    
    return hv


def calc_emission_direction(kx: float, ky: float, kz: float,
                            hv: float,
                            phi: float = 4.5,
                            V0: float = 10.0,
                            eb: float = 0.0,
                            cleavage_plane: Union[str, np.ndarray, list, Tuple] = '001') -> Tuple[float, float]:
    """
    Calculate the emission angles (theta, phi_azimuth) needed to reach a k-point.
    
    Given a target k-space point and photon energy, calculates the required
    emission angle θ (polar) and φ (azimuthal) for the measurement.
    
    Parameters
    ----------
    kx, ky, kz : float
        Target k-space coordinates in Å⁻¹
    hv : float
        Photon energy in eV
    phi : float
        Work function in eV
    V0 : float
        Inner potential in eV
    eb : float
        Binding energy in eV
    cleavage_plane : str, array-like, or tuple
        Cleavage plane specification
        
    Returns
    -------
    theta : float
        Emission angle in degrees (polar angle from surface normal)
    phi_azimuth : float
        Azimuthal angle in degrees (in-plane rotation)
        
    Raises
    ------
    ValueError
        If the k-point is not accessible with the given hv
    """
    # Normalize cleavage plane
    normal = _normalize_cleavage_plane(cleavage_plane)
    
    # Calculate kinetic energy
    Ek = hv - phi - np.abs(eb)
    if Ek <= 0:
        raise ValueError(f"Kinetic energy ({Ek:.2f} eV) must be positive")
    
    # For (001) surface, straightforward calculation
    if np.allclose(normal, [0, 0, 1]):
        k_parallel = np.sqrt(kx**2 + ky**2)
        
        # k∥ = K_FACTOR * sqrt(Ek) * sin(θ)
        sin_theta = k_parallel / (K_FACTOR * np.sqrt(Ek))
        
        if abs(sin_theta) > 1:
            raise ValueError(f"k-point not accessible with hv={hv} eV. "
                           f"Required sin(θ)={sin_theta:.3f} > 1")
        
        theta = np.degrees(np.arcsin(sin_theta))
        
        # Azimuthal angle from kx, ky
        if k_parallel < 1e-10:
            phi_azimuth = 0.0
        else:
            phi_azimuth = np.degrees(np.arctan2(ky, kx))
    else:
        # General case: rotate k-point to surface coordinates
        z_axis = np.array([0, 0, 1])
        if np.allclose(normal, z_axis):
            k_rot = np.array([kx, ky, kz])
        else:
            v = np.cross(normal, z_axis)
            s = np.linalg.norm(v)
            c = np.dot(normal, z_axis)
            vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2 + 1e-10)
            k_rot = R @ np.array([kx, ky, kz])
        
        k_parallel = np.sqrt(k_rot[0]**2 + k_rot[1]**2)
        sin_theta = k_parallel / (K_FACTOR * np.sqrt(Ek))
        
        if abs(sin_theta) > 1:
            raise ValueError(f"k-point not accessible with hv={hv} eV")
        
        theta = np.degrees(np.arcsin(sin_theta))
        
        if k_parallel < 1e-10:
            phi_azimuth = 0.0
        else:
            phi_azimuth = np.degrees(np.arctan2(k_rot[1], k_rot[0]))
    
    return theta, phi_azimuth


# Convenience alias for calc_emission_direction
calc_direction_for_kpoint = calc_emission_direction


# Public API
__all__ = [
    'map_arpes_to_bz',
    'create_kxky_plane',
    'create_kxkz_trajectory',
    'overlay_energy_slice',
    'overlay_hv_trajectory',
    'get_bz_slice',
    'get_bz_slice_at_kz',
    'plot_bz_slice_2d',
    'angle_to_k',
    'calc_kz',
    'K_FACTOR',
    # ARPES hv mapping features
    'calc_arpes_hemisphere',
    'plot_bz_with_arpes_arc',
    'calc_hv_for_kpoint',
    'calc_emission_direction',
    'calc_direction_for_kpoint',
]
