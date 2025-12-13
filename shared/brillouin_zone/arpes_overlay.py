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

# Import ARPES constant from centralized location
try:
    from shared.utils.constants import K_FACTOR, DEFAULT_V0, DEFAULT_WORK_FUNCTION
except ImportError:
    # Fallback for standalone usage
    K_FACTOR = 0.5123  # Å⁻¹ / sqrt(eV)
    DEFAULT_V0 = 12.57  # eV, inner potential
    DEFAULT_WORK_FUNCTION = 4.5  # eV, work function

# ARPES constant documentation:
# k = K_FACTOR * sqrt(Ek) * sin(theta)
# where Ek is kinetic energy in eV, theta in radians, result in Å⁻¹


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


# Public API
__all__ = [
    'get_bz_slice',
    'plot_bz_slice_2d',
    'K_FACTOR',
]
