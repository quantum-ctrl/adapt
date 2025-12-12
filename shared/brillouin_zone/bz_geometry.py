"""
Brillouin Zone Geometry Module

This module handles the construction of the first Brillouin Zone polyhedron
and high-symmetry point identification.

Mathematical Background:
------------------------
The first Brillouin Zone (BZ) is the Wigner-Seitz cell of the reciprocal lattice,
constructed as the Voronoi cell around the origin (Γ point).

Wigner-Seitz Construction Algorithm:
1. Generate a grid of reciprocal lattice points: G = n₁b₁ + n₂b₂ + n₃b₃
2. Compute the Voronoi tessellation
3. Extract the Voronoi region containing the origin
4. Return the vertices and triangular faces of this polyhedron

High-Symmetry Points:
The module includes databases of standard k-points for each crystal system,
following the conventions in:
- Setyawan & Curtarolo, Comp. Mat. Sci. 49, 299-312 (2010)

Usage:
------
    from brillouin_zone import generate_bz
    from brillouin_zone.lattice import load_from_parameters
    
    lattice = load_from_parameters(3.0, 3.0, 3.0)  # Simple cubic
    bz = generate_bz(lattice)
    print(bz.vertices.shape, bz.faces.shape)
"""

import numpy as np
from scipy.spatial import Voronoi, ConvexHull
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any

from .lattice import CrystalLattice


@dataclass
class BrillouinZone:
    """
    Dataclass representing a first Brillouin Zone.
    
    Attributes
    ----------
    vertices : np.ndarray
        Vertices of the BZ polyhedron, shape (N, 3) in Å⁻¹
    faces : np.ndarray
        Triangular faces as vertex indices, shape (M, 3)
    high_symmetry_points : Dict[str, np.ndarray]
        High-symmetry k-points with names as keys (e.g., 'Gamma', 'X', 'M')
    lattice : CrystalLattice
        The source crystal lattice
    reciprocal_basis : np.ndarray
        Reciprocal lattice vectors (3x3 array)
    """
    vertices: np.ndarray
    faces: np.ndarray
    high_symmetry_points: Dict[str, np.ndarray]
    lattice: CrystalLattice
    reciprocal_basis: np.ndarray
    
    @property
    def num_vertices(self) -> int:
        """Number of vertices in the BZ polyhedron."""
        return len(self.vertices)
    
    @property
    def num_faces(self) -> int:
        """Number of triangular faces."""
        return len(self.faces)
    
    def get_volume(self) -> float:
        """Compute the volume of the Brillouin Zone in Å⁻³."""
        # Volume via ConvexHull
        hull = ConvexHull(self.vertices)
        return hull.volume
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the bounding box of the BZ."""
        return self.vertices.min(axis=0), self.vertices.max(axis=0)
    
    def __repr__(self) -> str:
        hs_names = list(self.high_symmetry_points.keys())
        return (f"BrillouinZone(vertices={self.num_vertices}, faces={self.num_faces}, "
                f"crystal={self.lattice.crystal_system}, hs_points={hs_names})")


def generate_reciprocal_lattice_points(basis: np.ndarray, nrange: int = 4) -> np.ndarray:
    """
    Generate a grid of reciprocal lattice points.
    
    Parameters
    ----------
    basis : np.ndarray
        3x3 array with reciprocal lattice vectors as rows
    nrange : int
        Range of integer multiples: -nrange to +nrange for each direction
        
    Returns
    -------
    np.ndarray
        Array of shape ((2*nrange+1)³, 3) with all G vectors
    """
    pts = []
    for i in range(-nrange, nrange + 1):
        for j in range(-nrange, nrange + 1):
            for k in range(-nrange, nrange + 1):
                pts.append(i * basis[0] + j * basis[1] + k * basis[2])
    return np.array(pts)


def extract_wigner_seitz_cell(vor: Voronoi, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract the Voronoi region (Wigner-Seitz cell) containing the origin.
    
    Parameters
    ----------
    vor : Voronoi
        Computed Voronoi diagram
    points : np.ndarray
        The input points for the Voronoi diagram
        
    Returns
    -------
    vertices : np.ndarray
        Vertices of the Wigner-Seitz cell (N, 3)
    faces : np.ndarray
        Triangular faces as vertex indices (M, 3)
        
    Raises
    ------
    RuntimeError
        If the region is unbounded (need larger point cloud)
    """
    # Find point closest to origin
    dists = np.linalg.norm(points, axis=1)
    origin_idx = np.argmin(dists)
    
    # Get the region for this point
    region_idx = vor.point_region[origin_idx]
    region_vert_indices = vor.regions[region_idx]
    
    # Check for unbounded region
    if -1 in region_vert_indices:
        raise RuntimeError(
            "BZ region is unbounded. This shouldn't happen for a well-behaved lattice. "
            "Try increasing nrange parameter."
        )
    
    # Extract vertices
    vertices = vor.vertices[region_vert_indices]
    
    # Build faces using ConvexHull on the extracted vertices
    # This gives us the triangulated surface
    hull = ConvexHull(vertices)
    faces = hull.simplices
    
    return vertices, faces


def wigner_seitz_cell(reciprocal_basis: np.ndarray, nrange: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the first Brillouin Zone as a Wigner-Seitz cell.
    
    Parameters
    ----------
    reciprocal_basis : np.ndarray
        3x3 array with reciprocal lattice vectors as rows (b₁, b₂, b₃) in Å⁻¹
    nrange : int
        Range for reciprocal lattice point generation. Larger values give
        more robust results for unusual lattices.
        
    Returns
    -------
    vertices : np.ndarray
        Vertices of the first BZ polyhedron, shape (N, 3)
    faces : np.ndarray
        Triangular faces as vertex indices, shape (M, 3)
    """
    # Generate reciprocal lattice points
    points = generate_reciprocal_lattice_points(reciprocal_basis, nrange)
    
    # Compute Voronoi tessellation
    vor = Voronoi(points)
    
    # Extract the cell around the origin
    vertices, faces = extract_wigner_seitz_cell(vor, points)
    
    return vertices, faces


# ============================================================================
# High-Symmetry Points Database
# ============================================================================
# Following Setyawan & Curtarolo conventions

def get_high_symmetry_points_cubic(reciprocal_basis: np.ndarray, 
                                    bravais_type: str = 'fcc') -> Dict[str, np.ndarray]:
    """
    Get high-symmetry points for cubic lattices.
    
    Parameters
    ----------
    reciprocal_basis : np.ndarray
        Reciprocal lattice vectors
    bravais_type : str
        'fcc', 'bcc', or 'sc' (simple cubic)
        
    Returns
    -------
    dict
        Dictionary mapping point names to k-vectors
    """
    b1, b2, b3 = reciprocal_basis
    
    points = {
        'Gamma': np.array([0.0, 0.0, 0.0]),
    }
    
    if bravais_type == 'fcc':
        # FCC conventional BZ points (in Cartesian coordinates)
        points.update({
            'X': 0.5 * (b1 + b3),          # (0, 0.5, 0.5) fractional
            'L': 0.5 * (b1 + b2 + b3),     # (0.5, 0.5, 0.5)
            'W': 0.25 * b1 + 0.5 * b2 + 0.75 * b3,  # (0.25, 0.5, 0.75)
            'U': 0.25 * b1 + 0.625 * b2 + 0.625 * b3,
            'K': 0.375 * b1 + 0.375 * b2 + 0.75 * b3,
        })
    elif bravais_type == 'bcc':
        # BCC conventional BZ points
        points.update({
            'H': 0.5 * (b1 + b2 - b3),     # (0.5, 0.5, -0.5)
            'P': 0.25 * (b1 + b2 + b3),    # (0.25, 0.25, 0.25)
            'N': 0.5 * (b1 + b2),          # (0.5, 0.5, 0)
        })
    else:  # simple cubic
        points.update({
            'X': 0.5 * b1,                 # (0.5, 0, 0)
            'M': 0.5 * (b1 + b2),          # (0.5, 0.5, 0)
            'R': 0.5 * (b1 + b2 + b3),     # (0.5, 0.5, 0.5)
        })
    
    return points


def get_high_symmetry_points_hexagonal(reciprocal_basis: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get high-symmetry points for hexagonal lattices.
    
    The hexagonal BZ has points Γ, M, K, A, L, H.
    """
    b1, b2, b3 = reciprocal_basis
    
    return {
        'Gamma': np.array([0.0, 0.0, 0.0]),
        'M': 0.5 * b1,                        # (0.5, 0, 0)
        'K': (1.0/3.0) * b1 + (1.0/3.0) * b2, # (1/3, 1/3, 0)
        'A': 0.5 * b3,                        # (0, 0, 0.5)
        'L': 0.5 * b1 + 0.5 * b3,             # (0.5, 0, 0.5)
        'H': (1.0/3.0) * b1 + (1.0/3.0) * b2 + 0.5 * b3,  # (1/3, 1/3, 0.5)
    }


def get_high_symmetry_points_tetragonal(reciprocal_basis: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get high-symmetry points for tetragonal lattices.
    """
    b1, b2, b3 = reciprocal_basis
    
    return {
        'Gamma': np.array([0.0, 0.0, 0.0]),
        'X': 0.5 * b1,                     # (0.5, 0, 0)
        'M': 0.5 * (b1 + b2),              # (0.5, 0.5, 0)
        'Z': 0.5 * b3,                     # (0, 0, 0.5)
        'R': 0.5 * (b1 + b3),              # (0.5, 0, 0.5)
        'A': 0.5 * (b1 + b2 + b3),         # (0.5, 0.5, 0.5)
    }


def get_high_symmetry_points_orthorhombic(reciprocal_basis: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get high-symmetry points for orthorhombic lattices.
    """
    b1, b2, b3 = reciprocal_basis
    
    return {
        'Gamma': np.array([0.0, 0.0, 0.0]),
        'X': 0.5 * b1,                     # (0.5, 0, 0)
        'Y': 0.5 * b2,                     # (0, 0.5, 0)
        'Z': 0.5 * b3,                     # (0, 0, 0.5)
        'S': 0.5 * (b1 + b2),              # (0.5, 0.5, 0)
        'U': 0.5 * (b1 + b3),              # (0.5, 0, 0.5)
        'T': 0.5 * (b2 + b3),              # (0, 0.5, 0.5)
        'R': 0.5 * (b1 + b2 + b3),         # (0.5, 0.5, 0.5)
    }


def get_high_symmetry_points_rhombohedral(reciprocal_basis: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get high-symmetry points for rhombohedral lattices.
    """
    b1, b2, b3 = reciprocal_basis
    
    return {
        'Gamma': np.array([0.0, 0.0, 0.0]),
        'L': 0.5 * (b1 + b2 + b3),         # (0.5, 0.5, 0.5)
        'T': 0.5 * b1 + 0.5 * b2,          # (0.5, 0.5, 0)
        'F': 0.5 * b1,                     # (0.5, 0, 0)
        'X': 0.5 * (b2 + b3),              # (0, 0.5, 0.5)
    }


def get_high_symmetry_points_monoclinic(reciprocal_basis: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Get high-symmetry points for monoclinic lattices.
    """
    b1, b2, b3 = reciprocal_basis
    
    return {
        'Gamma': np.array([0.0, 0.0, 0.0]),
        'Y': 0.5 * b2,                     # (0, 0.5, 0)
        'Z': 0.5 * b3,                     # (0, 0, 0.5)
        'C': 0.5 * (b2 + b3),              # (0, 0.5, 0.5)
        'D': 0.5 * (b1 + b3),              # (0.5, 0, 0.5)
        'A': 0.5 * (b1 + b2 + b3),         # (0.5, 0.5, 0.5)
        'E': 0.5 * (b1 + b2),              # (0.5, 0.5, 0)
    }


def get_high_symmetry_points(crystal_system: str, 
                             reciprocal_basis: np.ndarray,
                             bravais_type: str = '') -> Dict[str, np.ndarray]:
    """
    Get high-symmetry k-points for a given crystal system.
    
    Parameters
    ----------
    crystal_system : str
        Crystal system: 'cubic', 'hexagonal', 'tetragonal', 
        'orthorhombic', 'rhombohedral', 'monoclinic', 'triclinic'
    reciprocal_basis : np.ndarray
        3x3 array with reciprocal lattice vectors as rows
    bravais_type : str, optional
        For cubic systems: 'fcc', 'bcc', or 'sc'. Auto-detected if not specified.
        
    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping point names (e.g., 'Gamma', 'X', 'M') to k-vectors in Å⁻¹
    """
    system_lower = crystal_system.lower()
    
    if system_lower == 'cubic':
        return get_high_symmetry_points_cubic(reciprocal_basis, bravais_type or 'sc')
    elif system_lower == 'hexagonal':
        return get_high_symmetry_points_hexagonal(reciprocal_basis)
    elif system_lower == 'tetragonal':
        return get_high_symmetry_points_tetragonal(reciprocal_basis)
    elif system_lower == 'orthorhombic':
        return get_high_symmetry_points_orthorhombic(reciprocal_basis)
    elif system_lower == 'rhombohedral':
        return get_high_symmetry_points_rhombohedral(reciprocal_basis)
    elif system_lower == 'monoclinic':
        return get_high_symmetry_points_monoclinic(reciprocal_basis)
    else:
        # Triclinic or unknown - just return Gamma
        return {'Gamma': np.array([0.0, 0.0, 0.0])}


def get_high_symmetry_path(points: Dict[str, np.ndarray],
                           path_spec: List[str],
                           n_points: int = 100) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, str]]]:
    """
    Generate a k-path through high-symmetry points.
    
    Parameters
    ----------
    points : Dict[str, np.ndarray]
        High-symmetry points dictionary
    path_spec : List[str]
        Path specification, e.g., ['Gamma', 'X', 'M', 'Gamma', 'R']
    n_points : int
        Total number of points along the path
        
    Returns
    -------
    k_path : np.ndarray
        Array of k-points along the path, shape (n_points, 3)
    k_dist : np.ndarray
        Cumulative distance along the path for plotting
    labels : List[Tuple[int, str]]
        List of (index, label) for tick marks at high-symmetry points
        
    Example
    -------
    >>> from brillouin_zone.bz_geometry import get_high_symmetry_points, get_high_symmetry_path
    >>> points = get_high_symmetry_points('cubic', reciprocal_basis, 'sc')
    >>> k_path, k_dist, labels = get_high_symmetry_path(points, ['Gamma', 'X', 'M', 'Gamma'])
    """
    if len(path_spec) < 2:
        raise ValueError("Path specification needs at least 2 points")
    
    # Calculate total path length
    segments = []
    total_length = 0.0
    for i in range(len(path_spec) - 1):
        start = points[path_spec[i]]
        end = points[path_spec[i + 1]]
        seg_length = np.linalg.norm(end - start)
        segments.append((start, end, seg_length))
        total_length += seg_length
    
    # Distribute points proportionally to segment length
    k_path = []
    k_dist = []
    labels = []
    current_dist = 0.0
    
    for seg_idx, (start, end, seg_length) in enumerate(segments):
        # Number of points for this segment
        n_seg = max(2, int(n_points * seg_length / total_length))
        if seg_idx == len(segments) - 1:
            # Last segment gets remaining points
            n_seg = n_points - len(k_path)
        
        # Add label for start point
        if seg_idx == 0:
            labels.append((0, path_spec[0]))
        
        # Generate points for this segment
        for j in range(n_seg):
            t = j / (n_seg - 1) if n_seg > 1 else 0
            point = (1 - t) * start + t * end
            k_path.append(point)
            k_dist.append(current_dist + t * seg_length)
        
        current_dist += seg_length
        labels.append((len(k_path) - 1, path_spec[seg_idx + 1]))
    
    return np.array(k_path), np.array(k_dist), labels


def generate_bz(lattice: CrystalLattice, 
                nrange: int = 4,
                bravais_type: str = '') -> BrillouinZone:
    """
    Generate the first Brillouin Zone for a crystal lattice.
    
    This is the main entry point for BZ construction. It computes the 
    Wigner-Seitz cell of the reciprocal lattice and identifies high-symmetry points.
    
    Parameters
    ----------
    lattice : CrystalLattice
        Crystal lattice object from lattice module
    nrange : int, optional
        Range for reciprocal lattice point generation. Default: 4
    bravais_type : str, optional
        Bravais lattice type for cubic systems ('fcc', 'bcc', 'sc').
        If not specified, defaults to 'sc'.
        
    Returns
    -------
    BrillouinZone
        Brillouin Zone object with vertices, faces, and high-symmetry points
        
    Examples
    --------
    >>> from brillouin_zone import generate_bz
    >>> from brillouin_zone.lattice import load_from_parameters
    >>> 
    >>> # Simple cubic
    >>> lat = load_from_parameters(3.0, 3.0, 3.0)
    >>> bz = generate_bz(lat)
    >>> print(bz)
    BrillouinZone(vertices=8, faces=12, crystal=cubic, hs_points=['Gamma', 'X', 'M', 'R'])
    
    >>> # Hexagonal
    >>> lat_hex = load_from_parameters(2.46, 2.46, 6.71, gamma=120)
    >>> bz_hex = generate_bz(lat_hex)
    >>> print(bz_hex.high_symmetry_points.keys())
    dict_keys(['Gamma', 'M', 'K', 'A', 'L', 'H'])
    """
    reciprocal_basis = lattice.reciprocal_vectors
    
    # Construct Wigner-Seitz cell
    vertices, faces = wigner_seitz_cell(reciprocal_basis, nrange)
    
    # Get high-symmetry points
    hs_points = get_high_symmetry_points(
        lattice.crystal_system, 
        reciprocal_basis,
        bravais_type
    )
    
    return BrillouinZone(
        vertices=vertices,
        faces=faces,
        high_symmetry_points=hs_points,
        lattice=lattice,
        reciprocal_basis=reciprocal_basis
    )


def get_bz_intersection_plane(bz: 'BrillouinZone', 
                             plane_normal: List[float], 
                             slice_value: float = 0.0) -> Optional[np.ndarray]:
    """
    Compute the 3D polygon formed by the intersection of the BZ with a plane.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object
    plane_normal : array-like
        Normal vector of the plane (e.g., Miller indices [h, k, l])
    slice_value : float
        Distance from origin (plane eq: normal · r = slice_value)
        
    Returns
    -------
    np.ndarray or None
        Vertices of the intersection polygon in 3D, shape (N, 3).
        Returns None if no intersection.
    """
    vertices = bz.vertices
    normal = np.array(plane_normal)
    norm_mag = np.linalg.norm(normal)
    if norm_mag < 1e-10:
        return None
    normal = normal / norm_mag
    
    intersection_points = []
    
    # Iterate over all faces (triangles)
    for face in bz.faces:
        v0, v1, v2 = vertices[face]
        # Edges of the triangle
        edges = [(v0, v1), (v1, v2), (v2, v0)]
        for p1, p2 in edges:
            d1 = np.dot(p1, normal) - slice_value
            d2 = np.dot(p2, normal) - slice_value
            
            # Check for intersection (signs differ)
            if d1 * d2 < 0:
                # Linear interpolation
                t = d1 / (d1 - d2)
                intersection = p1 + t * (p2 - p1)
                intersection_points.append(intersection)
            elif np.abs(d1) < 1e-9:
                # Vertex lies on plane (rare exact float match)
                intersection_points.append(p1)
    
    if len(intersection_points) < 3:
        return None
        
    points = np.array(intersection_points)
    
    # Remove duplicates
    unique_points = []
    for pt in points:
        if not any(np.linalg.norm(pt - upt) < 1e-6 for upt in unique_points):
            unique_points.append(pt)
    points = np.array(unique_points)
    
    if len(points) < 3:
        return None

    # Sort points to form a convex polygon
    # Project to 2D basis on the plane
    if np.abs(normal[0]) < 0.9:
        u = np.cross(normal, [1, 0, 0])
    else:
        u = np.cross(normal, [0, 1, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    
    coords_2d = points @ np.column_stack((u, v))
    
    try:
        hull = ConvexHull(coords_2d)
        sorted_points = points[hull.vertices]
        return sorted_points
    except Exception:
        return points


# Public API
__all__ = [
    'BrillouinZone',
    'generate_bz',
    'wigner_seitz_cell',
    'get_high_symmetry_points',
    'get_high_symmetry_path',
    'get_bz_intersection_plane',
]
