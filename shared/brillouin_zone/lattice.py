"""
Lattice Module for Brillouin Zone Construction

This module handles crystal lattice input, parsing, and reciprocal lattice computation.

Supports:
- CIF file input (via pymatgen)
- Manual lattice parameters + space group
- Chemical formula lookup (via Materials Project API)

Mathematical Background:
------------------------
Reciprocal lattice vectors are computed from real-space lattice vectors (a₁, a₂, a₃):

    b₁ = 2π (a₂ × a₃) / V
    b₂ = 2π (a₃ × a₁) / V
    b₃ = 2π (a₁ × a₂) / V

where V = a₁ · (a₂ × a₃) is the unit cell volume.

Usage:
------
    from brillouin_zone.lattice import load_from_cif, load_from_parameters
    
    # From CIF file
    lattice = load_from_cif("structure.cif")
    
    # From manual parameters
    lattice = load_from_parameters(a=3.0, b=3.0, c=3.0, 
                                   alpha=90, beta=90, gamma=90,
                                   space_group='Pm-3m')
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import os

# Optional imports with fallbacks
_PYMATGEN_ERROR = None
try:
    from pymatgen.core import Structure, Lattice as PymatgenLattice
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    _HAS_PYMATGEN = True
except ImportError as e:
    _HAS_PYMATGEN = False
    _PYMATGEN_ERROR = str(e)

_HAS_MP_API = False
_MP_API_ERROR = None
try:
    try:
        from mp_api.client import MPRester
    except ImportError:
        # Fallback for alternative or older mp-api paths
        from mp_api import MPRester
    _HAS_MP_API = True
except ImportError as e:
    _HAS_MP_API = False
    _MP_API_ERROR = str(e)
except Exception as e:
    # Catching other potential version-related initialization errors (e.g. numpy mismatch)
    _HAS_MP_API = False
    _MP_API_ERROR = f"{type(e).__name__}: {str(e)}"


# Crystal system classification based on lattice parameters
CRYSTAL_SYSTEMS = {
    'cubic': {'a=b=c': True, 'alpha=beta=gamma=90': True},
    'tetragonal': {'a=b≠c': True, 'alpha=beta=gamma=90': True},
    'hexagonal': {'a=b≠c': True, 'alpha=beta=90,gamma=120': True},
    'orthorhombic': {'a≠b≠c': True, 'alpha=beta=gamma=90': True},
    'rhombohedral': {'a=b=c': True, 'alpha=beta=gamma≠90': True},
    'monoclinic': {'a≠b≠c': True, 'alpha=gamma=90,beta≠90': True},
    'triclinic': {'a≠b≠c': True, 'alpha≠beta≠gamma': True},
}


@dataclass
class CrystalLattice:
    """
    Dataclass representing a crystal lattice.
    
    Attributes
    ----------
    real_vectors : np.ndarray
        Real-space lattice vectors as 3x3 array (rows are a₁, a₂, a₃) in Ångströms
    reciprocal_vectors : np.ndarray
        Reciprocal lattice vectors as 3x3 array (rows are b₁, b₂, b₃) in Å⁻¹
    a, b, c : float
        Lattice constants in Ångströms
    alpha, beta, gamma : float
        Lattice angles in degrees
    crystal_system : str
        Detected crystal system (cubic, hexagonal, etc.)
    space_group : str
        Space group symbol (e.g., 'Pm-3m', 'P6/mmm')
    formula : Optional[str]
        Chemical formula if available
    """
    real_vectors: np.ndarray
    reciprocal_vectors: np.ndarray
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    gamma: float
    crystal_system: str
    space_group: str = ""
    formula: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def volume(self) -> float:
        """Unit cell volume in Å³."""
        return np.abs(np.dot(self.real_vectors[0], 
                             np.cross(self.real_vectors[1], self.real_vectors[2])))
    
    @property
    def reciprocal_volume(self) -> float:
        """Reciprocal cell volume in Å⁻³."""
        return (2 * np.pi)**3 / self.volume
    
    def __repr__(self) -> str:
        return (f"CrystalLattice(a={self.a:.4f}, b={self.b:.4f}, c={self.c:.4f}, "
                f"α={self.alpha:.2f}°, β={self.beta:.2f}°, γ={self.gamma:.2f}°, "
                f"system={self.crystal_system}, sg={self.space_group})")


def compute_reciprocal_lattice(real_vectors: np.ndarray) -> np.ndarray:
    """
    Compute reciprocal lattice vectors from real-space lattice vectors.
    
    Uses the standard crystallographic convention with 2π factor.
    
    Parameters
    ----------
    real_vectors : np.ndarray
        3x3 array with real-space lattice vectors as rows (a₁, a₂, a₃) in Å
        
    Returns
    -------
    np.ndarray
        3x3 array with reciprocal lattice vectors as rows (b₁, b₂, b₃) in Å⁻¹
        
    Notes
    -----
    The reciprocal lattice vectors satisfy: aᵢ · bⱼ = 2π δᵢⱼ
    
    Formula:
        b₁ = 2π (a₂ × a₃) / V
        b₂ = 2π (a₃ × a₁) / V
        b₃ = 2π (a₁ × a₂) / V
    where V = a₁ · (a₂ × a₃)
    """
    a1, a2, a3 = real_vectors[0], real_vectors[1], real_vectors[2]
    
    # Unit cell volume
    volume = np.dot(a1, np.cross(a2, a3))
    
    if np.abs(volume) < 1e-10:
        raise ValueError("Lattice vectors are coplanar (volume ≈ 0)")
    
    # Reciprocal lattice vectors with 2π factor
    b1 = 2 * np.pi * np.cross(a2, a3) / volume
    b2 = 2 * np.pi * np.cross(a3, a1) / volume
    b3 = 2 * np.pi * np.cross(a1, a2) / volume
    
    return np.array([b1, b2, b3])


def detect_crystal_system(a: float, b: float, c: float,
                          alpha: float, beta: float, gamma: float,
                          tol: float = 0.1) -> str:
    """
    Detect crystal system from lattice parameters.
    
    Parameters
    ----------
    a, b, c : float
        Lattice constants in Ångströms
    alpha, beta, gamma : float  
        Lattice angles in degrees
    tol : float
        Tolerance for floating point comparisons (degrees and relative)
        
    Returns
    -------
    str
        Crystal system name: 'cubic', 'tetragonal', 'hexagonal', 
        'orthorhombic', 'rhombohedral', 'monoclinic', or 'triclinic'
    """
    def eq(x, y, rel_tol=tol/100):
        """Check if two values are approximately equal."""
        return np.abs(x - y) < rel_tol * max(abs(x), abs(y), 1)
    
    def angle_eq(x, y):
        """Check if two angles are approximately equal."""
        return np.abs(x - y) < tol
    
    # Check cubic: a = b = c, α = β = γ = 90°
    if eq(a, b) and eq(b, c) and angle_eq(alpha, 90) and angle_eq(beta, 90) and angle_eq(gamma, 90):
        return 'cubic'
    
    # Check hexagonal: a = b ≠ c, α = β = 90°, γ = 120°
    if eq(a, b) and not eq(a, c) and angle_eq(alpha, 90) and angle_eq(beta, 90) and angle_eq(gamma, 120):
        return 'hexagonal'
    
    # Check rhombohedral: a = b = c, α = β = γ ≠ 90°
    if eq(a, b) and eq(b, c) and angle_eq(alpha, beta) and angle_eq(beta, gamma) and not angle_eq(alpha, 90):
        return 'rhombohedral'
    
    # Check tetragonal: a = b ≠ c, α = β = γ = 90°
    if eq(a, b) and not eq(a, c) and angle_eq(alpha, 90) and angle_eq(beta, 90) and angle_eq(gamma, 90):
        return 'tetragonal'
    
    # Check orthorhombic: a ≠ b ≠ c, α = β = γ = 90°
    if angle_eq(alpha, 90) and angle_eq(beta, 90) and angle_eq(gamma, 90):
        return 'orthorhombic'
    
    # Check monoclinic: α = γ = 90°, β ≠ 90°
    if angle_eq(alpha, 90) and angle_eq(gamma, 90) and not angle_eq(beta, 90):
        return 'monoclinic'
    
    # Default to triclinic
    return 'triclinic'


def lattice_vectors_from_parameters(a: float, b: float, c: float,
                                     alpha: float, beta: float, gamma: float) -> np.ndarray:
    """
    Compute lattice vectors from lattice parameters.
    
    Uses the standard crystallographic convention where:
    - a₁ is along x-axis
    - a₂ is in the xy-plane
    - a₃ has components in all directions
    
    Parameters
    ----------
    a, b, c : float
        Lattice constants in Å
    alpha, beta, gamma : float
        Lattice angles in degrees (α: angle between b and c, etc.)
        
    Returns
    -------
    np.ndarray
        3x3 array with lattice vectors as rows
    """
    # Convert angles to radians
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    gamma_rad = np.radians(gamma)
    
    # a₁ along x-axis
    a1 = np.array([a, 0, 0])
    
    # a₂ in xy-plane
    a2 = np.array([b * np.cos(gamma_rad), b * np.sin(gamma_rad), 0])
    
    # a₃ general direction
    c1 = c * np.cos(beta_rad)
    c2 = c * (np.cos(alpha_rad) - np.cos(beta_rad) * np.cos(gamma_rad)) / np.sin(gamma_rad)
    c3 = np.sqrt(c**2 - c1**2 - c2**2)
    a3 = np.array([c1, c2, c3])
    
    return np.array([a1, a2, a3])


def load_from_parameters(a: float, b: float, c: float,
                         alpha: float = 90.0, beta: float = 90.0, gamma: float = 90.0,
                         space_group: str = "") -> CrystalLattice:
    """
    Create a CrystalLattice from manual lattice parameters.
    
    Parameters
    ----------
    a, b, c : float
        Lattice constants in Ångströms
    alpha, beta, gamma : float, optional
        Lattice angles in degrees. Default: 90° (cubic/orthorhombic)
    space_group : str, optional
        Space group symbol (e.g., 'Pm-3m')
        
    Returns
    -------
    CrystalLattice
        Crystal lattice object with computed reciprocal vectors
        
    Examples
    --------
    >>> # Simple cubic
    >>> lat = load_from_parameters(3.0, 3.0, 3.0)
    >>> print(lat.crystal_system)
    'cubic'
    
    >>> # Hexagonal graphite
    >>> lat = load_from_parameters(2.46, 2.46, 6.71, gamma=120)
    >>> print(lat.crystal_system)
    'hexagonal'
    """
    # Compute lattice vectors
    real_vectors = lattice_vectors_from_parameters(a, b, c, alpha, beta, gamma)
    
    # Compute reciprocal lattice
    reciprocal_vectors = compute_reciprocal_lattice(real_vectors)
    
    # Detect crystal system
    crystal_system = detect_crystal_system(a, b, c, alpha, beta, gamma)
    
    return CrystalLattice(
        real_vectors=real_vectors,
        reciprocal_vectors=reciprocal_vectors,
        a=a, b=b, c=c,
        alpha=alpha, beta=beta, gamma=gamma,
        crystal_system=crystal_system,
        space_group=space_group
    )


def load_from_cif(cif_path: str, use_primitive: bool = True) -> CrystalLattice:
    """
    Load crystal lattice from a CIF file.
    
    Parameters
    ----------
    cif_path : str
        Path to CIF file
    use_primitive : bool, optional
        If True, use the primitive cell. If False, use the conventional cell.
        Default: True
        
    Returns
    -------
    CrystalLattice
        Crystal lattice object
        
    Raises
    ------
    ImportError
        If pymatgen is not installed
    FileNotFoundError
        If CIF file does not exist
    """
    if not _HAS_PYMATGEN:
        msg = "pymatgen is required for CIF parsing."
        if _PYMATGEN_ERROR:
            msg += f" (Import error: {_PYMATGEN_ERROR})"
        raise ImportError(f"{msg} Install with: pip install pymatgen")
    
    if not os.path.exists(cif_path):
        raise FileNotFoundError(f"CIF file not found: {cif_path}")
    
    # Load structure
    structure = Structure.from_file(cif_path)
    
    # Get primitive cell if requested
    if use_primitive:
        try:
            sga = SpacegroupAnalyzer(structure)
            structure = sga.get_primitive_standard_structure()
            space_group = sga.get_space_group_symbol()
        except Exception:
            space_group = ""
    else:
        try:
            sga = SpacegroupAnalyzer(structure)
            space_group = sga.get_space_group_symbol()
        except Exception:
            space_group = ""
    
    lattice = structure.lattice
    
    return CrystalLattice(
        real_vectors=np.array(lattice.matrix),
        reciprocal_vectors=np.array(lattice.reciprocal_lattice.matrix),
        a=lattice.a, b=lattice.b, c=lattice.c,
        alpha=lattice.alpha, beta=lattice.beta, gamma=lattice.gamma,
        crystal_system=detect_crystal_system(
            lattice.a, lattice.b, lattice.c,
            lattice.alpha, lattice.beta, lattice.gamma
        ),
        space_group=space_group,
        formula=structure.composition.reduced_formula
    )


def load_from_formula(formula: str, api_key: Optional[str] = None,
                      use_primitive: bool = True) -> CrystalLattice:
    """
    Load crystal lattice by querying Materials Project database.
    
    Parameters
    ----------
    formula : str
        Chemical formula (e.g., 'PtGa', 'Si', 'GaAs')
    api_key : str, optional
        Materials Project API key. If not provided, reads from MP_API_KEY env variable.
    use_primitive : bool, optional
        If True, use the primitive cell. Default: True
        
    Returns
    -------
    CrystalLattice
        Crystal lattice for the most stable structure
        
    Raises
    ------
    ImportError
        If mp-api is not installed
    ValueError
        If no structures found for the formula
    """
    if not _HAS_MP_API:
        error_msg = "mp-api is required for Materials Project queries."
        if _MP_API_ERROR:
            error_msg += f" (Import error: {_MP_API_ERROR})"
        raise ImportError(f"{error_msg} Install with: pip install mp-api")
    
    if api_key is None:
        api_key = os.getenv("MP_API_KEY")
        if not api_key:
            raise ValueError("Materials Project API key not found. "
                             "Set MP_API_KEY environment variable or pass api_key parameter.")
    
    with MPRester(api_key) as mpr:
        results = mpr.materials.search(
            formula=formula,
            fields=["material_id", "formula_pretty", "symmetry", "structure"]
        )
    
    if not results:
        raise ValueError(f"No structures found for formula: {formula}")
    
    # Use first (most stable) result
    result = results[0]
    structure = result.structure
    
    if use_primitive:
        try:
            sga = SpacegroupAnalyzer(structure)
            structure = sga.get_primitive_standard_structure()
        except Exception:
            pass
    
    lattice = structure.lattice
    space_group = result.symmetry.symbol if result.symmetry else ""
    
    return CrystalLattice(
        real_vectors=np.array(lattice.matrix),
        reciprocal_vectors=np.array(lattice.reciprocal_lattice.matrix),
        a=lattice.a, b=lattice.b, c=lattice.c,
        alpha=lattice.alpha, beta=lattice.beta, gamma=lattice.gamma,
        crystal_system=detect_crystal_system(
            lattice.a, lattice.b, lattice.c,
            lattice.alpha, lattice.beta, lattice.gamma
        ),
        space_group=space_group,
        formula=result.formula_pretty,
        metadata={'material_id': str(result.material_id)}
    )


def load_from_material_id(material_id: str, api_key: Optional[str] = None,
                          use_primitive: bool = True) -> CrystalLattice:
    """
    Load crystal lattice by Materials Project material ID.
    
    Parameters
    ----------
    material_id : str
        Materials Project ID (e.g., 'mp-1078526')
    api_key : str, optional
        Materials Project API key. If not provided, reads from MP_API_KEY env variable.
    use_primitive : bool, optional
        If True, use the primitive cell. Default: True
        
    Returns
    -------
    CrystalLattice
        Crystal lattice for the specified material
        
    Raises
    ------
    ImportError
        If mp-api is not installed
    ValueError
        If no structure found for the material ID
    """
    if not _HAS_MP_API:
        error_msg = "mp-api is required for Materials Project queries."
        if _MP_API_ERROR:
            error_msg += f" (Import error: {_MP_API_ERROR})"
        raise ImportError(f"{error_msg} Install with: pip install mp-api")
    
    if api_key is None:
        api_key = os.getenv("MP_API_KEY")
        if not api_key:
            raise ValueError("Materials Project API key not found. "
                             "Set MP_API_KEY environment variable or pass api_key parameter.")
    
    with MPRester(api_key) as mpr:
        # Use get_structure_by_material_id for direct ID lookup
        try:
            structure = mpr.get_structure_by_material_id(material_id)
            # Get symmetry info separately
            result = mpr.materials.search(
                material_ids=[material_id],
                fields=["material_id", "formula_pretty", "symmetry"]
            )
            if result:
                formula = result[0].formula_pretty
                space_group = result[0].symmetry.symbol if result[0].symmetry else ""
            else:
                formula = structure.composition.reduced_formula
                space_group = ""
        except Exception as e:
            raise ValueError(f"Failed to load structure for material ID {material_id}: {e}")
    
    if structure is None:
        raise ValueError(f"No structure found for material ID: {material_id}")
    
    if use_primitive:
        try:
            sga = SpacegroupAnalyzer(structure)
            structure = sga.get_primitive_standard_structure()
            if not space_group:
                space_group = sga.get_space_group_symbol()
        except Exception:
            pass
    
    lattice = structure.lattice
    
    return CrystalLattice(
        real_vectors=np.array(lattice.matrix),
        reciprocal_vectors=np.array(lattice.reciprocal_lattice.matrix),
        a=lattice.a, b=lattice.b, c=lattice.c,
        alpha=lattice.alpha, beta=lattice.beta, gamma=lattice.gamma,
        crystal_system=detect_crystal_system(
            lattice.a, lattice.b, lattice.c,
            lattice.alpha, lattice.beta, lattice.gamma
        ),
        space_group=space_group,
        formula=formula,
        metadata={'material_id': material_id}
    )


# Public API
__all__ = [
    'CrystalLattice',
    'load_from_parameters',
    'load_from_cif',
    'load_from_formula',
    'load_from_material_id',
    'compute_reciprocal_lattice',
    'detect_crystal_system',
]
