"""
Brillouin Zone Construction Module for ADAPT

This module provides tools for constructing and visualizing first Brillouin Zones
for ARPES data analysis and momentum-space mapping workflows.

Features:
---------
- Multiple input formats: CIF files, manual parameters, Materials Project API
- Support for all major crystal systems
- High-symmetry point identification and k-path generation
- 3D visualization with matplotlib/plotly/pyvista
- ARPES data overlay utilities

Quick Start:
------------
    from brillouin_zone import generate_bz, plot_bz
    from brillouin_zone.lattice import load_from_parameters
    
    # Create a simple cubic lattice
    lattice = load_from_parameters(a=3.0, b=3.0, c=3.0)
    
    # Generate the Brillouin Zone
    bz = generate_bz(lattice)
    
    # Visualize
    plot_bz(bz)

Crystal Systems Supported:
-------------------------
- Cubic (simple, fcc, bcc)
- Tetragonal
- Hexagonal
- Orthorhombic
- Rhombohedral
- Monoclinic
- Triclinic
"""

# Core classes and functions
from .lattice import (
    CrystalLattice,
    load_from_parameters,
    load_from_cif,
    load_from_formula,
    load_from_material_id,
    compute_reciprocal_lattice,
    detect_crystal_system,
)

from .bz_geometry import (
    BrillouinZone,
    generate_bz,
    wigner_seitz_cell,
    get_high_symmetry_points,
    get_high_symmetry_path,
    get_bz_intersection_plane,
)

from .bz_visualization import (
    plot_bz,
    plot_bz_matplotlib,
    plot_bz_plotly,
    plot_bz_pyvista,
    plot_bz_3_views,
    add_kpoints_to_bz,
    plot_kpath_on_bz,
)

from .arpes_overlay import (
    map_arpes_to_bz,
    create_kxky_plane,
    create_kxkz_trajectory,
    overlay_energy_slice,
    overlay_hv_trajectory,
    get_bz_slice,
    get_bz_slice_at_kz,
    plot_bz_slice_2d,
    K_FACTOR,
    # ARPES hv mapping features
    calc_arpes_hemisphere,
    plot_bz_with_arpes_arc,
    calc_hv_for_kpoint,
    calc_emission_direction,
    calc_direction_for_kpoint,

)


__all__ = [
    # Lattice
    'CrystalLattice',
    'load_from_parameters',
    'load_from_cif',
    'load_from_formula',
    'load_from_material_id',
    'compute_reciprocal_lattice',
    'detect_crystal_system',
    # BZ Geometry
    'BrillouinZone',
    'generate_bz',
    'wigner_seitz_cell',
    'get_high_symmetry_points',
    'get_high_symmetry_path',
    'get_bz_intersection_plane',
    # Visualization
    'plot_bz',
    'plot_bz_matplotlib',
    'plot_bz_plotly',
    'plot_bz_pyvista',
    'plot_bz_3_views',
    'add_kpoints_to_bz',
    'plot_kpath_on_bz',
    # ARPES Overlay
    'map_arpes_to_bz',
    'create_kxky_plane',
    'create_kxkz_trajectory',
    'overlay_energy_slice',
    'overlay_hv_trajectory',
    'get_bz_slice',
    'get_bz_slice_at_kz',
    'plot_bz_slice_2d',
    'K_FACTOR',
    # ARPES hv mapping features
    'calc_arpes_hemisphere',
    'plot_bz_with_arpes_arc',
    'calc_hv_for_kpoint',
    'calc_emission_direction',
    'calc_direction_for_kpoint',

]

__version__ = '0.1.0'
