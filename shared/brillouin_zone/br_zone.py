# br_zone.py  —  a minimal runnable example
# Requires: numpy, scipy, pymatgen, pyvista (optional), plotly (optional), matplotlib

import numpy as np
from scipy.spatial import Voronoi
from pymatgen.core import Structure, Lattice
import math

try:
    import pyvista as pv
    _HAS_PYVISTA = True
except Exception:
    _HAS_PYVISTA = False

try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except Exception:
    _HAS_PLOTLY = False

# ------------------------
# Utilities
# ------------------------
def reciprocal_lattice_from_real(lattice: Lattice):
    """Return reciprocal lattice vectors (3x3) in units of 1/Angstrom (pymatgen Lattice -> matrix)"""
    # pymatgen Lattice has method reciprocal_lattice_crystallographic / reciprocal_lattice
    rec = lattice.reciprocal_lattice_crystallographic.matrix  # 3x3 ndarray (1/Å)
    return np.array(rec)

def generate_reciprocal_points(basis, nrange=3):
    """Generate grid of reciprocal lattice points: sum_i ni * basis[i], ni in [-nrange,nrange]"""
    pts = []
    rng = range(-nrange, nrange+1)
    for i in rng:
        for j in rng:
            for k in rng:
                pts.append(i*basis[0] + j*basis[1] + k*basis[2])
    return np.array(pts)

def extract_region_for_origin(vor: Voronoi, points):
    """
    Find Voronoi region index corresponding to origin point in 'points' array and return vertices & faces.
    The SciPy Voronoi stores regions with indices to vertices.
    """
    # Find index of point closest to origin
    dists = np.linalg.norm(points, axis=1)
    origin_idx = np.argmin(dists)
    region_idx = vor.point_region[origin_idx]
    region_vert_indices = vor.regions[region_idx]
    if -1 in region_vert_indices:
        raise RuntimeError("Region is unbounded. Increase point cloud size.")
    vertices = vor.vertices[region_vert_indices]
    # Build faces (convex hull) - SciPy doesn't directly give faces for region polygon in 3D; do convex hull on vertices
    from scipy.spatial import ConvexHull
    hull = ConvexHull(vertices)
    faces = hull.simplices  # triangles indices
    return vertices, faces

# ------------------------
# Main: load structure and compute BZ
# ------------------------
def compute_brillouin_zone_from_cif(cif_path, nrange=4):
    # Load structure
    st = Structure.from_file(cif_path)
    lattice = st.lattice  # pymatgen Lattice object

    # Ensure primitive? Optionally: use SpacegroupAnalyzer to get conventional primitive cell
    try:
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        sga = SpacegroupAnalyzer(st)
        prim = sga.get_primitive_standard_structure()
        lattice = prim.lattice
    except Exception:
        pass

    # reciprocal lattice vectors (rows)
    rec_basis = reciprocal_lattice_from_real(lattice)  # shape (3,3)

    # Generate reciprocal lattice points centered at origin
    pts = generate_reciprocal_points(rec_basis, nrange=nrange)

    # Compute Voronoi
    vor = Voronoi(pts)

    # Extract region for origin
    verts, faces = extract_region_for_origin(vor, pts)
    return verts, faces, rec_basis, st

# ------------------------
# Visualization helpers
# ------------------------
def visualize_mesh_pyvista(vertices, faces):
    # faces should be triangles (ntri x 3)
    import pyvista as pv
    # convert faces to pyvista face format: [n_idx, i0, i1, i2, n_idx, ...]
    faces_flat = []
    for tri in faces:
        faces_flat.append(3)
        faces_flat.extend(tri.tolist())
    mesh = pv.PolyData(np.array(vertices), np.hstack(faces_flat))
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color="lightblue", show_edges=True, opacity=0.6)
    plotter.add_axes()
    plotter.show()

def visualize_mesh_plotly(vertices, faces):
    tri_x = vertices[faces][:,:,0]
    tri_y = vertices[faces][:,:,1]
    tri_z = vertices[faces][:,:,2]
    # create mesh3d
    i = faces[:,0]
    j = faces[:,1]
    k = faces[:,2]
    fig = go.Figure(data=[go.Mesh3d(x=vertices[:,0], y=vertices[:,1], z=vertices[:,2],
                                   i=i, j=j, k=k, color='lightblue', opacity=0.6)])
    fig.update_layout(scene=dict(aspectmode='data'))
    fig.show()

def visualize_matplotlib(vertices, faces):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    mesh_verts = vertices[faces]
    poly = Poly3DCollection(mesh_verts, alpha=0.4, facecolor='cyan', edgecolor='k')
    ax.add_collection3d(poly)
    # autoscale
    scale = vertices.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    plt.show()

# ------------------------
# Example usage
# ------------------------
if __name__ == "__main__":
    # replace with your CIF path
    cif_path = "example.cif"  # put a real file here
    verts, faces, rec_basis, st = compute_brillouin_zone_from_cif(cif_path, nrange=4)
    print("Vertices:", verts.shape, "Faces:", faces.shape)

    if _HAS_PYVISTA:
        visualize_mesh_pyvista(verts, faces)
    elif _HAS_PLOTLY:
        visualize_mesh_plotly(verts, faces)
    else:
        visualize_matplotlib(verts, faces)
