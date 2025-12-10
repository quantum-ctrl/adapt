"""
Brillouin Zone Visualization Module

This module provides plotting utilities for Brillouin Zone visualization,
supporting matplotlib, plotly, and optionally pyvista backends.

Usage:
------
    from brillouin_zone import generate_bz, plot_bz
    from brillouin_zone.lattice import load_from_parameters
    
    lattice = load_from_parameters(3.0, 3.0, 3.0)
    bz = generate_bz(lattice)
    plot_bz(bz)  # Interactive 3D plot
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from typing import Optional, List, Tuple, Dict, Any, Union

from .bz_geometry import BrillouinZone

# Check for optional backends
try:
    import plotly.graph_objects as go
    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    import pyvista as pv
    _HAS_PYVISTA = True
except ImportError:
    _HAS_PYVISTA = False


def plot_bz_matplotlib(bz: BrillouinZone,
                       ax: Optional[plt.Axes] = None,
                       figsize: Tuple[int, int] = (8, 8),
                       facecolor: str = 'cyan',
                       edgecolor: str = 'black',
                       alpha: float = 0.3,
                       show_hs_points: bool = True,
                       hs_color: str = 'red',
                       hs_size: int = 50,
                       show_axes_labels: bool = True,
                       title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot Brillouin Zone using matplotlib 3D.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object to plot
    ax : matplotlib.axes.Axes, optional
        Existing 3D axes. If None, creates new figure.
    figsize : tuple, optional
        Figure size in inches
    facecolor : str
        Face color for the BZ polyhedron
    edgecolor : str
        Edge color for the BZ polyhedron
    alpha : float
        Transparency (0-1)
    show_hs_points : bool
        Whether to show high-symmetry points
    hs_color : str
        Color for high-symmetry point markers
    hs_size : int
        Size of high-symmetry point markers
    show_axes_labels : bool
        Whether to show kx, ky, kz axis labels
    title : str, optional
        Plot title
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.get_figure()
    
    # Get mesh vertices for faces
    mesh_vertices = bz.vertices[bz.faces]
    
    # Create Poly3DCollection
    poly_kwargs = {
        'facecolor': facecolor,
        'edgecolor': edgecolor,
        'linewidth': 1.0 if facecolor == 'none' or (isinstance(facecolor, tuple) and len(facecolor) == 4 and facecolor[3] == 0) else 0.5
    }
    if alpha is not None:
        poly_kwargs['alpha'] = alpha
        
    poly = Poly3DCollection(mesh_vertices, **poly_kwargs)
    ax.add_collection3d(poly)
    
    # Plot high-symmetry points
    if show_hs_points:
        for name, point in bz.high_symmetry_points.items():
            ax.scatter(*point, c=hs_color, s=hs_size, marker='o', depthshade=False)
            # Add label with offset
            offset = 0.05 * np.max(np.abs(bz.vertices))
            ax.text(point[0] + offset, point[1] + offset, point[2] + offset,
                    name if name != 'Gamma' else 'Γ',
                    fontsize=10, fontweight='bold', color='darkred')
    
    # Set equal aspect ratio
    bbox_min, bbox_max = bz.get_bounding_box()
    max_range = np.max(bbox_max - bbox_min)
    center = (bbox_min + bbox_max) / 2
    
    ax.set_xlim(center[0] - max_range/2, center[0] + max_range/2)
    ax.set_ylim(center[1] - max_range/2, center[1] + max_range/2)
    ax.set_zlim(center[2] - max_range/2, center[2] + max_range/2)
    
    # Labels
    if show_axes_labels:
        ax.set_xlabel('$k_x$ (Å$^{-1}$)', fontsize=12)
        ax.set_ylabel('$k_y$ (Å$^{-1}$)', fontsize=12)
        ax.set_zlabel('$k_z$ (Å$^{-1}$)', fontsize=12)
    
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    elif bz.lattice.formula:
        ax.set_title(f'{bz.lattice.formula} - {bz.lattice.crystal_system.capitalize()} BZ',
                     fontsize=14, fontweight='bold')
    
    return fig, ax


def plot_bz_plotly(bz: BrillouinZone,
                   facecolor: str = 'lightblue',
                   opacity: float = 0.5,
                   show_hs_points: bool = True,
                   wireframe_color: str = 'white',
                   title: Optional[str] = None) -> 'go.Figure':
    """
    Plot Brillouin Zone using Plotly for interactive 3D visualization.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object to plot
    facecolor : str
        Face color for the mesh
    opacity : float
        Transparency (0-1)
    show_hs_points : bool
        Whether to show and label high-symmetry points
    wireframe_color : str
        Color for the wireframe edges (default: 'white')
    title : str, optional
        Plot title
        
    Returns
    -------
    go.Figure
        Plotly figure object
    """
    if not _HAS_PLOTLY:
        raise ImportError("Plotly is required for interactive plots. "
                          "Install with: pip install plotly")
    
    # Create mesh trace
    mesh = go.Mesh3d(
        x=bz.vertices[:, 0],
        y=bz.vertices[:, 1],
        z=bz.vertices[:, 2],
        i=bz.faces[:, 0],
        j=bz.faces[:, 1],
        k=bz.faces[:, 2],
        color=facecolor,
        opacity=opacity,
        flatshading=True,
        name='First BZ',
        hoverinfo='skip'
    )
    
    traces = [mesh]
    
    # Calculate unique edges for wireframe
    edges = set()
    for face in bz.faces:
        # Sort indices to ensure uniqueness of edge (i, j) vs (j, i)
        edges.add(tuple(sorted((face[0], face[1]))))
        edges.add(tuple(sorted((face[1], face[2]))))
        edges.add(tuple(sorted((face[2], face[0]))))
    
    # Create lines for edges
    xe, ye, ze = [], [], []
    for edge in edges:
        v1 = bz.vertices[edge[0]]
        v2 = bz.vertices[edge[1]]
        xe.extend([v1[0], v2[0], None])
        ye.extend([v1[1], v2[1], None])
        ze.extend([v1[2], v2[2], None])
        
    wireframe = go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode='lines',
        line=dict(color=wireframe_color, width=4),
        name='Edges',
        hoverinfo='skip',
        showlegend=False
    )
    traces.append(wireframe)
    
    # Add high-symmetry points
    if show_hs_points:
        hs_x, hs_y, hs_z = [], [], []
        hs_labels = []
        for name, point in bz.high_symmetry_points.items():
            hs_x.append(point[0])
            hs_y.append(point[1])
            hs_z.append(point[2])
            hs_labels.append('Γ' if name == 'Gamma' else name)
        
        hs_trace = go.Scatter3d(
            x=hs_x, y=hs_y, z=hs_z,
            mode='markers+text',
            marker=dict(size=6, color='red'),
            text=hs_labels,
            textposition='top center',
            name='High-symmetry points'
        )
        traces.append(hs_trace)
    
    fig = go.Figure(data=traces)
    
    # Set layout
    plot_title = title or f'{bz.lattice.formula or ""} {bz.lattice.crystal_system.capitalize()} BZ'
    fig.update_layout(
        title=plot_title,
        scene=dict(
            xaxis_title='k_x (Å⁻¹)',
            yaxis_title='k_y (Å⁻¹)',
            zaxis_title='k_z (Å⁻¹)',
            # Ensure aspect ratio is preserved
            xaxis=dict(nticks=5,),
            yaxis=dict(nticks=5,),
            zaxis=dict(nticks=5,),
            aspectmode='data'
        ),
        showlegend=True,
        margin=dict(l=0, r=0, t=50, b=0)
    )
    
    return fig


def plot_bz_pyvista(bz: BrillouinZone,
                    color: str = 'lightblue',
                    opacity: float = 0.6,
                    show_hs_points: bool = True,
                    show_edges: bool = True) -> Any:
    """
    Plot Brillouin Zone using PyVista for high-quality 3D rendering.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object
    color : str
        Mesh color
    opacity : float
        Transparency
    show_hs_points : bool
        Whether to show high-symmetry points
    show_edges : bool
        Whether to show mesh edges
        
    Returns
    -------
    pv.Plotter
        PyVista plotter object
    """
    if not _HAS_PYVISTA:
        raise ImportError("PyVista is required. Install with: pip install pyvista")
    
    # Convert faces to PyVista format: [n_verts, v0, v1, v2, ...]
    faces_flat = []
    for face in bz.faces:
        faces_flat.append(3)  # Number of vertices per face
        faces_flat.extend(face.tolist())
    
    mesh = pv.PolyData(bz.vertices, np.array(faces_flat))
    
    plotter = pv.Plotter()
    plotter.add_mesh(mesh, color=color, show_edges=show_edges, opacity=opacity)
    
    # Add high-symmetry points
    if show_hs_points:
        for name, point in bz.high_symmetry_points.items():
            plotter.add_point_labels(
                [point],
                ['Γ' if name == 'Gamma' else name],
                point_size=10,
                font_size=14,
                text_color='red'
            )
    
    plotter.add_axes()
    
    return plotter


def plot_bz(bz: BrillouinZone,
            backend: str = 'matplotlib',
            **kwargs) -> Any:
    """
    Plot Brillouin Zone using the specified backend.
    
    This is the main plotting function that dispatches to backend-specific
    implementations.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object to plot
    backend : str
        Plotting backend: 'matplotlib', 'plotly', or 'pyvista'
    **kwargs
        Additional arguments passed to the backend-specific function
        
    Returns
    -------
    Depends on backend:
        - matplotlib: (fig, ax) tuple
        - plotly: go.Figure
        - pyvista: pv.Plotter
        
    Examples
    --------
    >>> from brillouin_zone import generate_bz, plot_bz
    >>> from brillouin_zone.lattice import load_from_parameters
    >>> 
    >>> lat = load_from_parameters(3.0, 3.0, 3.0)
    >>> bz = generate_bz(lat)
    >>> 
    >>> # matplotlib (default)
    >>> fig, ax = plot_bz(bz)
    >>> plt.show()
    >>> 
    >>> # Interactive Plotly
    >>> fig = plot_bz(bz, backend='plotly')
    >>> fig.show()
    """
    backend_lower = backend.lower()
    
    if backend_lower == 'matplotlib':
        result = plot_bz_matplotlib(bz, **kwargs)
        plt.show()
        return result
    elif backend_lower == 'plotly':
        fig = plot_bz_plotly(bz, **kwargs)
        fig.show()
        return fig
    elif backend_lower == 'pyvista':
        plotter = plot_bz_pyvista(bz, **kwargs)
        plotter.show()
        return plotter
    else:
        raise ValueError(f"Unknown backend: {backend}. "
                         f"Choose from: 'matplotlib', 'plotly', 'pyvista'")


def plot_bz_3_views(bz: BrillouinZone,
                    figsize: Tuple[int, int] = (15, 5),
                    facecolor: str = 'cyan',
                    alpha: float = 0.3,
                    show_hs_points: bool = True,
                    title_prefix: str = "") -> Tuple[plt.Figure, List[plt.Axes]]:
    """
    Plot three orthogonal projections of the Brillouin Zone.
    
    Creates a figure with three panels showing the BZ from:
    - Front view (kx-kz plane, looking along ky)
    - Top view (kx-ky plane, looking along kz)
    - Side view (ky-kz plane, looking along kx)
    
    This matches the style of processing.visualization.plot_3d_data() for
    consistency with ARPES data visualization.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object
    figsize : tuple
        Figure size
    facecolor : str
        Face color for BZ
    alpha : float
        Transparency
    show_hs_points : bool
        Whether to show high-symmetry points
    title_prefix : str
        Prefix for subplot titles
        
    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : List[matplotlib.axes.Axes]
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize, subplot_kw={'projection': '3d'})
    
    # View angles: (elevation, azimuth)
    views = [
        (0, 0, 'Front (kx-kz)'),    # Looking along ky
        (90, -90, 'Top (kx-ky)'),   # Looking along kz
        (0, 90, 'Side (ky-kz)')     # Looking along kx
    ]
    
    for ax, (elev, azim, view_name) in zip(axes, views):
        plot_bz_matplotlib(
            bz, ax=ax,
            facecolor=facecolor,
            alpha=alpha,
            show_hs_points=show_hs_points,
            show_axes_labels=True,
            title=f"{title_prefix}{view_name}"
        )
        ax.view_init(elev=elev, azim=azim)
    
    plt.tight_layout()
    return fig, axes


def add_kpoints_to_bz(ax: plt.Axes,
                      k_points: np.ndarray,
                      color: str = 'blue',
                      size: int = 20,
                      alpha: float = 0.7,
                      label: Optional[str] = None) -> None:
    """
    Add k-points overlay to an existing BZ plot.
    
    Useful for overlaying experimental ARPES data or band structure k-paths.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        3D axes from plot_bz_matplotlib
    k_points : np.ndarray
        K-points to overlay, shape (N, 3) or (N, 2) for 2D slice
    color : str
        Marker color
    size : int
        Marker size
    alpha : float
        Transparency
    label : str, optional
        Legend label
    """
    if k_points.shape[1] == 2:
        # 2D points - assume kz=0
        k_3d = np.column_stack([k_points, np.zeros(len(k_points))])
    else:
        k_3d = k_points
    
    ax.scatter(k_3d[:, 0], k_3d[:, 1], k_3d[:, 2],
               c=color, s=size, alpha=alpha, label=label)
    
    if label:
        ax.legend()


def plot_kpath_on_bz(bz: BrillouinZone,
                     path_spec: List[str],
                     figsize: Tuple[int, int] = (8, 8),
                     path_color: str = 'red',
                     path_width: float = 2.0) -> Tuple[plt.Figure, plt.Axes]:
    """
    Plot a k-path through the Brillouin Zone.
    
    Parameters
    ----------
    bz : BrillouinZone
        Brillouin Zone object
    path_spec : List[str]
        Path specification, e.g., ['Gamma', 'X', 'M', 'Gamma']
    figsize : tuple
        Figure size
    path_color : str
        Color for the path line
    path_width : float
        Line width for path
        
    Returns
    -------
    fig, ax
    """
    fig, ax = plot_bz_matplotlib(bz, figsize=figsize)
    
    # Draw path segments
    points = bz.high_symmetry_points
    for i in range(len(path_spec) - 1):
        start = points[path_spec[i]]
        end = points[path_spec[i + 1]]
        ax.plot([start[0], end[0]], 
                [start[1], end[1]], 
                [start[2], end[2]],
                color=path_color, linewidth=path_width)
    
    return fig, ax


# Public API
__all__ = [
    'plot_bz',
    'plot_bz_matplotlib',
    'plot_bz_plotly',
    'plot_bz_pyvista',
    'plot_bz_3_views',
    'add_kpoints_to_bz',
    'plot_kpath_on_bz',
]
