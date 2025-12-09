"""
ADAPT Image Enhancement Module

Scientific image enhancement tools for microscopy, spectroscopy, and materials data.
Supports Python scripts and Jupyter Notebooks.

Usage:
    from processing.enhancement import sharpen, histogram_equalize, clahe
    
    # Sharpen
    sharpened = sharpen(image, strength=1.5)
    
    # Enhance contrast
    enhanced = histogram_equalize(sharpened)
    
    # CLAHE (adaptive contrast enhancement)
    clahe_result = clahe(image, clip_limit=0.01)
"""

from .filter import (
    sharpen,
    edge_enhance
)

from .intensity import (
    histogram_equalize,
    clahe,
    adjust_contrast,
    gamma_correction
)

from .geometry import (
    rotate,
    flip,
    scale
)

from .curvature import (
    curvature_luo,
    curvature_second_derivative,
    auto_curvature
)

__all__ = [
    # Filtering
    'sharpen',
    'edge_enhance',
    # Intensity
    'histogram_equalize',
    'clahe',
    'adjust_contrast',
    'gamma_correction',
    # Geometry
    'rotate',
    'flip',
    'scale',
    # Curvature Analysis
    'curvature_luo',
    'curvature_second_derivative',
    'auto_curvature'
]

__version__ = '1.0.0'

