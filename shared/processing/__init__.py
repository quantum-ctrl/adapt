"""
ADAPT Processing Module

Data processing and analysis tools for ARPES data.
"""

# Robust imports to handle missing dependencies (e.g. cv2, scipy)

# Fermi Edge Fit (needs scipy)
try:
    from .fermi_edge_fit import (
        fit_fermi_edge,
        fit_fermi_edge_3d,
        plot_fit_results,
        plot_batch_fit_results,
        print_fit_summary
    )
except ImportError:
    # Define stubs or just skip
    pass

# Align (needs xarray, numpy)
try:
    from .align import (
        align_axis,
        align_energy,
        align_energy_3d
    )
except ImportError:
    pass

# Visualization (needs matplotlib)
try:
    from .visualization import (
        plot_2d_data,
        plot_3d_data
    )
except ImportError:
    pass

# Interactive - Commented out as missing
# from .interactive import (
#     InteractiveCrosshair2D,
#     InteractiveCrosshair3D,
#     interactive_plot_2d,
#     interactive_plot_3d
# )

# Geometry (needs cv2)
try:
    from .enhancement.geometry import (
        get_slice
    )
except ImportError:
    pass

# Dimension
try:
    from .dimension import (
        reduce_to_2d,
        merge_to_3d,
        interpolate_to_common_grid,
        interpolate_to_common_grid,
        crop,
        normalize_slices
    )
except ImportError:
    pass

# Edit (needs xarray)
try:
    from .edit import crop_data
except ImportError:
    pass

# Angle to K
try:
    from .angle_to_k import (
        convert_angle_to_k,
        convert_2d_angle_to_k,
        convert_3d_kxky_to_k,
        convert_3d_hv_to_k,
        convert_hv_to_kxkz,
    )
except ImportError:
    pass

# Artifacts
try:
    from .artifacts import (
        remove_hot_pixels,
        remove_dead_pixels,
        destripe_horizontal,
        destripe_vertical,
        remove_salt_pepper_noise
    )
except ImportError:
    pass

# Background
try:
    from .background import (
        shirley_background,
        snip_background,
        poly_background
    )
except ImportError:
    pass

# Analysis Peaks
try:
    from .analysis.peaks import (
        detect_peaks_1d,
        fit_peak_lorentzian,
        extract_mdc_peaks,
        extract_edc_peaks,
        extract_band_dispersion
    )
except ImportError:
    pass

# Analysis Band Extraction
try:
    from .analysis.band_extraction import (
        extract_bands,
        export_to_svg,
        export_to_pdf,
        extract_and_export,
        overlay_bands_on_data
    )
except ImportError:
    pass

__all__ = [
    # Fermi edge fitting
    'fit_fermi_edge',
    'fit_fermi_edge_3d',
    'plot_fit_results',
    'plot_batch_fit_results',
    'print_fit_summary',
    # Alignment
    'align_axis',
    'align_energy',
    'align_energy_3d',
    # Visualization
    'plot_2d_data',
    'plot_3d_data',
    # Interactive visualization
    'InteractiveCrosshair2D',
    'InteractiveCrosshair3D',
    'interactive_plot_2d',
    'interactive_plot_3d',
    'get_slice',
    # Dimension manipulation
    'reduce_to_2d',
    'merge_to_3d',
    'interpolate_to_common_grid',
    'crop',
    'normalize_slices',
    # Momentum conversion
    'convert_angle_to_k',
    'convert_2d_angle_to_k',
    'convert_3d_kxky_to_k',
    'convert_3d_hv_to_k',
    'convert_hv_to_kxkz',
    # Artifact removal
    'remove_hot_pixels',
    'remove_dead_pixels',
    'destripe_horizontal',
    'destripe_vertical',
    'remove_salt_pepper_noise',
    # Background subtraction
    'shirley_background',
    'snip_background',
    'poly_background',
    # Peak detection and band dispersion
    'detect_peaks_1d',
    'fit_peak_lorentzian',
    'extract_mdc_peaks',
    'extract_edc_peaks',
    'extract_band_dispersion',
    # Band extraction and vector export
    'extract_bands',
    'export_to_svg',
    'export_to_pdf',
    'extract_and_export',
    'overlay_bands_on_data',
]






