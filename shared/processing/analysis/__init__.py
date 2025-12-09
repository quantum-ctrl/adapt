"""
ADAPT Processing Analysis Submodule

Peak detection and band dispersion extraction for ARPES data.
"""

from .peaks import (
    detect_peaks_1d,
    fit_peak_lorentzian,
    extract_mdc_peaks,
    extract_edc_peaks,
    extract_band_dispersion
)

from .band_extraction import (
    preprocess_data,
    extract_bands,
    export_to_svg,
    export_to_pdf,
    overlay_bands_on_data,
    extract_and_export
)

__all__ = [
    'detect_peaks_1d',
    'fit_peak_lorentzian',
    'extract_mdc_peaks',
    'extract_edc_peaks',
    'extract_band_dispersion',
    # Band extraction
    'preprocess_data',
    'extract_bands',
    'export_to_svg',
    'export_to_pdf',
    'overlay_bands_on_data',
    'extract_and_export'
]

