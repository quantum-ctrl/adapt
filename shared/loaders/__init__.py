"""
ADAPT Shared Loaders Package.

This package provides unified data loaders for various ARPES data formats:
- ADRESS beamline (SLS) - HDF5
- SIS beamline - HDF5
- SES (Scienta Energy Series) - ZIP archives
- Igor Binary Wave - IBW

All loaders return xarray.DataArray with consistent dimension naming.
"""

from .load_adress_data import load as load_adress
from .load_sis_data import load_sis_data as load_sis
from .load_ses_zip import load as load_ses
from .load_ibw_data import load as load_ibw
from .load_pxt_data import load as load_pxt
from .unified import SUPPORTED_EXTENSIONS, get_file_type, is_supported_file, load_data_file

__all__ = [
    "load_adress",
    "load_sis", 
    "load_ses",
    "load_ibw",
    "load_pxt",
    "SUPPORTED_EXTENSIONS",
    "get_file_type",
    "is_supported_file",
    "load_data_file",
]
