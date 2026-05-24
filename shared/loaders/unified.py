"""
Unified data loading helpers for ADAPT.

Both the desktop Browser and web Edit app use this module so supported formats
and fallback behavior stay consistent.
"""

import os

from .load_adress_data import load as load_adress
from .load_sis_data import load_sis_data as load_sis
from .load_ses_zip import load as load_ses
from .load_ibw_data import load as load_ibw
from .load_pxt_data import load as load_pxt


SUPPORTED_EXTENSIONS = {
    "h5": "HDF5",
    "hdf5": "HDF5",
    "nxs": "HDF5",
    "ibw": "IBW",
    "zip": "ZIP",
    "pxt": "PXT",
    "pxp": "PXP",
}


def load_data_file(filepath: str):
    """
    Load an ARPES data file as an xarray.DataArray.

    HDF5-like files try the ADRESS loader first, then the SIS loader. This
    mirrors the historical Browser behavior and is shared by ADAPT Edit.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower().lstrip(".")

    if ext in ("h5", "hdf5", "nxs"):
        try:
            return load_adress(filepath)
        except Exception as adress_error:
            try:
                return load_sis(filepath)
            except Exception as sis_error:
                raise ValueError(
                    f"Failed to load HDF5 file.\nADRESS: {adress_error}\nSIS: {sis_error}"
                ) from sis_error

    if ext == "ibw":
        return load_ibw(filepath)
    if ext == "zip":
        return load_ses(filepath)
    if ext in ("pxt", "pxp"):
        return load_pxt(filepath)

    raise ValueError(f"Unsupported file type: {ext}")


def is_supported_file(filepath: str) -> bool:
    """Check whether a path has a supported ADAPT data extension."""
    ext = os.path.splitext(filepath)[1].lower().lstrip(".")
    return ext in SUPPORTED_EXTENSIONS


def get_file_type(filepath: str):
    """Return a user-facing file type label for a supported file."""
    ext = os.path.splitext(filepath)[1].lower().lstrip(".")
    return SUPPORTED_EXTENSIONS.get(ext)
