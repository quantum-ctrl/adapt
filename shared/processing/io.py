"""
I/O helpers for processed ADAPT data.
"""

import os
import tempfile
from typing import List, Optional

import h5py
import xarray as xr


def save_processed_dataarray(
    data: xr.DataArray,
    prefix: str,
    temp_files: Optional[List[str]] = None,
    suffix: str = ".h5",
) -> str:
    """
    Save a processed DataArray to a temporary HDF5-compatible file.

    The xarray path keeps rich metadata when possible. The h5py fallback writes
    a simple layout understood by the existing ADRESS fallback loader.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix=prefix) as tmp:
        save_path = tmp.name

    if temp_files is not None:
        temp_files.append(save_path)

    try:
        data.to_netcdf(save_path, engine="h5netcdf")
    except Exception:
        with h5py.File(save_path, "w") as f:
            f.create_dataset("data", data=data.values)
            for coord in data.coords:
                f.create_dataset(coord, data=data.coords[coord].values)
            for key, value in data.attrs.items():
                try:
                    f.attrs[key] = value
                except (TypeError, ValueError, OSError):
                    f.attrs[key] = str(value)

    return os.path.abspath(save_path)
