import os
import re
import numpy as np
import xarray as xr
import igor2.binarywave as ibw
from datetime import datetime
from typing import Dict, Any, Optional, List

def _parse_igor_note(note_content: str) -> Dict[str, str]:
    """
    Parses the Igor wave note which is formatted as key=value pairs.
    """
    data = {}
    if not note_content:
        return data
        
    for line in note_content.splitlines():
        line = line.strip()
        if not line:
            continue
        # Check for [Section] headers, we might ignore them or store them?
        # For now, we just look for key=value
        if "=" in line:
            parts = line.split("=", 1)
            key = parts[0].strip()
            value = parts[1].strip()
            data[key] = value
    return data

def _safe_float(value: Any, default: float = 0.0) -> float:
    """Safely converts a value to float."""
    if value is None:
        return default
    try:
        return float(str(value).replace(",", "."))
    except (ValueError, TypeError):
        return default

def load(path: str) -> xr.DataArray:
    """
    Loads an Igor Packed Experiment (.pxt) file (containing an SES binary wave) 
    and returns an xarray.DataArray.
    
    Compatible with ADAPT-lab data structure.
    
    Args:
        path (str): Absolute path to the .pxt file.
        
    Returns:
        xr.DataArray: The loaded data.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    # Attempt to load using igor2 library
    # Try loading as packed experiment first if extension is .pxt or .pxp, or fallback
    try:
        # Check extensions or just try both
        if path.lower().endswith('.pxt') or path.lower().endswith('.pxp'):
             import igor2.packed
             data = igor2.packed.load(path)
             # data is (records, filesystem)
             records, _ = data
             if not records:
                 raise ValueError("No waves found in packed experiment.")
             # Assume the first record is the main data
             # WaveRecord.wave is {'version': ..., 'wave': ...}
             wave = records[0].wave['wave']
        else:
             data = ibw.load(path)
             wave = data['wave']
    except Exception as e:
        # If failed, try the other method as fallback
        try:
             import igor2.packed
             data = igor2.packed.load(path)
             records, _ = data
             if not records:
                 raise ValueError("No waves found in packed experiment.")
             wave = records[0].wave['wave']
        except Exception:
             # Try ibw as last resort if not tried yet or just re-raise original
             try:
                 data = ibw.load(path)
                 wave = data['wave']
             except:
                 raise IOError(f"Failed to load file with igor2: {e}")

    w_data = wave['wData']
    header = wave['wave_header']
    note = wave.get('note', b'')
    if isinstance(note, bytes):
        note = note.decode('utf-8', errors='ignore')
    
    # Parse the note for metadata
    # The note contains keys like 'Excitation Energy', 'Pass Energy', etc.
    meta_dict = _parse_igor_note(note)
    
    # --- Dimensions and Axes ---
    # According to 'pxt loader.txt', we should prioritize the actual data dimensions (sfA, sfB)
    # over the header text for scaling.
    
    shape = w_data.shape
    ndim = w_data.ndim
    
    # sfA is scale (delta), sfB is offset (start)
    sfA = header.get('sfA', [1.0] * ndim)
    sfB = header.get('sfB', [0.0] * ndim)
    
    coords = {}
    dims: List[str] = []
    
    # Dimension 0: Energy (usually rows in Igor, axis 0 in numpy from igor2)
    # Check 'Energy Scale' (Kinetic vs Binding) from note
    energy_scale_mode = meta_dict.get("Energy Scale", "Kinetic")
    hv = _safe_float(meta_dict.get("Excitation Energy"))
    
    # Construct generic axes first
    axes = []
    for i in range(ndim):
        n = shape[i]
        start = sfB[i]
        delta = sfA[i]
        axis = start + np.arange(n) * delta
        axes.append(axis)
        
    # Map to Energy, Angle, Scan
    # Usually: Dim 0 = Energy, Dim 1 = Angle, Dim 2 = Scan (if 3D)
    
    if ndim >= 1:
        dims.append("energy")
        energy_axis = axes[0]
        
        # 'pxt loader.txt': "Data in *.pxt can be saved with either KE or +BE scales"
        # "Axis Units is "eV" for both KE and BE scaling"
        # We respect the file's axis values (sfA/sfB).
        # Labeling depends on 'Energy Scale'.
        
        energy_long_name = "Binding energy" if energy_scale_mode == "Binding" else "Kinetic energy"
        coords["energy"] = (["energy"], energy_axis, {"units": "eV", "long_name": energy_long_name})
        
    if ndim >= 2:
        dims.append("angle")
        # Usually Analyzer Angle
        angle_axis = axes[1]
        coords["angle"] = (["angle"], angle_axis, {"units": "degree", "long_name": "Analyzer angle"})
        
    if ndim >= 3:
        dims.append("scan")
        # Usually Deflector Angle or Z scan
        scan_axis = axes[2]
        coords["scan"] = (["scan"], scan_axis, {"units": "degree", "long_name": "Deflector angle"})

    # --- Metadata Construction ---
    attrs = {
        "FileName": os.path.basename(path),
        "H5file": path,
        "TimeStamp": "",
        "Type": "",
        "hv": hv,
        "meta": meta_dict
    }
    
    # Try to parse timestamps
    date_str = meta_dict.get("Date", "")
    time_str = meta_dict.get("Time", "")
    if date_str and time_str:
        attrs["TimeStamp"] = f"{date_str} {time_str}"
    else:
        try:
            mtime = os.path.getmtime(path)
            attrs["TimeStamp"] = datetime.fromtimestamp(mtime).isoformat()
        except Exception:
            pass

    # Populate standard attributes expected by other tools
    # 'load_ses_zip.py' maps many fields to 'meta' and some to top-level attrs.
    
    # Pass Energy
    attrs["meta"]["Epass"] = _safe_float(meta_dict.get("Pass Energy"))
    
    # Lens Mode
    attrs["meta"]["Mode"] = meta_dict.get("Lens Mode", "")
    
    # Angles (Manipulator)
    # pxt loader.txt does not explicitly map these standard keys to standard Manipulator T/P/A,
    # but they might be present in the note anyway like in SES zip.
    # We copy them if found.
    # Common keys: 'Polar', 'Azimuth', 'Tilt' or 'Theta', 'Phi'
    # From 'pxt loader.txt' ReadSEShdr:
    # Polar = StringByKey("Polar", ...)
    # Azimuth = StringByKey("Azimuth", ...)
    # Temp_B = StringByKey("Temperature Sensor B", ...)
    
    attrs["tltE"] = _safe_float(meta_dict.get("Polar", 0)) # Polar -> tltE (usually)
    attrs["tltM"] = _safe_float(meta_dict.get("Azimuth", 0)) # Azimuth -> tltM (or similar) or Deflector if 3D
    attrs["Temp"] = _safe_float(meta_dict.get("Temperature Sensor B", 0))
    
    # Determine 'Type'
    if ndim == 3:
        attrs["Type"] = "Eb(kx,ky)"
        # If 3D, 'scan' axis usually covers the second angular dimension
    elif ndim == 2:
        attrs["Type"] = "Eb(k)"
    else:
        attrs["Type"] = "Spec"

    # Create DataArray
    da = xr.DataArray(
        data=w_data,
        coords=coords,
        dims=tuple(dims),
        attrs=attrs,
        name="intensity"
    )
    
    return da

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
        try:
            da = load(fpath)
            print(da)
            print(da.attrs)
        except Exception as e:
            print(f"Error: {e}")
