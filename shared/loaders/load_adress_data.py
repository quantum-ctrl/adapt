#!/usr/bin/env python3
"""
Python loader for ADRESS beamline data (SLS).
Converted from load_adress_data.m.
"""

import os
import re
import numpy as np
import h5py
import xarray as xr
from typing import Optional, Dict, Any, Tuple, Union

def _extract_note_value(note: str, key: str, value_type: str = 'float') -> Union[float, str, None]:
    """
    Parses the text note using regex to find values, replicating MATLAB logic.
    
    Args:
        note: The full note string.
        key: The key to search for (e.g., "hv").
        value_type: 'float' or 'string'.
        
    Returns:
        The extracted value or None/NaN if not found.
    """
    if not note:
        return np.nan if value_type == 'float' else ""

    if value_type == 'float':
        # Pattern handles standard floats and the 'ones(x)*' prefix logic found in MATLAB code
        # MATLAB: hv = str2double(Note(tmpPos+9:tmpPos+13-i)); etc.
        # We use regex to be more robust but match the intent.
        # Matches: key = value, key = ones(N)*value
        pattern = rf"{key}\s*=\s*(?:ones\(\d+\)\*)?([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)"
        match = re.search(pattern, note, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, TypeError):
                return np.nan
        return np.nan
        
    elif value_type == 'string':
        # Grab rest of line or specific words
        pattern = rf"{key}\s*=\s*(.+)"
        match = re.search(pattern, note, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
    
    return None

def _parse_metadata_from_note(note: str) -> Dict[str, Any]:
    """
    Replicates metadata parsing from load_adress_data.m.
    """
    meta = {}
    
    # Epass (Pass Energy)
    epass = _extract_note_value(note, "Epass")
    meta['Epass'] = epass if not np.isnan(epass) else None
    
    # Derived resolutions
    meta['dhv'] = 75e-3
    meta['deb'] = 0.5 * epass / 1000.0 if epass and not np.isnan(epass) else None
    
    # Theta, Temp, ADef (Electrostatic Tilt), Slit, X, Y, Z
    meta['Theta'] = _extract_note_value(note, "Theta")
    meta['Temp'] = _extract_note_value(note, "Temp")
    meta['ADef'] = _extract_note_value(note, "ADef")
    meta['Slit'] = _extract_note_value(note, "Slit")
    meta['X'] = _extract_note_value(note, "X")
    meta['Y'] = _extract_note_value(note, "Y")
    meta['Z'] = _extract_note_value(note, "Z")
    
    # Polarization Mapping
    pol_raw = _extract_note_value(note, "Pol", 'string')
    if pol_raw:
        if "LV" in pol_raw: meta['Pol'] = "LV (p-pol)"
        elif "LH" in pol_raw: meta['Pol'] = "LH (s-pol)"
        elif "C+" in pol_raw: meta['Pol'] = "C+"
        elif "C-" in pol_raw: meta['Pol'] = "C-"
        else: meta['Pol'] = ""
    else:
        meta['Pol'] = ""
        
    # Mode Mapping
    mode_raw = _extract_note_value(note, "Mode", 'string')
    if mode_raw:
        if "MAD" in mode_raw: meta['Mode'] = "MAD"
        elif "MAM" in mode_raw: meta['Mode'] = "MAM"
        elif "LAD" in mode_raw: meta['Mode'] = "LAD"
        elif "WAM" in mode_raw: meta['Mode'] = "WAM"
        else: meta['Mode'] = ""
    else:
        meta['Mode'] = ""
    
    return meta

def _find_data_and_axes(h5_file: h5py.File) -> Tuple[np.ndarray, Dict[str, Any], str]:
    """
    Locates the main data array and extracts axis information.
    Replicates ReaderHDF5 behavior implicitly by finding the largest numeric dataset
    and looking for IGOR metadata.
    """
    # Common patterns for data
    common_patterns = [
        ['data', 'intensity', 'counts', 'signal'],
        ['arpes', 'spectrum', 'image'],
        ['measurement', 'scan', 'result']
    ]
    
    intensity_data = None
    data_key = None
    
    # 1. Try to find main data array
    for pattern_group in common_patterns:
        for key in h5_file.keys():
            if any(pattern in key.lower() for pattern in pattern_group):
                try:
                    # Check if it's a dataset
                    if isinstance(h5_file[key], h5py.Dataset):
                        intensity_data = np.array(h5_file[key])
                        data_key = key
                        break
                except Exception:
                    continue
        if intensity_data is not None:
            break
            
    # 2. If not found, search recursively for largest numeric dataset
    if intensity_data is None:
        max_size = 0
        
        def search_group(group, path=""):
            nonlocal intensity_data, data_key, max_size
            for key in group.keys():
                item = group[key]
                current_path = f"{path}/{key}" if path else key
                
                if isinstance(item, h5py.Dataset):
                    if item.size > max_size and np.issubdtype(item.dtype, np.number):
                        # Heuristic: ARPES data is usually 2D or 3D
                        if len(item.shape) >= 2:
                            intensity_data = np.array(item)
                            data_key = current_path
                            max_size = item.size
                elif isinstance(item, h5py.Group):
                    search_group(item, current_path)
        
        search_group(h5_file)

    if intensity_data is None:
        raise ValueError("Could not find valid data array in HDF5 file")

    # 3. Extract Axes from IGOR metadata
    # IGOR Wave Scaling: [unused, Y-axis (angle), X-axis (energy), Z-axis (scan)]
    # Note: This maps to the RAW data dimensions in the HDF5 file.
    axes_info = {}
    data_item = h5_file[data_key]
    
    if 'IGORWaveScaling' in data_item.attrs:
        scaling = data_item.attrs['IGORWaveScaling']
        # scaling[1] -> dim 0 (Angle)
        # scaling[2] -> dim 1 (Energy)
        # scaling[3] -> dim 2 (Scan)
        
        if len(scaling) >= 2:
            axes_info['dim0'] = {'delta': scaling[1][0], 'start': scaling[1][1]} # Angle
        if len(scaling) >= 3:
            axes_info['dim1'] = {'delta': scaling[2][0], 'start': scaling[2][1]} # Energy
        if len(scaling) >= 4:
            axes_info['dim2'] = {'delta': scaling[3][0], 'start': scaling[3][1]} # Scan

    return intensity_data, axes_info, data_key

def _create_axis(info: Dict[str, float], n_points: int) -> np.ndarray:
    """Creates an axis array from start/delta info."""
    if not info:
        return np.arange(n_points)
    start = info['start']
    delta = info['delta']
    if delta == 0:
        return np.full(n_points, start)
    return np.linspace(start, start + delta * (n_points - 1), n_points)

def load(path: str) -> xr.DataArray:
    """
    Loads ADRESS beamline HDF5 data into an xarray.DataArray.
    
    Args:
        path: Path to the .h5 file.
        
    Returns:
        xarray.DataArray with dimensions (energy, angle, [scan]) and metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    with h5py.File(path, 'r') as f:
        # 1. Get Data and Axes Info
        raw_data, axes_info, data_key = _find_data_and_axes(f)
        
        # 2. Get Note (try file attrs, top-level dataset, then dataset attrs like IGORWaveNote)
        note = ""
        # 2a: file attribute 'note'
        if 'note' in f.attrs:
            raw_note = f.attrs['note']
            if isinstance(raw_note, (bytes, np.bytes_)):
                note = raw_note.decode('utf-8', errors='ignore')
            else:
                note = str(raw_note)
        # 2b: top-level dataset 'note'
        elif 'note' in f:
            try:
                note_ds = f['note']
                if isinstance(note_ds, h5py.Dataset):
                    arr = np.array(note_ds)
                    # arr may be bytes, array of bytes, or string
                    if isinstance(arr, (bytes, np.bytes_)):
                        note = arr.decode('utf-8', errors='ignore')
                    else:
                        try:
                            # Join if array of strings/bytes
                            if hasattr(arr, 'dtype') and arr.dtype.kind in ('S', 'U'):
                                if arr.dtype.kind == 'S':
                                    note = ''.join(x.decode('utf-8', errors='ignore') if isinstance(x, (bytes, np.bytes_)) else str(x) for x in arr.ravel())
                                else:
                                    note = ''.join(str(x) for x in arr.ravel())
                            else:
                                note = str(arr)
                        except:
                            note = str(arr)
            except:
                pass

        # 2c: check dataset attributes for IGOR-like note fields (e.g., IGORWaveNote)
        if not note and data_key in f:
            try:
                data_item = f[data_key]
                for attr_name, attr_val in data_item.attrs.items():
                    lname = str(attr_name).lower()
                    if 'note' in lname or 'igor' in lname:
                        raw = attr_val
                        if isinstance(raw, (bytes, np.bytes_)):
                            note = raw.decode('utf-8', errors='ignore')
                        else:
                            note = str(raw)
                        break
            except Exception:
                pass
                
        # 3. Permute Data (MATLAB Logic)
        # MATLAB: 
        # if ndims(Data)==3; Data = double(permute(Data,[2 1 3])); else Data=double(Data'); end
        # Python (0-indexed): 
        # 3D: [2, 1, 3] in MATLAB (1-based) -> [1, 0, 2] in Python
        # 2D: Transpose -> [1, 0]
        
        if raw_data.ndim == 3:
            data = np.transpose(raw_data, (1, 0, 2))
            # New dims: 0=Energy (was 1), 1=Angle (was 0), 2=Scan (was 2)
            
            # Map axes info to new dimensions
            # axes_info['dim0'] was Angle (now dim 1)
            # axes_info['dim1'] was Energy (now dim 0)
            # axes_info['dim2'] was Scan (now dim 2)
            
            energy_axis = _create_axis(axes_info.get('dim1'), data.shape[0])
            angle_axis = _create_axis(axes_info.get('dim0'), data.shape[1])
            scan_axis = _create_axis(axes_info.get('dim2'), data.shape[2])
            
        elif raw_data.ndim == 2:
            data = raw_data.T
            # New dims: 0=Energy (was 1), 1=Angle (was 0)
            
            energy_axis = _create_axis(axes_info.get('dim1'), data.shape[0])
            angle_axis = _create_axis(axes_info.get('dim0'), data.shape[1])
            scan_axis = np.array([]) # Empty for 2D
            
        else:
            # Fallback for unexpected dims
            data = raw_data
            energy_axis = np.arange(data.shape[0])
            angle_axis = np.arange(data.shape[1]) if data.ndim > 1 else np.array([])
            scan_axis = np.array([])

        # 4. Identify Type
        # MATLAB:
        # if size(Scan, 1) == 0; Type = "Eb(k)";
        # elseif range(Scan) == 0; Type = "Eb(k,i)";
        # elseif max(Scan(:)) > 100; Type = "Eb(kx,kz)";
        # else; Type = "Eb(kx,ky)";
        
        scan_type = "Eb(k)"
        if data.ndim == 3:
            scan_range = np.ptp(scan_axis) if scan_axis.size > 0 else 0
            scan_max = np.max(scan_axis) if scan_axis.size > 0 else 0
            
            if scan_axis.size == 0:
                scan_type = "Eb(k)"
            elif scan_range == 0:
                scan_type = "Eb(k,i)"
            elif scan_max > 100:
                scan_type = "Eb(kx,kz)"
            else:
                scan_type = "Eb(kx,ky)"
        
        # 5. Parse Metadata (hv, Tilt) based on Type
        hv = np.nan
        tilt = np.nan
        
        if scan_type in ["Eb(k)", "Eb(k,i)"]:
            hv = _extract_note_value(note, "hv")
            if np.isnan(hv): # Try fallback parsing logic if needed, but regex should handle it
                 pass 
            
            tilt = _extract_note_value(note, "Tilt")
            
        elif scan_type == "Eb(kx,kz)":
            # MATLAB: hv = Scan; Tilt = ...
            # We can't easily set hv to an array in a scalar field, 
            # but we can store it in the coords if it varies.
            # For the scalar attribute 'hv', we might leave it as NaN or the first value.
            # However, the requirement is to map fields.
            # Let's store the scalar extracted from note as a fallback, 
            # but acknowledge that for this mode, hv varies.
            
            # Actually, MATLAB sets dataStr.hv = Scan (vector).
            # We will handle this by putting it in coords or attrs.
            # For the single scalar 'hv' attr, we'll use the note value if available.
            hv = scan_axis # This will be an array
            
            tilt = _extract_note_value(note, "Tilt")
            
        elif scan_type == "Eb(kx,ky)":
            hv = _extract_note_value(note, "hv")
            # MATLAB: Tilt = Scan;
            tilt = scan_axis # This will be an array

        # 6. Parse General Metadata
        meta_info = _parse_metadata_from_note(note)
        
        # 7. Construct xarray
        
        # Prepare Coords and Dims
        if data.ndim == 3:
            dims = ("energy", "angle", "scan")
            coords = {
                "energy": energy_axis,
                "angle": angle_axis,
                "scan": scan_axis
            }
        else:
            dims = ("energy", "angle")
            coords = {
                "energy": energy_axis,
                "angle": angle_axis
            }
            
        # Prepare Attributes
        file_info = os.stat(path)
        # TimeStamp in MATLAB is from dir(), we can use os.stat or try to find it in file attrs
        timestamp = ""
        if 'TimeStamp' in f.attrs:
             timestamp = f.attrs['TimeStamp']
             if isinstance(timestamp, bytes): timestamp = timestamp.decode('utf-8')

        attrs = {
            "FileName": os.path.splitext(os.path.basename(path))[0],
            "H5file": os.path.basename(path),
            "TimeStamp": timestamp,
            "Type": scan_type,
            "hv": hv,
            "tltM": tilt,
            "tltE": meta_info.get('ADef'),
            "thtM": meta_info.get('Theta'),
            "Temp": meta_info.get('Temp'),
            "meta": meta_info,
            "raw_shape": raw_data.shape,
            "raw_note": note
        }
        
        # Create DataArray
        da = xr.DataArray(
            data=data,
            dims=dims,
            coords=coords,
            attrs=attrs,
            name="intensity"
        )
        
        return da

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        try:
            da = load(sys.argv[1])
            print(da)
        except Exception as e:
            print(f"Error loading file: {e}")
    else:
        print("Usage: python load_adress_data.py <path_to_h5_file>")
