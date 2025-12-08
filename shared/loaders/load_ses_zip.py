import os
import zipfile
import numpy as np
import xarray as xr
from datetime import datetime
from typing import Dict, Any, Tuple, Optional

def _parse_ini_content(content: str) -> Dict[str, str]:
    """Parses the content of an INI file into a dictionary."""
    data = {}
    for line in content.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data

def _safe_float(value: str) -> float:
    """Safely converts a string to float, handling comma decimals."""
    try:
        return float(value.replace(",", "."))
    except (ValueError, TypeError):
        return 0.0

def load(path: str) -> xr.DataArray:
    """
    Loads an SES zip file containing ARPES data.

    Args:
        path: Absolute path to the .zip file.

    Returns:
        xarray.DataArray: The loaded data with dimensions (energy, angle, scan) and metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with zipfile.ZipFile(path, 'r') as z:
        file_names = z.namelist()
        
        # Find the binary file to identify the region name
        bin_file_name = next((n for n in file_names if n.startswith('Spectrum_') and n.endswith('.bin')), None)
        
        if not bin_file_name:
            raise ValueError("Could not find a 'Spectrum_*.bin' file in the zip archive.")
            
        region_name = bin_file_name.replace('Spectrum_', '').replace('.bin', '')
        
        ini_spectrum_name = f"Spectrum_{region_name}.ini"
        ini_region_name = f"{region_name}.ini"
        
        if ini_spectrum_name not in file_names:
             raise ValueError(f"Could not find '{ini_spectrum_name}' in the zip archive.")
        if ini_region_name not in file_names:
             raise ValueError(f"Could not find '{ini_region_name}' in the zip archive.")

        # Read and parse INI files
        with z.open(ini_spectrum_name) as f:
            spectrum_ini = _parse_ini_content(f.read().decode('utf-8', errors='ignore'))
            
        with z.open(ini_region_name) as f:
            region_ini = _parse_ini_content(f.read().decode('utf-8', errors='ignore'))

        # Read Binary Data
        with z.open(bin_file_name) as f:
            raw_bytes = f.read()
            # MATLAB: fread(fid, inf, 'float32')
            flat_data = np.frombuffer(raw_bytes, dtype=np.float32)

    # Extract Axis Information from Spectrum INI
    # Energy Axis
    low_energy = _safe_float(spectrum_ini.get("widthoffset", "0"))
    num_energy = int(_safe_float(spectrum_ini.get("width", "0")))
    energy_step = _safe_float(spectrum_ini.get("widthdelta", "0"))
    
    # Analyzer Angle Axis (theta)
    low_analyzer = _safe_float(spectrum_ini.get("heightoffset", "0"))
    num_analyzer = int(_safe_float(spectrum_ini.get("height", "0")))
    analyzer_step = _safe_float(spectrum_ini.get("heightdelta", "0"))
    
    # Deflector Angle Axis (phi/tilt)
    low_deflector = _safe_float(spectrum_ini.get("depthoffset", "0"))
    num_deflector = int(_safe_float(spectrum_ini.get("depth", "0")))
    deflector_step = _safe_float(spectrum_ini.get("depthdelta", "0"))

    # Construct Axes
    # MATLAB: linspace(start, start + (n-1)*step, n)
    # Note: Python's linspace includes endpoint, so this logic matches.
    energy_axis = np.linspace(low_energy, low_energy + (num_energy - 1) * energy_step, num_energy)
    analyzer_axis = np.linspace(low_analyzer, low_analyzer + (num_analyzer - 1) * analyzer_step, num_analyzer)
    deflector_axis = np.linspace(low_deflector, low_deflector + (num_deflector - 1) * deflector_step, num_deflector)

    # Reshape Data
    # MATLAB Logic:
    # Data = zeros(numel(Energy), numel(Angle), numel(Tilt));
    # for i = 1:numel(Angle)
    #     idx = (i-1)*numel(Energy); 
    #     for j = 1:numel(Tilt)
    #        Data(:,i,j) = binData((j-1)*numel(Angle)*numel(Energy) + idx + (1:numel(Energy))); 
    #     end
    # end
    #
    # Let E=num_energy, A=num_analyzer, T=num_deflector.
    # Flat index k = (j-1)*A*E + (i-1)*E + (0 to E-1)
    # This corresponds to C-order shape (T, A, E) where T is slowest, E is fastest.
    # So we reshape to (num_deflector, num_analyzer, num_energy).
    
    if flat_data.size != num_energy * num_analyzer * num_deflector:
        # Fallback or error if sizes don't match
        raise ValueError(f"Data size mismatch. Expected {num_energy}*{num_analyzer}*{num_deflector} = {num_energy*num_analyzer*num_deflector}, got {flat_data.size}")

    data_reshaped = flat_data.reshape((num_deflector, num_analyzer, num_energy))
    
    # MATLAB returns (Energy, Angle, Tilt).
    # We will return (Eb, theta, phi) corresponding to (Energy, Analyzer, Deflector).
    # So we need to transpose (T, A, E) -> (E, A, T).
    data_final = data_reshaped.transpose((2, 1, 0))

    # Extract Metadata
    # Region INI mapping
    meta = {}
    
    # Direct mappings
    pass_energy = _safe_float(region_ini.get("Pass Energy", "0"))
    lens_mode = region_ini.get("Lens Mode", "")
    excitation_energy = _safe_float(region_ini.get("Excitation Energy", "0"))
    energy_scale = region_ini.get("Energy Scale", "")
    date_str = region_ini.get("Date", "")
    time_str = region_ini.get("Time", "")
    comments = region_ini.get("Comments", "")
    
    # Manipulator
    manipulator_tilt = _safe_float(region_ini.get("T", "0")) # T=
    manipulator_polar = _safe_float(region_ini.get("P", "0")) # P=
    manipulator_azimuth = _safe_float(region_ini.get("A", "0")) # A=
    manipulator_x = _safe_float(region_ini.get("X", "0"))
    manipulator_y = _safe_float(region_ini.get("Y", "0"))
    manipulator_z = _safe_float(region_ini.get("Z", "0"))

    # Store remaining unknown fields in meta
    known_keys = {
        "Pass Energy", "Lens Mode", "Excitation Energy", "Energy Scale", 
        "Date", "Time", "Comments", "T", "P", "A", "X", "Y", "Z",
        "Region Name", "Number of Sweeps", "Acquisition Mode", "Energy Unit",
        "Step Time", "Detector First X-Channel", "Detector Last X-Channel",
        "Detector First Y-Channel", "Detector Last Y-Channel", "Number of Slices",
        "Sequence", "Spectrum Name", "Instrument", "Location", "User", "Sample",
        "Time per Spectrum Channel", "DetectorMode"
    }
    
    for k, v in region_ini.items():
        if k not in known_keys:
            meta[k] = v
            
    # Also store the known ones that aren't top-level attrs in meta for completeness if desired,
    # or just the ones requested by the user constraints.
    # The user requested specific mapping to attrs["meta"].
    
    meta["thtM"] = manipulator_tilt
    meta["tltE"] = manipulator_polar
    meta["info"] = comments
    meta["Epass"] = pass_energy
    meta["Mode"] = lens_mode
    meta["X"] = manipulator_x
    meta["Y"] = manipulator_y
    meta["Z"] = manipulator_z
    meta["Azimuth"] = manipulator_azimuth
    
    # Add other fields from region_ini to meta if they are useful
    meta.update({k: v for k, v in region_ini.items() if k not in meta})

    # Construct Attributes
    attrs = {
        "FileName": os.path.basename(path),
        "H5file": path, # Keeping naming consistent with request, though it's a zip
        "TimeStamp": f"{date_str} {time_str}".strip(),
        "Type": "Eb(kx,ky)", # As per MATLAB script
        "hv": excitation_energy,
        "tltM": deflector_axis, # This is the axis array in MATLAB struct
        "tltE": manipulator_polar, # This seems to be a single value in MATLAB struct (meta.tltE)
        "thtM": manipulator_tilt, # Single value
        "Temp": 0.0, # Not found in parsing, default
        "meta": meta,
        "raw_shape": data_final.shape
    }
    
    # Determine Axis Labels/Units
    # MATLAB: 
    # if startsWith(EnergyScale,'Kinetic') -> 'Kinetic energy' else 'Binding energy'
    # Units: 'eV', 'degree', 'degree'
    
    energy_label = "Kinetic energy" if energy_scale.startswith("Kinetic") else "Binding energy"
    
    coords = {
        "energy": (["energy"], energy_axis, {"units": "eV", "long_name": energy_label}),
        "angle": (["angle"], analyzer_axis, {"units": "degree", "long_name": "Analyzer angle"}),
        "scan": (["scan"], deflector_axis, {"units": "degree", "long_name": "Deflector angle"}),
    }

    # Create DataArray
    # Dimensions: (energy, angle, scan)
    da = xr.DataArray(
        data=data_final,
        coords=coords,
        dims=("energy", "angle", "scan"),
        attrs=attrs,
        name="intensity"
    )
    
    return da
