import os
import re
import numpy as np
import xarray as xr
import igor2.binarywave as ibw
from datetime import datetime

def load(path: str) -> xr.DataArray:
    """
    Loads an Igor Binary Wave (.ibw) file and returns an xarray.DataArray.
    
    Replicates the logic of the MATLAB 'IBWread.m' loader:
    - Parses wave data and headers.
    - Extracts metadata from the wave note (Excitation Energy, Low Energy, etc.).
    - Constructs physical axes (Binding Energy, Theta, Tilt/Phi).
    - Preserves metadata in attrs.
    
    Args:
        path (str): Absolute path to the .ibw file.
        
    Returns:
        xr.DataArray: The loaded data with coordinates and metadata.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    try:
        # Load the IBW file using igor2
        # ibw.load returns a dictionary with 'wave' key
        ibw_data = ibw.load(path)
        wave = ibw_data['wave']
    except Exception as e:
        raise IOError(f"Failed to load IBW file with igor2: {e}")

    # Extract raw data
    # igor2 handles complex numbers and reshaping automatically in most cases
    raw_data = wave['wData']
    header = wave['wave_header']
    note = wave.get('note', b'').decode('utf-8', errors='ignore')
    
    # Determine dimensions
    # header['nDim'] is a list of dimension sizes. 
    # In MATLAB: dataStr.Ndim = sum(dataStr.waveHeader.nDim>0);
    # In Python, raw_data.ndim should usually match, but we check header for axis scaling.
    
    # Note: igor2 might return raw_data with shape matching nDim.
    # We need to ensure the shape matches our expectations for (Eb, Theta, Phi).
    
    shape = raw_data.shape
    ndim = raw_data.ndim
    
    # Extract scaling parameters
    # sfA is step (delta), sfB is start (offset)
    # They are lists/arrays in the header.
    sfA = header.get('sfA', [1.0]*ndim)
    sfB = header.get('sfB', [0.0]*ndim)
    
    # --- Metadata Extraction (Regex parsing of Note) ---
    attrs = {
        "FileName": os.path.basename(path),
        "H5file": path, # Keeping consistency with other loaders
        "TimeStamp": "", # Will try to get file modification time if not in note
        "Type": "",
        "meta": {}
    }
    
    # File timestamp (fallback to filesystem as in MATLAB script logic is slightly different but we want robust)
    # MATLAB: FileInfo = dir(...); TimeStamp = string(FileInfo.date);
    try:
        mtime = os.path.getmtime(path)
        attrs["TimeStamp"] = datetime.fromtimestamp(mtime).isoformat()
    except:
        pass

    # Parse Note
    # Look for hv
    hv = None
    match_hv = re.search(r'Excitation Energy=([\d.]+)', note)
    if match_hv:
        hv = float(match_hv.group(1))
        attrs["hv"] = hv
        
    # Look for kinetic energy (Low Energy)
    ke_start = None
    match_ke = re.search(r'Low Energy=([\d.]+)', note)
    if match_ke:
        ke_start = float(match_ke.group(1))
        
    # Look for Energy Scale (Binding vs Kinetic)
    is_binding = False
    match_scale = re.search(r'Energy Scale=(\w+)', note)
    if match_scale and match_scale.group(1) == 'Binding':
        is_binding = True
        
    # Look for tilt angle (P-Axis)
    tltM = 0.0
    match_tlt = re.search(r'P-Axis=([\d.]+)', note)
    if match_tlt:
        tltM = float(match_tlt.group(1))
    
    # Store parsed values
    attrs["tltM"] = tltM
    # Default other metadata
    attrs["tltE"] = 0
    attrs["thtM"] = 0
    attrs["Temp"] = np.nan # Not parsed in MATLAB script, setting default
    
    # Store raw note in meta
    attrs["meta"]["IBW_Note"] = note
    attrs["meta"]["IBW_Header"] = str(header)
    
    # --- Axis Construction ---
    coords = {}
    dims = []
    
    # Axis 0: Energy (usually)
    # In MATLAB script, dimension 1 is treated as Energy.
    # raw_eb = (ke_start : step : end) - hv + 4.5
    
    # Construct generic axes first based on sfA/sfB
    axes_coords = []
    for i in range(ndim):
        n_points = shape[i]
        start = sfB[i]
        step = sfA[i]
        axis_vals = start + np.arange(n_points) * step
        axes_coords.append(axis_vals)
        
    # Now map to physical meanings based on dimensions
    
    # Dimension 0: Energy
    if ndim >= 1:
        dims.append("energy")
        
        # If we successfully parsed ke_start and hv, we override the generic axis 0
        if ke_start is not None and hv is not None:
            # Reconstruct energy axis using ke_start logic from MATLAB
            # raw_eb = (ke_start:dataStr.waveHeader.sfA(1):...)
            # Note: sfA[0] should be the step
            step_e = sfA[0]
            # Calculate axis based on ke_start
            # MATLAB: raw_eb = (ke_start : step : ...)
            # We use the size of the data
            raw_eb_kinetic = ke_start + np.arange(shape[0]) * step_e
            
            # Flip if Binding
            if is_binding:
                raw_eb_kinetic = np.flip(raw_eb_kinetic)
                
            # Convert to Binding Energy: Eb = Ek - hv + 4.5
            # MATLAB: dataStr.raw_eb = raw_eb-hv+4.5;
            raw_eb = raw_eb_kinetic - hv + 4.5
            
            coords["energy"] = raw_eb
            attrs["Type"] = "Eb(k)" # Default 2D type
        else:
            # Fallback if metadata missing
            coords["energy"] = axes_coords[0]
            
    # Dimension 1: Angle
    if ndim >= 2:
        dims.append("angle")
        # MATLAB: raw_tht = sfB(2):sfA(2):...
        # This matches our generic axes_coords[1]
        coords["angle"] = axes_coords[1]
        
    # Dimension 2: Scan (Tilt/Phi)
    if ndim >= 3:
        dims.append("scan")
        # MATLAB: dataStr.tltM = sfB(3):sfA(3):...
        # MATLAB sets Type = "Eb(kx,ky)"
        coords["scan"] = axes_coords[2]
        attrs["Type"] = "Eb(kx,ky)"
        
        # In MATLAB, if 3D, tltM becomes the axis array. 
        # If not 3D, tltM is a single value from P-Axis.
        # We handle this by having 'phi' coord for 3D.
        # And we already set attrs["tltM"] to the single value (or 0) earlier.
        # If 3D, we might want to update attrs["tltM"] to be the range? 
        # The MATLAB script does: dataStr.tltM = <array> if 3D.
        # xarray handles this via the coordinate 'phi'. 
        # We can leave attrs['tltM'] as the single value parsed from notes (often the center or setpoint) 
        # or just remove it if it conflicts. 
        # For now, we leave the scalar in attrs if it was parsed.
        
    # Handle Transpose if necessary
    # MATLAB: dataStr.raw_data = fread(...); reshape(...);
    # MATLAB is column-major (Fortran order). Python/Numpy is row-major (C order).
    # igor2 usually returns data indexed as [x, y, z] corresponding to Igor's rows, cols, layers.
    # Igor: x (rows), y (cols), z (layers).
    # MATLAB: size(raw_data) = [Nx, Ny, Nz].
    # Numpy from igor2: usually [Nx, Ny, Nz] if not transposed.
    # However, xarray/plotting often expects [Energy, Angle] or [Angle, Energy].
    # In ARPES, usually Dimension 0 is Energy (rows in Igor), Dimension 1 is Angle (cols).
    # So [Nx, Ny] = [Energy, Angle].
    # This matches the dims assignment above: dims=("Eb", "theta").
    
    # Create DataArray
    da = xr.DataArray(
        data=raw_data,
        dims=tuple(dims),
        coords=coords,
        attrs=attrs
    )
    
    return da

if __name__ == "__main__":
    # Simple test block
    import sys
    if len(sys.argv) > 1:
        fpath = sys.argv[1]
        try:
            da = load(fpath)
            print(da)
        except Exception as e:
            print(f"Error: {e}")
