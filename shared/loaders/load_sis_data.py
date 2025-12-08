import h5py
import xarray as xr
import numpy as np
import os
import warnings

def load_sis_data(path: str) -> xr.DataArray:
    """
    Loads SIS beamline ARPES data from an HDF5 file into an xarray.DataArray.

    Parameters
    ----------
    path : str
        Path to the .h5 file.

    Returns
    -------
    xr.DataArray
        The loaded data with coordinates and metadata.
        Dimensions are typically ('energy', 'angle') for 2D or ('energy', 'angle', 'scan') for 3D.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    with h5py.File(path, 'r') as f:
        # 1. Read Data
        # Path based on MATLAB loader: /Electron Analyzer/Image Data
        data_path = '/Electron Analyzer/Image Data'
        if data_path not in f:
            raise KeyError(f"Could not find data at {data_path}")
        
        dset = f[data_path]
        data = dset[:]
        
        # 2. Read Axes
        # Helper to read attributes safely
        def get_attr(obj, key, default=None):
            if key in obj.attrs:
                val = obj.attrs[key]
                if isinstance(val, bytes):
                    return val.decode('utf-8')
                return val
            return default

        # Axis 0: Energy (usually)
        ax0_scale = get_attr(dset, 'Axis0.Scale')
        ax0_n = data.shape[0]
        if ax0_scale is not None and len(ax0_scale) == 2:
            # start, step
            coords_0 = np.arange(ax0_n) * ax0_scale[1] + ax0_scale[0]
        else:
            coords_0 = np.arange(ax0_n)
            
        # Axis 1: Angle (usually)
        ax1_scale = get_attr(dset, 'Axis1.Scale')
        if len(data.shape) > 1:
            ax1_n = data.shape[1]
            if ax1_scale is not None and len(ax1_scale) == 2:
                coords_1 = np.arange(ax1_n) * ax1_scale[1] + ax1_scale[0]
            else:
                coords_1 = np.arange(ax1_n)
        else:
            coords_1 = None

        # Axis 2: Scan (if 3D)
        ax2_scale = get_attr(dset, 'Axis2.Scale')
        if len(data.shape) > 2:
            ax2_n = data.shape[2]
            if ax2_scale is not None and len(ax2_scale) == 2:
                coords_2 = np.arange(ax2_n) * ax2_scale[1] + ax2_scale[0]
            else:
                coords_2 = np.arange(ax2_n)
        else:
            coords_2 = None

        # 3. Construct Coordinates and Dimensions
        coords = {}
        dims = []
        
        # Mapping based on MATLAB analysis:
        # MATLAB Axis0 -> Energy
        # MATLAB Axis1 -> Angle
        # MATLAB Axis2 -> Scan
        # Python h5py shape matches this order (energy, angle, scan) if saved as C-order.
        # We use standardized names: energy, angle, scan
        
        coords['energy'] = coords_0
        dims.append('energy')
        
        if coords_1 is not None:
            coords['angle'] = coords_1
            dims.append('angle')
            
        if coords_2 is not None:
            coords['scan'] = coords_2
            dims.append('scan')

        # 4. Read Metadata
        attrs = {}
        attrs['FileName'] = os.path.basename(path).replace('.h5', '')
        attrs['H5file'] = os.path.basename(path)
        attrs['H5path'] = os.path.dirname(path)
        
        # Collect all attributes from Image Data
        meta = {}
        for k, v in dset.attrs.items():
            if isinstance(v, bytes):
                meta[k] = v.decode('utf-8')
            else:
                meta[k] = v
        
        # Collect Instrument Data
        inst_path = '/Other Instruments'
        if inst_path in f:
            inst_group = f[inst_path]
            for k in inst_group.keys():
                try:
                    # These are datasets usually containing single values or arrays
                    val = inst_group[k][:]
                    if val.size == 1:
                        val = val.item()
                    # Clean up key name to match MATLAB style if desired, or keep original
                    clean_key = k.replace(' ', '_').replace('.', '')
                    meta[clean_key] = val
                    
                    # Map specific fields to top-level attrs as per requirement
                    if 'hv' in k:
                        attrs['hv'] = val
                    if 'Tilt' in k:
                        attrs['tltM'] = val
                    if 'Theta' in k:
                        attrs['thtM'] = val
                    if 'Temperature A' in k: # Cryostat
                        attrs['Temp'] = val
                        
                except Exception:
                    pass

        attrs['meta'] = meta
        
        # 5. Determine Type (Eb(k), Eb(kx,ky), etc.)
        # Logic from MATLAB:
        # if length(Tilt) > 1 -> Eb(kx,ky)
        # elseif length(hv) > 1 -> Eb(kx,kz)
        # else -> Eb(k)
        
        tlt = attrs.get('tltM', 0)
        hv = attrs.get('hv', 0)
        
        # Helper to check if array has > 1 element
        def is_scan(val):
            return isinstance(val, (np.ndarray, list)) and np.size(val) > 1

        if is_scan(tlt):
            attrs['Type'] = "Eb(kx,ky)"
        elif is_scan(hv):
            attrs['Type'] = "Eb(kx,kz)"
        elif len(dims) == 3:
             attrs['Type'] = "Eb(k,i)" # Generic 3D
        else:
            attrs['Type'] = "Eb(k)"

        # 6. Create DataArray
        da = xr.DataArray(
            data,
            coords=coords,
            dims=dims,
            attrs=attrs
        )
        
        return da

if __name__ == "__main__":
    # Simple test block (not executed during import)
    pass
