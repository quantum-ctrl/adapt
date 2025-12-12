import os
import sys
import glob
import numpy as np
import shutil
import atexit
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import xarray as xr
from io import BytesIO
# Add shared module to path for session manager
_shared_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)

try:
    from brillouin_zone import generate_bz, load_from_parameters, load_from_material_id, plot_bz_matplotlib, get_bz_intersection_plane
except ImportError as e:
    import traceback
    traceback.print_exc()
    print(f"WARNING: Could not import brillouin_zone module: {e}")

try:
    from session import read_session, clear_session
except ImportError:
    # Fallback if shared module not available
    print("WARNING: Session manager not available. Session integration disabled.")
    def read_session():
        return None
    def clear_session():
        return False

# Import the data loaders from the shared loaders package
# Path to shared/ was added to sys.path above
try:
    from loaders import load_adress as load_hdf5_data
    from loaders import load_ses as load_ses_zip
    from loaders import load_ibw
    from loaders import load_pxt
except ImportError as e:
    print(f"WARNING: Could not import loaders: {e}")
    # Fallback stubs if loaders are not available
    def load_hdf5_data(path):
        raise ImportError("Shared loaders not available")
    def load_ses_zip(path):
        raise ImportError("Shared loaders not available")
    def load_ibw(path):
        raise ImportError("Shared loaders not available")
    def load_pxt(path):
        raise ImportError("Shared loaders not available")

# Import Processing Modules with granular error handling
align_error = None
fermi_fit_error = None

# 1. Align Module
try:
    from processing.align import align_axis, align_energy, align_energy_3d
except ImportError as e:
    align_error = str(e)
    print(f"WARNING: Could not import alignment module: {e}")
    def align_axis(*args, **kwargs): raise ImportError(f"Alignment not available: {align_error}")
    def align_energy(*args, **kwargs): raise ImportError(f"Alignment not available: {align_error}")
    def align_energy_3d(*args, **kwargs): raise ImportError(f"Alignment not available: {align_error}")

# 2. Fermi Edge Fit Module
try:
    from processing.fermi_edge_fit import fit_fermi_edge, fit_fermi_edge_3d
except ImportError as e:
    fermi_fit_error = str(e)
    print(f"WARNING: Could not import fermi fit module: {e}")
    def fit_fermi_edge(*args, **kwargs): raise ImportError(f"Fermi fit not available: {fermi_fit_error}")
    def fit_fermi_edge_3d(*args, **kwargs): raise ImportError(f"Fermi fit not available: {fermi_fit_error}")

# 3. Angle to K Module
try:
    from processing.angle_to_k import convert_angle_to_k
except ImportError as e:
    print(f"WARNING: Could not import angle_to_k module: {e}")
    def convert_angle_to_k(*args, **kwargs): raise ImportError(f"Angle to K conversion not available: {e}")

# 4. Edit Module (Crop)
try:
    from processing.edit import crop_data
except ImportError as e:
    print(f"WARNING: Could not import edit module: {e}")
    def crop_data(*args, **kwargs): raise ImportError(f"Crop not available: {e}")

app = FastAPI(title="ARPES Visualization Tool")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MAX_UPLOAD_SIZE = 2_000_000_000  # 2GB limit for uploaded files

# Track temporary uploaded files for cleanup
TEMP_FILES = []

def cleanup_temp_files():
    """Clean up temporary files on shutdown."""
    for tmp in TEMP_FILES:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception as e:
                print(f"Failed to clean up {tmp}: {e}")

atexit.register(cleanup_temp_files)


def _load_data_file(file_path: str):
    """
    Load a data file using the appropriate loader based on file extension.
    
    Args:
        file_path: Absolute path to the data file
        
    Returns:
        xarray.DataArray or dict with data
        
    Raises:
        HTTPException: If file type is not supported
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in ('.h5', '.nxs', '.hdf5'):
        return load_hdf5_data(file_path)
    elif ext == '.ibw':
        return load_ibw(file_path)
    elif ext == '.zip':
        return load_ses_zip(file_path)
    elif ext in ('.pxt', '.pxp'):
        return load_pxt(file_path)
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")


@app.get("/api/session")
async def get_session():
    """
    Read session data from ~/.adapt/session.json.
    
    This endpoint is called by the frontend when ?session=1 is present in the URL.
    It returns the file path and metadata stored by ADAPT Browser.
    
    Returns:
        JSON with session data including file_path and metadata.
        Returns 404 if no session file exists.
    """
    session_data = read_session()
    
    if session_data is None:
        raise HTTPException(
            status_code=404,
            detail="No active session found. Please select a file in ADAPT Browser first."
        )
    
    # Validate file exists
    file_path = session_data.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=f"Session file not found: {file_path}"
        )
    
    return {
        "session": session_data,
        "file_path": file_path,
        "metadata": session_data.get("metadata", {})
    }


@app.delete("/api/session")
async def delete_session():
    """
    Clear the current session.
    """
    success = clear_session()
    return {"cleared": success}

@app.get("/api/files")
async def list_files():
    """List all HDF5 files in the data directory."""
    if not os.path.exists(DATA_DIR):
        return {"files": []}
    
    # Search for .h5, .nxs, .hdf5 and .zip (SES archives)
    # Search for .h5, .nxs, .hdf5, .zip (SES archives), .ibw (Igor Binary Wave), and .pxt/.pxp
    extensions = ['*.h5', '*.nxs', '*.hdf5', '*.zip', '*.ibw', '*.pxt', '*.pxp']
    files = set()
    for ext in extensions:
        # Use recursive search for all
        found = glob.glob(os.path.join(DATA_DIR, "**", ext), recursive=True)
        files.update(found)
    
    # Return relative paths
    rel_files = [os.path.relpath(f, DATA_DIR) for f in files]
    return {"files": sorted(rel_files)}

@app.get("/api/browse")
async def browse_files(path: str = Query(None, description="Absolute path to browse")):
    """Browse files in the filesystem."""
    # Default to DATA_DIR if no path provided
    if not path:
        path = DATA_DIR
    
    # Resolve path
    path = os.path.abspath(path)
    
    # Note: This allows browsing the entire filesystem for local use.
    # For public deployment, consider adding authentication or path restrictions
    # via environment variables (e.g., RESTRICT_BROWSING=true).
    
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Path not found")
    
    if not os.path.isdir(path):
        raise HTTPException(status_code=400, detail="Path is not a directory")
    
    try:
        items = os.listdir(path)
        dirs = []
        files = []
        
        for item in items:
            if item.startswith('.'): continue # Skip hidden files
            
            full_path = os.path.join(path, item)
            if os.path.isdir(full_path):
                dirs.append(item)
            elif item.lower().endswith(('.h5', '.nxs', '.hdf5', '.ibw', '.zip', '.pxt', '.pxp')):
                files.append(item)
        
        return {
            "parent": os.path.dirname(path),
            "current": path,
            "dirs": sorted(dirs),
            "files": sorted(files)
        }
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/load")
async def load_metadata(path: str = Query(..., description="Path to the file")):
    """Load file metadata and axis info."""
    # Check if path is absolute (e.g. temp file) or relative
    if os.path.isabs(path):
        file_path = path
    else:
        file_path = os.path.join(DATA_DIR, path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Load data using the centralized loader
        result = _load_data_file(file_path)

        if isinstance(result, xr.DataArray):
            da = result
            # Build axes dict from coords
            axes = {}
            for coord in da.coords:
                try:
                    axes[coord] = np.array(da.coords[coord]).tolist()
                except Exception:
                    axes[coord] = list(da.coords[coord].values)

            # Provide normalized axis names for frontend compatibility
            # Common mapping: Eb -> energy, theta/angle -> kx, scan/tilt -> ky
            normalized_axes = {}
            if 'Eb' in axes and 'energy' not in axes:
                normalized_axes['energy'] = axes['Eb']
            # Map theta or angle to kx
            if 'theta' in axes and 'kx' not in axes:
                normalized_axes['kx'] = axes['theta']
            if 'angle' in axes and 'kx' not in axes and 'kx' not in normalized_axes:
                normalized_axes['kx'] = axes['angle']
            # Map scan/phi/tilt to ky
            if 'scan' in axes and 'ky' not in axes:
                normalized_axes['ky'] = axes['scan']
            if 'phi' in axes and 'ky' not in normalized_axes and 'ky' not in axes:
                normalized_axes['ky'] = axes['phi']
            if 'tilt' in axes and 'ky' not in normalized_axes and 'ky' not in axes:
                normalized_axes['ky'] = axes['tilt']

            # Map 'k' to 'kx' (common output from conversion)
            if 'k' in axes and 'kx' not in axes and 'kx' not in normalized_axes:
                normalized_axes['kx'] = axes['k']

            # Map 'kz' to 'ky' (for kx-kz converted data)
            if 'kz' in axes and 'ky' not in axes and 'ky' not in normalized_axes:
                normalized_axes['ky'] = axes['kz']

            # Merge normalized keys into axes (without overwriting existing keys)
            for k, v in normalized_axes.items():
                if k not in axes:
                    axes[k] = v

            # metadata from attrs
            metadata = dict(da.attrs or {})
            
            # Helper to sanitize metadata for JSON serialization
            def sanitize_for_json(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.float32, np.float64)):
                    val = float(obj)
                    # Convert NaN and infinity to None for JSON compliance
                    if np.isnan(val) or np.isinf(val):
                        return None
                    return val
                elif isinstance(obj, (np.int32, np.int64, np.int16, np.int8, np.uint32, np.uint64)):
                    return int(obj)
                elif isinstance(obj, float):
                    # Handle Python float NaN and infinity
                    if np.isnan(obj) or np.isinf(obj):
                        return None
                    return obj
                elif isinstance(obj, dict):
                    return {k: sanitize_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [sanitize_for_json(i) for i in obj]
                elif isinstance(obj, tuple):
                    return tuple(sanitize_for_json(i) for i in obj)
                return obj

            metadata = sanitize_for_json(metadata)

            data = da.values
            data_info = {
                "shape": data.shape,
                "dtype": str(data.dtype),
                "min": float(np.nanmin(data)),
                "max": float(np.nanmax(data)),
                "ndim": data.ndim
            }

            # Determine units based on conversion
            kx_unit = None
            ky_unit = None
            if metadata.get('conversion') in ['angle_to_k', 'angle_hv_to_kxkz', 'angle_to_k_kz', 'kx_ky']:
                 kx_unit = "1/Å"
                 if data.ndim == 3:
                     ky_unit = "1/Å"
            
            # Use '1/Å' if kx is explicitly in axes (fallback)
            if 'kx' in axes and ('angle' not in axes and 'theta' not in axes):
                # If we only have kx and no angle, assume it's k-space (weak heuristic)
                pass

            resp = {
                "filename": path,
                "metadata": metadata,
                "axes": axes,
                "data_info": data_info
            }
            if kx_unit: resp['kx_unit'] = kx_unit
            if ky_unit: resp['ky_unit'] = ky_unit
            
            return resp
        else:
            # Backwards compatibility: handle dict result (legacy)
            metadata = result.get("metadata", {})
            axes = {}
            for k, v in result.get("axes", {}).items():
                axes[k] = v.tolist() if isinstance(v, np.ndarray) else v

            # Normalize legacy axis names if possible
            if 'energy' not in axes:
                if 'Eb' in axes:
                    axes['energy'] = axes['Eb']
            if 'kx' not in axes:
                if 'theta' in axes:
                    axes['kx'] = axes['theta']
                elif 'angle' in axes:
                    axes['kx'] = axes['angle']
            if 'ky' not in axes:
                if 'scan' in axes:
                    axes['ky'] = axes['scan']
                elif 'tilt' in axes:
                    axes['ky'] = axes['tilt']
            data = result.get("data")
            data_info = {
                "shape": data.shape,
                "dtype": str(data.dtype),
                "min": float(np.min(data)),
                "max": float(np.max(data)),
                "ndim": data.ndim
            }
            return {
                "filename": path,
                "metadata": metadata,
                "axes": axes,
                "data_info": data_info
            }
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": error_details 
            }
        )

@app.get("/api/data")
async def get_data(path: str = Query(..., description="Path to the file")):
    """Get the raw data array as a binary stream."""
    # Check if path is absolute (e.g. temp file) or relative
    if os.path.isabs(path):
        file_path = path
    else:
        file_path = os.path.join(DATA_DIR, path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        # Load data using the centralized loader
        result = _load_data_file(file_path)

        if isinstance(result, xr.DataArray):
            data = result.values
        else:
            data = result.get("data")

        # Ensure float32 for consistency and size
        if data.dtype != np.float32:
            data = data.astype(np.float32)

        return Response(content=data.tobytes(), media_type="application/octet-stream")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "traceback": error_details
            }
        )

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file to the server using a temporary file."""
    try:
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        size = file.file.tell()
        if size > MAX_UPLOAD_SIZE:
            raise HTTPException(
                status_code=413, 
                detail=f"File too large (max {MAX_UPLOAD_SIZE/1e9:.1f}GB)"
            )
        file.file.seek(0)  # Reset to beginning
        
        # Create a temporary file
        # delete=False ensures it persists after closing, so we can read it later
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
            TEMP_FILES.append(tmp_path)  # Track for cleanup
            
        # Return absolute path
        return {"filename": tmp_path}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Processing Endpoints
# =============================================================================

class AlignRequest(BaseModel):
    path: str
    axis: str
    offset: float
    scan_axis: Optional[str] = "scan"
    scan_offset: Optional[float] = None
    method: str = "manual"
    hv_mapping_enabled: bool = False
    fit_range: Optional[float] = None

class FitFermiRequest(BaseModel):
    path: str
    energy_window: List[float]
    theta_range: Optional[List[float]] = None

class ConvertKRequest(BaseModel):
    path: str
    hv: Optional[float] = None
    work_function: float = 4.5
    inner_potential: float = 10.0
    hv_mapping_enabled: bool = False
    is_hv_scan: bool = False # Optional, for explicit scan type override
    hv_dim: Optional[str] = None
    convert_hv_to_kz: bool = False

class CropRequest(BaseModel):
    path: str
    ranges: dict # {'x': [start, end], 'y': [start, end], 'z': [start, end]}

@app.post("/api/process/align")
async def align_data(request: AlignRequest):
    """Align data along an axis and return path to new file."""
    file_path = request.path
    if not os.path.isabs(file_path):
        file_path = os.path.join(DATA_DIR, file_path)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        # Load data using centralized loader
        data = _load_data_file(file_path)

        if not isinstance(data, xr.DataArray):
             raise HTTPException(status_code=400, detail="Data must be xarray for alignment")
             
        # Perform alignment
        if request.axis == 'energy' or request.axis == 'Eb':
            # Check for 3D HV Mapping Case
            # Conditions: 3D data AND hv_mapping_enabled is True
            if data.ndim == 3 and request.hv_mapping_enabled:
                 if request.fit_range is None:
                     raise HTTPException(status_code=400, detail="fit_range is required for 3D HV mapping alignment")
                 
                 # 1. Fit Fermi Edge for each slice
                 # energy_window = (ef textbox - range, ef textbox + range)
                 # Here request.offset corresponds to the 'ef textbox' value passed from frontend
                 e_min = request.offset - request.fit_range
                 e_max = request.offset + request.fit_range
                 energy_window = (e_min, e_max)
                 
                 # We need to identify the scan dimension. 
                 # Usually it is the last dimension or named 'scan', 'ky', etc.
                 # Let's try to detect it or use a default.
                 # data.dims usually looks like ('energy', 'angle', 'scan') or similar.
                 scan_dim = 'scan'
                 if 'ky' in data.dims: scan_dim = 'ky'
                 
                 # Call fit_fermi_edge_3d
                 # Note: fit_fermi_edge_3d is imported from processing.fermi_edge_fit
                 fit_results = fit_fermi_edge_3d(
                     data, 
                     energy_window=energy_window, 
                     scan_dim=scan_dim,
                     show_progress=False
                 )
                 
                 E_F_array = fit_results['E_F']
                 
                 # 2. Align Energy 3D
                 aligned = align_energy_3d(
                     data, 
                     E_F_array=E_F_array, 
                     scan_dim=scan_dim,
                     energy_axis=request.axis
                 )
                 
                 # Update metadata to reflect 3D alignment
                 aligned.attrs['alignment_method'] = f"3D_HV_Mapping_Fit (Center={request.offset}, Range={request.fit_range})"

            else:
                # Standard 2D alignment or 3D alignment with constant shift
                # Use align_energy for energy axis
                aligned = align_energy(data, E_F=request.offset, energy_axis=request.axis)
        else:
            # Handle Axis (Angle/Theta) Alignment
            
            # Resolve axis alias
            target_axis = request.axis
            # Heuristic: if requested 'angle' or 'theta' but data has 'kx', use 'kx'
            if target_axis not in data.coords:
                if target_axis in ['angle', 'theta'] and 'kx' in data.coords:
                    target_axis = 'kx'
                elif target_axis in ['angle', 'kx'] and 'theta' in data.coords:
                    target_axis = 'theta'
                elif target_axis in ['kx', 'theta'] and 'angle' in data.coords:
                    target_axis = 'angle'
            
            # 1. Align Primary Axis (e.g. Theta/kx)
            aligned = align_axis(data, axis_name=target_axis, offset=request.offset, method=request.method)
            
            # 2. Check for 3D logic (Composite Alignment)
            if aligned.ndim == 3:
                # Check 3D status
                # If HV Mapping is Convert (Enabled): Align Theta ONLY. (Done above)
                # If HV Mapping is Fermi Surf (Disabled): Align Theta AND Scan (if scan provided).
                
                if not request.hv_mapping_enabled:
                    # Case: Fermi Surface Mapping -> Align BOTH Theta and Scan
                    if request.scan_offset is not None and request.scan_offset != 0:
                         # Resolve scan axis
                         scan_axis = request.scan_axis
                         if scan_axis not in aligned.coords:
                             # Default heuristics
                             if 'ky' in aligned.coords: scan_axis = 'ky'
                             elif 'scan' in aligned.coords: scan_axis = 'scan'
                             elif 'tilt' in aligned.coords: scan_axis = 'tilt'
                         
                         if scan_axis in aligned.coords:
                             aligned = align_axis(
                                 aligned, 
                                 axis_name=scan_axis, 
                                 offset=request.scan_offset, 
                                 method=f"{request.method}_plus_scan"
                             )
                
                # If hv_mapping_enabled is True, we ONLY aligned theta/kx above, which is correct.
            
        # Add processed flag
        aligned.attrs['is_adapt_processed'] = True

        # Save to new temp file
        import h5py
        # Always save as .h5 because we write HDF5/NetCDF format
        suffix = ".h5"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="aligned_") as tmp:
            save_path = tmp.name
            TEMP_FILES.append(save_path)
        
        # Save using xarray to netcdf/h5 if possible, or custom saver
        # Since we use h5py loaders usually, let's try to save as h5 using a simple saver or xarray's to_netcdf
        # For compatibility with our loader, xarray to_netcdf (h5) is best if h5netcdf is installed.
        # Otherwise, we might need a simple save function. 
        # For now, let's assume we can save to netcdf/h5 which is standard for xarray.
        try:
            # Drop complex attributes that fail serialization
            # simplified_attrs = {k: str(v) for k, v in aligned.attrs.items()}
            # aligned.attrs = simplified_attrs
            aligned.to_netcdf(save_path, engine='h5netcdf')
        except Exception as e:
            # Fallback: if h5netcdf fails, try scipy or just pickle? No, pickle is bad for interop.
            # Let's try native h5py if available, or just fail for now.
            # Actually, standard ARPES data in this project seems to be HDF5. 
            with h5py.File(save_path, 'w') as f:
                ds = f.create_dataset('data', data=aligned.values)
                # Save coords
                for coord in aligned.coords:
                    f.create_dataset(coord, data=aligned.coords[coord].values)
                # Save attrs
                for k, v in aligned.attrs.items():
                    try:
                        f.attrs[k] = v
                    except (TypeError, ValueError, OSError):
                        f.attrs[k] = str(v)
                        
        return {"filename": save_path}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process/fit_fermi_edge")
async def fit_fermi_edge_endpoint(request: FitFermiRequest):
    """Fit Fermi edge and return parameters."""
    file_path = request.path
    if not os.path.isabs(file_path):
        file_path = os.path.join(DATA_DIR, file_path)
        
    try:
        # Load data using centralized loader
        data = _load_data_file(file_path)
             
        if not isinstance(data, xr.DataArray):
             raise HTTPException(status_code=400, detail="Data must be xarray")
             
        # Determine axes names
        # Simple heuristic or check coords
        energy_dim = 'energy'
        if 'Eb' in data.dims: energy_dim = 'Eb'
        
        angle_dim = 'angle'
        if 'theta' in data.dims: angle_dim = 'theta'
        elif 'kx' in data.dims: angle_dim = 'kx'
        
        results = fit_fermi_edge(
            data, 
            energy_window=tuple(request.energy_window),
            theta_range=tuple(request.theta_range) if request.theta_range else None,
            energy_dim=energy_dim,
            angle_dim=angle_dim
        )
        
        # Convert numpy types to native for JSON
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, (np.float32, np.float64)):
                clean_results[k] = float(v)
            elif k in ['success', 'error']:
                clean_results[k] = v
                
        return clean_results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process/convert_k")
async def convert_k(request: ConvertKRequest):
    """Convert data to k-space."""
    file_path = request.path
    if not os.path.isabs(file_path):
        file_path = os.path.join(DATA_DIR, file_path)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Load data using centralized loader
        data = _load_data_file(file_path)

        if not isinstance(data, xr.DataArray):
             raise HTTPException(status_code=400, detail="Data must be xarray for conversion")

        # Call conversion logic
        # Logic:
        # 2D -> angle_to_k(data, hv=hv)
        # 3D (HV Mapping Disabled) -> angle_to_k(data, hv=hv) to treat as kx-ky mapping
        # 3D (HV Mapping Enabled) -> This implies variable hv. 
        #    If user passed explicit 'hv', it overrides? 
        #    Standard logic for hv mapping usually reads hv from coords.
        #    However, the request logic here specifically asks to handle 2D and 3D (disabled) with explicit HV.
        
        # We pass request.hv. If it's physically relevant (e.g. constant hv slice), it will be used.
        converted = convert_angle_to_k(
            data, 
            hv=request.hv, 
            phi=request.work_function, 
            V0=request.inner_potential,
            is_hv_scan=request.hv_mapping_enabled or request.is_hv_scan, 
            hv_dim=request.hv_dim if request.hv_dim else 'scan',
            convert_hv_to_kz=request.convert_hv_to_kz
        )
        
        # Add processed flag
        converted.attrs['is_adapt_processed'] = True

        # Save to new temp file
        import h5py
        suffix = ".h5"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="k_converted_") as tmp:
            save_path = tmp.name
            TEMP_FILES.append(save_path)
        
        try:
            converted.to_netcdf(save_path, engine='h5netcdf')
        except Exception as e:
            with h5py.File(save_path, 'w') as f:
                ds = f.create_dataset('data', data=converted.values)
                for coord in converted.coords:
                    f.create_dataset(coord, data=converted.coords[coord].values)
                for k, v in converted.attrs.items():
                    try:
                        f.attrs[k] = v
                    except (TypeError, ValueError, OSError):
                        f.attrs[k] = str(v)
                        
        return {"filename": save_path}


    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class BZRequest(BaseModel):
    method: str = "manual" # 'manual' or 'mp'
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    mp_id: Optional[str] = None
    crystal_name: Optional[str] = None
    theme: str = 'dark' # 'dark' or 'light'
    # Optional intersection plane parameters
    plane_miller: Optional[List[float]] = None  # Miller indices [h, k, l]
    plane_distance: Optional[float] = 0.0  # Distance from origin
    plane_color: Optional[str] = 'green'  # Color for intersection plane


@app.post("/api/process/brillouin_zone")
async def calculate_bz(request: BZRequest):
    """Calculate and plot Brillouin Zone, returning an image."""
    try:
        if request.method == 'mp' and request.mp_id:
            lattice = load_from_material_id(request.mp_id)
        else:
            # Defaults
            a = request.a if request.a is not None else 3.5
            b = request.b if request.b is not None else 3.5
            c = request.c if request.c is not None else 3.5
            alpha = request.alpha if request.alpha is not None else 90
            beta = request.beta if request.beta is not None else 90
            gamma = request.gamma if request.gamma is not None else 90
            lattice = load_from_parameters(float(a), float(b), float(c), float(alpha), float(beta), float(gamma))
            
        bz = generate_bz(lattice)
        
        # Determine colors based on theme
        is_dark = request.theme == 'dark'
        
        if is_dark:
            bg_color = '#121212'
            text_color = 'white'
            grid_color = '#444'
            wireframe_color = 'white' # Reverting to white/cyan-ish for visibility against black
        else:
            bg_color = '#ffffff'
            text_color = 'black'
            grid_color = '#e0e0e0'
            wireframe_color = 'black' # Black against white
        
        # Plot using Plotly
        try:
             # Check if plotly is available (it should be if we imported it)
             # We need to import it here or at top. 
             # bz_visualization handles imports, but we need to call plot_bz_plotly
             # Let's import it from the shared module if available or just use the backend dispatcher
             from brillouin_zone import plot_bz_plotly
             import plotly.graph_objects as go
             
             title = request.crystal_name if request.crystal_name else None
             
             # Use cyan for faces in both, but adjust opacity? 
             # Keeping cyan face with customized wireframe
             fig = plot_bz_plotly(bz, title=title, facecolor='cyan', opacity=0.3, wireframe_color=wireframe_color)

             # Customize for theme
             layout_args = dict(
                 paper_bgcolor=bg_color,
                 plot_bgcolor=bg_color,
                 font=dict(color=text_color),
                 scene=dict(
                     xaxis=dict(color=text_color, gridcolor=grid_color, backgroundcolor=bg_color),
                     yaxis=dict(color=text_color, gridcolor=grid_color, backgroundcolor=bg_color),
                     zaxis=dict(color=text_color, gridcolor=grid_color, backgroundcolor=bg_color),
                     bgcolor=bg_color
                 ),
                 # Increase top margin slightly to accommodate title
                 margin=dict(l=0, r=0, t=30, b=0),
                 autosize=True
             )
             
             # Explicitly position title to prevent clipping
             if title:
                 layout_args['title'] = dict(
                     text=title,
                     x=0.05,
                     y=0.98, # Slightly lower than top edge
                     xanchor='left',
                     yanchor='top',
                     font=dict(size=14, color=text_color)
                 )
                 
             fig.update_layout(**layout_args)
             
             # Enhance layout for visibility
             fig.update_layout(
                 scene=dict(
                     aspectmode='data'
                 )
             )
             
             # Add intersection plane if Miller indices provided
             if request.plane_miller and len(request.plane_miller) == 3:
                 try:
                     plane_distance = request.plane_distance if request.plane_distance is not None else 0.0
                     intersection_points = get_bz_intersection_plane(bz, request.plane_miller, plane_distance)
                     
                     if intersection_points is not None and len(intersection_points) >= 3:
                         # Create a filled polygon using Mesh3d
                         # For a convex polygon, we can use a simple fan triangulation
                         n = len(intersection_points)
                         # Fan triangulation: all triangles share vertex 0
                         i_indices = [0] * (n - 2)
                         j_indices = list(range(1, n - 1))
                         k_indices = list(range(2, n))
                         
                         plane_color = request.plane_color if request.plane_color else 'green'
                         
                         # Create filled polygon using Mesh3d with proper settings
                         plane_trace = go.Mesh3d(
                             x=intersection_points[:, 0].tolist(),
                             y=intersection_points[:, 1].tolist(),
                             z=intersection_points[:, 2].tolist(),
                             i=i_indices,
                             j=j_indices,
                             k=k_indices,
                             color=plane_color,
                             opacity=0.6,
                             flatshading=True,
                             lighting=dict(ambient=0.8, diffuse=0.5),
                             name=f"Plane ({int(request.plane_miller[0])},{int(request.plane_miller[1])},{int(request.plane_miller[2])})"
                         )
                         fig.add_trace(plane_trace)
                         
                         # Also add edge outline for clarity
                         # Close the polygon by repeating the first point
                         closed_points = list(intersection_points) + [intersection_points[0]]
                         # Use a darker shade for edge - for common colors, use 'dark' prefix
                         edge_color = f'dark{plane_color}' if plane_color in ['red', 'green', 'blue', 'cyan', 'magenta', 'orange'] else plane_color
                         edge_trace = go.Scatter3d(
                             x=[p[0] for p in closed_points],
                             y=[p[1] for p in closed_points],
                             z=[p[2] for p in closed_points],
                             mode='lines',
                             line=dict(color=edge_color, width=4),
                             name='Plane Edge',
                             showlegend=False
                         )
                         fig.add_trace(edge_trace)
                 except Exception as plane_error:
                     # Log but don't fail the entire request
                     print(f"Warning: Could not render intersection plane: {plane_error}")
             
             return Response(content=fig.to_json(), media_type="application/json")
        except Exception as e:
            # Fallback to matplotlib if plotly fails for some reason
            import matplotlib.pyplot as plt
            # Ensure non-interactive backend for server
            plt.switch_backend('Agg')
            
            title = request.crystal_name if request.crystal_name else None
            fig, ax = plot_bz_matplotlib(bz, title=title, figsize=(6, 6))
            
            buf = BytesIO()
            fig.patch.set_facecolor(bg_color) # Dark background
            ax.set_facecolor(bg_color) # Dark axis
            ax.xaxis.label.set_color(text_color)
            ax.yaxis.label.set_color(text_color)
            ax.zaxis.label.set_color(text_color)
            ax.tick_params(colors=text_color)
            if ax.title: ax.title.set_color(text_color)
            
            # Matplotlib doesn't have easy wireframe color override in this function yet, 
            # but we prioritized Plotly.
            
            fig.savefig(buf, format='png', bbox_inches='tight', dpi=100, facecolor=bg_color)
            buf.seek(0)
            plt.close(fig)
            return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

class UpdatePlaneRequest(BaseModel):
    method: str = "manual"
    a: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    alpha: Optional[float] = None
    beta: Optional[float] = None
    gamma: Optional[float] = None
    mp_id: Optional[str] = None
    plane_miller: List[float]
    plane_distance: float = 0.0
    plane_color: Optional[str] = 'green'

@app.post("/api/process/update_intersection_plane")
async def update_intersection_plane(request: UpdatePlaneRequest):
    """Update intersection plane only, returning new plane traces for Plotly."""
    try:
        import plotly.graph_objects as go
        
        # Regenerate BZ from cached parameters
        if request.method == 'mp' and request.mp_id:
            lattice = load_from_material_id(request.mp_id)
        else:
            a = request.a if request.a is not None else 3.5
            b = request.b if request.b is not None else 3.5
            c = request.c if request.c is not None else 3.5
            alpha = request.alpha if request.alpha is not None else 90
            beta = request.beta if request.beta is not None else 90
            gamma = request.gamma if request.gamma is not None else 90
            lattice = load_from_parameters(float(a), float(b), float(c), float(alpha), float(beta), float(gamma))
        
        bz = generate_bz(lattice)
        
        # Calculate intersection plane
        intersection_points = get_bz_intersection_plane(bz, request.plane_miller, request.plane_distance)
        
        if intersection_points is None or len(intersection_points) < 3:
            return {"traces": [], "message": "No valid intersection at this distance"}
        
        # Create plane trace
        n = len(intersection_points)
        i_indices = [0] * (n - 2)
        j_indices = list(range(1, n - 1))
        k_indices = list(range(2, n))
        
        plane_color = request.plane_color if request.plane_color else 'green'
        
        plane_trace = {
            "type": "mesh3d",
            "x": intersection_points[:, 0].tolist(),
            "y": intersection_points[:, 1].tolist(),
            "z": intersection_points[:, 2].tolist(),
            "i": i_indices,
            "j": j_indices,
            "k": k_indices,
            "color": plane_color,
            "opacity": 0.6,
            "flatshading": True,
            "lighting": {"ambient": 0.8, "diffuse": 0.5},
            "name": f"Plane ({int(request.plane_miller[0])},{int(request.plane_miller[1])},{int(request.plane_miller[2])})"
        }
        
        # Also add edge outline
        closed_points = list(intersection_points) + [intersection_points[0]]
        edge_color = f'dark{plane_color}' if plane_color in ['red', 'green', 'blue', 'cyan', 'magenta', 'orange'] else plane_color
        edge_trace = {
            "type": "scatter3d",
            "x": [p[0] for p in closed_points],
            "y": [p[1] for p in closed_points],
            "z": [p[2] for p in closed_points],
            "mode": "lines",
            "line": {"color": edge_color, "width": 4},
            "name": "Plane Edge",
            "showlegend": False
        }
        
        return {"traces": [plane_trace, edge_trace]}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process/crop")
async def crop_endpoint(request: CropRequest):
    """Crop data and return path to new file."""
    file_path = request.path
    if not os.path.isabs(file_path):
        file_path = os.path.join(DATA_DIR, file_path)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        # Load data using centralized loader
        data = _load_data_file(file_path)

        if not isinstance(data, xr.DataArray):
             raise HTTPException(status_code=400, detail="Data must be xarray for cropping")

        # Call crop logic
        cropped = crop_data(data, request.ranges)
        
        # Add processed flag
        cropped.attrs['is_adapt_processed'] = True
        cropped.attrs['crop_ranges'] = str(request.ranges)

        # Save to new temp file
        import h5py
        suffix = ".h5"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, prefix="cropped_") as tmp:
            save_path = tmp.name
            TEMP_FILES.append(save_path)
        
        try:
            cropped.to_netcdf(save_path, engine='h5netcdf')
        except Exception as e:
            with h5py.File(save_path, 'w') as f:
                ds = f.create_dataset('data', data=cropped.values)
                for coord in cropped.coords:
                    f.create_dataset(coord, data=cropped.coords[coord].values)
                for k, v in cropped.attrs.items():
                    try:
                        f.attrs[k] = v
                    except (TypeError, ValueError, OSError):
                        f.attrs[k] = str(v)
                        
        return {"filename": save_path}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files
# We'll create the static directory next
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
