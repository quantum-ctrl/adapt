import os
import sys
import glob
import numpy as np
import shutil
import atexit
from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tempfile
import xarray as xr

# Add shared module to path for session manager
_shared_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "shared"))
if _shared_path not in sys.path:
    sys.path.insert(0, _shared_path)

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
except ImportError as e:
    print(f"WARNING: Could not import loaders: {e}")
    # Fallback stubs if loaders are not available
    def load_hdf5_data(path):
        raise ImportError("Shared loaders not available")
    def load_ses_zip(path):
        raise ImportError("Shared loaders not available")
    def load_ibw(path):
        raise ImportError("Shared loaders not available")

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
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
                print(f"Cleaned up temp file: {tmp}")
        except Exception as e:
            print(f"Failed to clean up {tmp}: {e}")

atexit.register(cleanup_temp_files)


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
    # Search for .h5, .nxs, .hdf5, .zip (SES archives), and .ibw (Igor Binary Wave)
    extensions = ['*.h5', '*.nxs', '*.hdf5', '*.zip', '*.ibw']
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
            elif item.lower().endswith(('.h5', '.nxs', '.hdf5', '.ibw', '.zip')):
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
        # Load data using the appropriate loader (may return xarray.DataArray)
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.zip':
            result = load_ses_zip(file_path)
        elif ext == '.ibw':
            result = load_ibw(file_path)
        else:
            result = load_hdf5_data(file_path)

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

            return {
                "filename": path,
                "metadata": metadata,
                "axes": axes,
                "data_info": data_info
            }
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
        print("="*60)
        print("ERROR in /api/load endpoint:")
        print(f"File path: {file_path}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Full traceback:")
        print(error_details)
        print("="*60)
        raise HTTPException(status_code=500, detail=str(e))


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
        # Choose loader based on extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.zip':
            result = load_ses_zip(file_path)
        elif ext == '.ibw':
            result = load_ibw(file_path)
        else:
            result = load_hdf5_data(file_path)

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
        print("="*60)
        print("ERROR in /api/data endpoint:")
        print(f"File path: {file_path}")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        print("Full traceback:")
        print(error_details)
        print("="*60)
        raise HTTPException(status_code=500, detail=str(e))


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

# Mount static files
# We'll create the static directory next
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
