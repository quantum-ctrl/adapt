import xarray as xr

def crop_data(data: xr.DataArray, ranges: dict) -> xr.DataArray:
    """
    Crop xarray DataArray based on index ranges.
    
    Args:
        data: Input DataArray
        ranges: Dictionary of start/end indices for each dimension.
                Keys should be 'x', 'y', 'z' corresponding to dims.
                Expected format: {'x': [start, end], 'y': [start, end], ...}
                
                Mapping logic (consistent with Visualizer):
                - 2D: y=dim0 (Energy), x=dim1 (Angle)
                - 3D: y=dim0 (Energy), x=dim1 (Angle), z=dim2 (Scan)
                
    Returns:
        Cropped DataArray
    """
    
    # Identify dimensions
    dims = data.dims
    slices = {}
    
    # Helper to create slice object
    def make_slice(rng):
        start = int(rng[0])
        end = int(rng[1])
        return slice(start, end + 1) # +1 because python slice is exclusive, but frontend indices are inclusive
    
    if data.ndim == 2:
        # dim0 = y (Energy), dim1 = x (Angle)
        if 'y' in ranges:
            slices[dims[0]] = make_slice(ranges['y'])
        if 'x' in ranges:
            slices[dims[1]] = make_slice(ranges['x'])
            
    elif data.ndim == 3:
        # dim0 = y (Energy), dim1 = x (Angle), dim2 = z (Scan)
        # Note: This index mapping [0, 1, 2] -> [y, x, z] matches visualizer.js logic
        if 'y' in ranges:
            slices[dims[0]] = make_slice(ranges['y'])
        if 'x' in ranges:
            slices[dims[1]] = make_slice(ranges['x'])
        if 'z' in ranges:
            slices[dims[2]] = make_slice(ranges['z'])
            
    # Apply slicing
    cropped = data.isel(**slices)
    
    return cropped
