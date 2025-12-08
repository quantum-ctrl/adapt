"""
Metadata formatting utilities for display in the UI.
"""

from typing import Any, Dict
import numpy as np


def format_value(value: Any, max_array_len: int = 5) -> str:
    """
    Format a single value for display.
    
    Args:
        value: Any value to format
        max_array_len: Maximum array elements to show
        
    Returns:
        Formatted string representation
    """
    if value is None:
        return "N/A"
    
    if isinstance(value, (np.ndarray, list)):
        arr = np.asarray(value)
        if arr.size == 0:
            return "[]"
        elif arr.size == 1:
            return f"{arr.flat[0]:.6g}" if np.issubdtype(arr.dtype, np.floating) else str(arr.flat[0])
        elif arr.size <= max_array_len:
            formatted = [f"{x:.4g}" if isinstance(x, float) else str(x) for x in arr.flat]
            return f"[{', '.join(formatted)}]"
        else:
            return f"Array shape={arr.shape}, dtype={arr.dtype}"
    
    if isinstance(value, float):
        if np.isnan(value):
            return "N/A"
        return f"{value:.6g}"
    
    if isinstance(value, dict):
        return f"Dict with {len(value)} keys"
    
    return str(value)


def format_metadata(meta: Dict[str, Any], indent: int = 0) -> str:
    """
    Format a metadata dictionary for display.
    
    Args:
        meta: Metadata dictionary
        indent: Indentation level
        
    Returns:
        Formatted multi-line string
    """
    lines = []
    prefix = "  " * indent
    
    # Priority keys to show first
    priority_keys = ['FileName', 'Type', 'hv', 'Temp', 'tltM', 'thtM', 'Epass', 'Mode', 'Pol']
    
    # Show priority keys first
    for key in priority_keys:
        if key in meta:
            value = meta[key]
            if key != 'meta':  # Skip nested meta for now
                lines.append(f"{prefix}{key}: {format_value(value)}")
    
    # Show remaining keys
    for key, value in sorted(meta.items()):
        if key not in priority_keys and key != 'meta' and key != 'raw_note':
            lines.append(f"{prefix}{key}: {format_value(value)}")
    
    # Handle nested 'meta' dict
    if 'meta' in meta and isinstance(meta['meta'], dict):
        nested = meta['meta']
        if nested:
            lines.append(f"{prefix}--- Additional Metadata ---")
            for key, value in sorted(nested.items()):
                if key not in ('IBW_Header', 'raw_note'):  # Skip very long fields
                    lines.append(f"{prefix}  {key}: {format_value(value)}")
    
    return "\n".join(lines)


def format_shape_dtype(data: np.ndarray) -> str:
    """
    Format shape and dtype information.
    
    Args:
        data: NumPy array
        
    Returns:
        Formatted string like "Shape: (100, 200), dtype: float32"
    """
    return f"Shape: {data.shape}, dtype: {data.dtype}"
