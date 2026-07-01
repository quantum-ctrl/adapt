"""
Shared PyQtGraph plotting helpers, used by both ViewerPanel and ComparePanel
so the colormap implementation isn't duplicated between them.
"""

import numpy as np
import pyqtgraph as pg

from ADAPT_browser.utils.logger import logger


def apply_colormap_to_image(image_item: pg.ImageItem, cmap_name: str, invert: bool = False):
    """Apply a matplotlib colormap to a pyqtgraph ImageItem as a lookup table."""
    try:
        from matplotlib import colormaps
        cmap = colormaps.get_cmap(cmap_name)

        if invert:
            lut = (cmap(np.linspace(1, 0, 256)) * 255).astype(np.uint8)
        else:
            lut = (cmap(np.linspace(0, 1, 256)) * 255).astype(np.uint8)

        image_item.setLookupTable(lut)
    except Exception as e:
        logger.warning(f"Failed to apply colormap {cmap_name}: {e}")
