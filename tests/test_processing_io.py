import os
import sys
import unittest

import numpy as np
import xarray as xr


sys.path.insert(0, os.path.abspath("shared"))

from loaders import load_data_file  # noqa: E402
from processing.io import save_processed_dataarray  # noqa: E402


class ProcessingIOTest(unittest.TestCase):
    def test_save_processed_dataarray_can_be_loaded(self):
        data = xr.DataArray(
            np.arange(6, dtype=np.float32).reshape(2, 3),
            dims=("energy", "angle"),
            coords={"energy": [0.0, 1.0], "angle": [-1.0, 0.0, 1.0]},
            attrs={"is_adapt_processed": True, "nested": {"value": 1}},
            name="intensity",
        )

        path = save_processed_dataarray(data, prefix="adapt_test_")
        try:
            loaded = load_data_file(path)
            self.assertEqual(loaded.dims, ("energy", "angle"))
            self.assertEqual(loaded.shape, (2, 3))
            np.testing.assert_array_equal(loaded.values, data.values)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
