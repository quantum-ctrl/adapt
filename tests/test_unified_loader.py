import os
import tempfile
import unittest

import h5py
import numpy as np


from shared.loaders import get_file_type, is_supported_file, load_data_file


class UnifiedLoaderTest(unittest.TestCase):
    def test_supported_file_helpers_include_nxs(self):
        self.assertTrue(is_supported_file("sample.nxs"))
        self.assertEqual(get_file_type("sample.nxs"), "HDF5")

    def test_sis_hdf5_fallback(self):
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.close(fd)
        try:
            with h5py.File(path, "w") as h5:
                dataset = h5.create_dataset(
                    "/Electron Analyzer/Image Data",
                    data=np.ones((3, 4), dtype=np.float32),
                )
                dataset.attrs["Axis0.Scale"] = np.array([0.0, 0.1])
                dataset.attrs["Axis1.Scale"] = np.array([-1.0, 0.5])

            loaded = load_data_file(path)
            self.assertEqual(loaded.dims, ("energy", "angle"))
            self.assertEqual(loaded.shape, (4, 3))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_unsupported_extension_raises(self):
        fd, path = tempfile.mkstemp(suffix=".txt")
        os.close(fd)
        try:
            with self.assertRaises(ValueError):
                load_data_file(path)
        finally:
            if os.path.exists(path):
                os.remove(path)


if __name__ == "__main__":
    unittest.main()
