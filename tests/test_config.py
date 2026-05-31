import os
import unittest
from unittest.mock import patch


from shared.utils.config import (
    allow_filesystem_browse,
    get_browse_roots,
    get_data_dir,
    get_host,
    get_max_upload_size,
    get_port,
)


class ConfigTest(unittest.TestCase):
    def test_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(get_host(), "127.0.0.1")
            self.assertEqual(get_port(), 8000)
            self.assertEqual(get_max_upload_size(), 2_000_000_000)
            self.assertFalse(allow_filesystem_browse())

    def test_environment_overrides(self):
        env = {
            "ADAPT_HOST": "localhost",
            "ADAPT_PORT": "9000",
            "ADAPT_DATA_DIR": "/tmp/adapt-data",
            "ADAPT_MAX_UPLOAD_SIZE": "1024",
            "ADAPT_ALLOW_FILESYSTEM_BROWSE": "true",
            "ADAPT_BROWSE_ROOTS": os.pathsep.join(["/tmp", "~/data"]),
        }
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(get_host(), "localhost")
            self.assertEqual(get_port(), 9000)
            self.assertEqual(get_data_dir("/fallback"), "/tmp/adapt-data")
            self.assertEqual(get_max_upload_size(), 1024)
            self.assertTrue(allow_filesystem_browse())
            self.assertEqual(
                get_browse_roots("/fallback"),
                [os.path.abspath("/tmp"), os.path.abspath(os.path.expanduser("~/data"))],
            )


if __name__ == "__main__":
    unittest.main()
