import contextlib
import io
import unittest
from unittest.mock import patch

import adapt_cli


class AdaptCliTest(unittest.TestCase):
    def test_main_starts_edit_and_browser_by_default(self):
        with patch("adapt_cli.run_both", return_value=0) as run_both:
            result = adapt_cli.main([])

        self.assertEqual(result, 0)
        run_both.assert_called_once_with(None, adapt_cli.DEFAULT_HOST, adapt_cli.DEFAULT_PORT)

    def test_main_passes_initial_folder_host_and_port_to_combined_launcher(self):
        with patch("adapt_cli.run_both", return_value=0) as run_both:
            result = adapt_cli.main(["/some/folder", "--host", "127.0.0.1", "--port", "8001"])

        self.assertEqual(result, 0)
        run_both.assert_called_once_with("/some/folder", "127.0.0.1", 8001)

    def test_legacy_commands_show_migration_error(self):
        for legacy_command in ("browser", "edit", "both"):
            with self.subTest(legacy_command=legacy_command):
                stderr = io.StringIO()
                with patch("adapt_cli.run_both") as run_both:
                    with contextlib.redirect_stderr(stderr):
                        with self.assertRaises(SystemExit) as raised:
                            adapt_cli.main([legacy_command])

                self.assertEqual(raised.exception.code, 2)
                self.assertIn(f"`adapt {legacy_command}` is no longer supported", stderr.getvalue())
                self.assertIn("uv run adapt", stderr.getvalue())
                run_both.assert_not_called()


if __name__ == "__main__":
    unittest.main()
