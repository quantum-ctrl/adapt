import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shared.session import app_state


class AppStateTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        state_file = Path(self._tmpdir.name) / "app_state.json"
        self._patcher = patch.object(app_state, "STATE_FILE", state_file)
        self._patcher.start()
        self.addCleanup(self._patcher.stop)

    def test_recent_folders_mru_and_dedupe(self):
        app_state.add_recent_folder("/data/a")
        app_state.add_recent_folder("/data/b")
        app_state.add_recent_folder("/data/a")  # re-add moves to front

        self.assertEqual(
            app_state.get_recent_folders(),
            ["/data/a", "/data/b"],
        )

    def test_recent_folders_truncates_to_max_items(self):
        for i in range(5):
            app_state.add_recent_folder(f"/data/{i}", max_items=3)

        self.assertEqual(
            app_state.get_recent_folders(),
            ["/data/4", "/data/3", "/data/2"],
        )

    def test_collection_lifecycle(self):
        self.assertTrue(app_state.create_collection("Favorites"))
        self.assertFalse(app_state.create_collection("Favorites"))  # already exists

        self.assertTrue(app_state.add_file_to_collection("Favorites", "/data/scan1.h5"))
        self.assertTrue(app_state.add_file_to_collection("Favorites", "/data/scan1.h5"))  # no dup
        self.assertEqual(
            app_state.get_collections()["Favorites"]["files"],
            ["/data/scan1.h5"],
        )

        self.assertTrue(app_state.remove_file_from_collection("Favorites", "/data/scan1.h5"))
        self.assertEqual(app_state.get_collections()["Favorites"]["files"], [])

        self.assertTrue(app_state.delete_collection("Favorites"))
        self.assertFalse(app_state.delete_collection("Favorites"))  # already gone

    def test_operations_on_missing_collection_return_false(self):
        self.assertFalse(app_state.add_file_to_collection("Nope", "/data/x.h5"))
        self.assertFalse(app_state.remove_file_from_collection("Nope", "/data/x.h5"))


if __name__ == "__main__":
    unittest.main()
