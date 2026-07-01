import os
import tempfile
import unittest

from ADAPT_browser.core import ratings


class RatingsTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        self.folder = self._tmpdir.name
        self.filepath = os.path.join(self.folder, "scan1.h5")

    def test_defaults_when_no_sidecar(self):
        self.assertEqual(ratings.get_rating(self.filepath), {"rating": 0, "rejected": False})

    def test_set_and_get_rating(self):
        self.assertTrue(ratings.set_rating(self.filepath, 4))
        self.assertEqual(ratings.get_rating(self.filepath), {"rating": 4, "rejected": False})

    def test_rating_is_clamped(self):
        ratings.set_rating(self.filepath, 99)
        self.assertEqual(ratings.get_rating(self.filepath)["rating"], 5)

        ratings.set_rating(self.filepath, -3)
        self.assertEqual(ratings.get_rating(self.filepath)["rating"], 0)

    def test_set_rejected_preserves_rating(self):
        ratings.set_rating(self.filepath, 3)
        ratings.set_rejected(self.filepath, True)
        self.assertEqual(ratings.get_rating(self.filepath), {"rating": 3, "rejected": True})

    def test_ratings_are_scoped_per_folder(self):
        other_folder = os.path.join(self.folder, "sub")
        os.makedirs(other_folder)
        other_file = os.path.join(other_folder, "scan1.h5")  # same filename, different folder

        ratings.set_rating(self.filepath, 5)
        self.assertEqual(ratings.get_rating(other_file)["rating"], 0)

    def test_sidecar_is_valid_json_in_folder(self):
        ratings.set_rating(self.filepath, 2)
        sidecar = os.path.join(self.folder, ratings.SIDECAR_NAME)
        self.assertTrue(os.path.isfile(sidecar))
        self.assertEqual(
            ratings.load_folder_ratings(self.folder),
            {"scan1.h5": {"rating": 2, "rejected": False}},
        )


if __name__ == "__main__":
    unittest.main()
