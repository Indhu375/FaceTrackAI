"""
tests/test_config.py
--------------------
Verify that config.json loads correctly and contains all required keys.
"""

import json
import os
import sys
import unittest

# Make sure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestConfig(unittest.TestCase):
    """Tests for config.json validity."""

    CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.json"))

    def setUp(self) -> None:
        with open(self.CONFIG_PATH, "r", encoding="utf-8") as f:
            self.config = json.load(f)

    def test_file_exists(self) -> None:
        self.assertTrue(os.path.exists(self.CONFIG_PATH), "config.json must exist")

    def test_top_level_keys(self) -> None:
        required = {
            "video_source", "detection", "recognition",
            "tracking", "database", "logging", "storage", "display",
        }
        for key in required:
            self.assertIn(key, self.config, f"Missing top-level key: {key}")

    def test_detection_keys(self) -> None:
        det = self.config["detection"]
        for key in ("model", "confidence_threshold", "detection_skip_frames"):
            self.assertIn(key, det, f"Missing detection key: {key}")

    def test_detection_skip_frames_is_int(self) -> None:
        skip = self.config["detection"]["detection_skip_frames"]
        self.assertIsInstance(skip, int, "detection_skip_frames must be an integer")
        self.assertGreater(skip, 0, "detection_skip_frames must be > 0")

    def test_confidence_threshold_range(self) -> None:
        conf = self.config["detection"]["confidence_threshold"]
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_similarity_threshold_range(self) -> None:
        sim = self.config["recognition"]["similarity_threshold"]
        self.assertGreaterEqual(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_database_path_present(self) -> None:
        self.assertIn("path", self.config["database"])

    def test_log_file_present(self) -> None:
        self.assertIn("log_file", self.config["logging"])


if __name__ == "__main__":
    unittest.main()
