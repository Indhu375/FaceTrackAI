"""
tests/test_database.py
-----------------------
Unit tests for DatabaseManager.

Uses an in-memory SQLite database so no files are written during testing.
"""

import sys
import os
import unittest
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.logger import setup_logger

# Set up a minimal logger pointing to /dev/null (or NUL on Windows)
setup_logger(log_file="logs/events.log")

from src.database import DatabaseManager


class TestDatabaseManager(unittest.TestCase):
    """Tests for the SQLite persistence layer."""

    def setUp(self) -> None:
        # Use a temporary file so each test gets a fresh database
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.db = DatabaseManager(db_path=self.tmp.name)

    def tearDown(self) -> None:
        self.db.close()
        os.unlink(self.tmp.name)

    # ------------------------------------------------------------------
    # Visitor tests
    # ------------------------------------------------------------------

    def test_insert_and_retrieve_visitor(self) -> None:
        embedding = np.random.rand(512).astype(np.float32)
        self.db.insert_visitor("visitor_001", embedding, "2026-01-01 10:00:00")
        visitors = self.db.get_all_visitors()
        self.assertEqual(len(visitors), 1)
        self.assertEqual(visitors[0]["face_id"], "visitor_001")

    def test_embedding_round_trip(self) -> None:
        """Embedding stored and loaded must be numerically identical."""
        original = np.random.rand(512).astype(np.float32)
        self.db.insert_visitor("visitor_001", original)
        loaded = self.db.get_all_visitors()[0]["embedding"]
        np.testing.assert_array_almost_equal(original, loaded, decimal=5)

    def test_unique_visitor_count(self) -> None:
        for i in range(1, 4):
            emb = np.random.rand(512).astype(np.float32)
            self.db.insert_visitor(f"visitor_{i:03d}", emb)
        self.assertEqual(self.db.get_visitor_count(), 3)

    def test_duplicate_face_id_ignored(self) -> None:
        emb = np.random.rand(512).astype(np.float32)
        self.db.insert_visitor("visitor_001", emb)
        self.db.insert_visitor("visitor_001", emb)  # should not raise
        self.assertEqual(self.db.get_visitor_count(), 1)

    def test_get_next_face_id_sequential(self) -> None:
        self.assertEqual(self.db.get_next_face_id(), "visitor_001")
        emb = np.random.rand(512).astype(np.float32)
        self.db.insert_visitor("visitor_001", emb)
        self.assertEqual(self.db.get_next_face_id(), "visitor_002")

    def test_face_id_exists(self) -> None:
        emb = np.random.rand(512).astype(np.float32)
        self.db.insert_visitor("visitor_001", emb)
        self.assertTrue(self.db.face_id_exists("visitor_001"))
        self.assertFalse(self.db.face_id_exists("visitor_999"))

    # ------------------------------------------------------------------
    # Event tests
    # ------------------------------------------------------------------

    def test_insert_and_retrieve_event(self) -> None:
        emb = np.random.rand(512).astype(np.float32)
        self.db.insert_visitor("visitor_001", emb)
        self.db.insert_event("visitor_001", "ENTRY", "2026-01-01 10:00:00", "/path/img.jpg")
        events = self.db.get_events_for_visitor("visitor_001")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["event_type"], "ENTRY")

    def test_entry_and_exit_events(self) -> None:
        emb = np.random.rand(512).astype(np.float32)
        self.db.insert_visitor("visitor_001", emb)
        self.db.insert_event("visitor_001", "ENTRY")
        self.db.insert_event("visitor_001", "EXIT")
        events = self.db.get_events_for_visitor("visitor_001")
        event_types = [e["event_type"] for e in events]
        self.assertIn("ENTRY", event_types)
        self.assertIn("EXIT", event_types)

    def test_unique_count_with_repeated_appearances(self) -> None:
        """Same visitor appearing multiple times must still count as 1."""
        emb = np.random.rand(512).astype(np.float32)
        self.db.insert_visitor("visitor_001", emb)
        # Simulate multiple events for the same person
        for _ in range(5):
            self.db.insert_event("visitor_001", "ENTRY")
        self.assertEqual(self.db.get_visitor_count(), 1)


if __name__ == "__main__":
    unittest.main()
