"""
tests/test_events.py
--------------------
Unit tests for EventManager (ENTRY/EXIT state machine, snapshots, DB records).
"""

import os
import shutil
import tempfile
import pytest
import numpy as np

from src.database import DatabaseManager
from src.tracker import STrack
from src.event_manager import EventManager


class TestEventManager:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "test_events.db")
        self.entry_dir = os.path.join(self.tmpdir, "entries")
        self.exit_dir = os.path.join(self.tmpdir, "exits")

        self.db = DatabaseManager(self.db_path)
        self.em = EventManager(
            db=self.db,
            entry_dir=self.entry_dir,
            exit_dir=self.exit_dir,
        )

        # Pre-seed a visitor in DB
        self.db.insert_visitor("visitor_001", np.zeros(512, dtype=np.float32))

    def teardown_method(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_entry_event_triggered_and_saved(self):
        track = STrack([10, 10, 60, 60], 0.9)
        track.face_id = "visitor_001"
        crop = np.ones((50, 50, 3), dtype=np.uint8) * 120

        ev = self.em.on_track_identified(track, crop)
        assert ev is not None
        assert ev["event_type"] == "ENTRY"
        assert ev["face_id"] == "visitor_001"
        assert os.path.exists(ev["image_path"])

        # Check DB record
        events = self.db.get_events_for_visitor("visitor_001")
        assert len(events) == 1
        assert events[0]["event_type"] == "ENTRY"
        assert events[0]["image_path"] == ev["image_path"]

    def test_no_duplicate_entry_while_active(self):
        track = STrack([10, 10, 60, 60], 0.9)
        track.face_id = "visitor_001"
        crop = np.ones((50, 50, 3), dtype=np.uint8) * 120

        ev1 = self.em.on_track_identified(track, crop)
        assert ev1 is not None

        # Same visitor in next frame -> no duplicate event
        ev2 = self.em.on_track_identified(track, crop)
        assert ev2 is None

        events = self.db.get_events_for_visitor("visitor_001")
        assert len(events) == 1

    def test_exit_event_triggered_and_saved(self):
        track = STrack([10, 10, 60, 60], 0.9)
        track.face_id = "visitor_001"
        crop = np.ones((50, 50, 3), dtype=np.uint8) * 120

        # 1. Entry
        self.em.on_track_identified(track, crop)
        assert self.em.get_active_count() == 1

        # 2. Exit when track is lost
        ev_exit = self.em.on_track_lost(track)
        assert ev_exit is not None
        assert ev_exit["event_type"] == "EXIT"
        assert self.em.get_active_count() == 0
        assert os.path.exists(ev_exit["image_path"])

        events = self.db.get_events_for_visitor("visitor_001")
        assert len(events) == 2
        assert events[0]["event_type"] == "ENTRY"
        assert events[1]["event_type"] == "EXIT"

    def test_flush_all_exits(self):
        # Visitor 1
        t1 = STrack([10, 10, 50, 50], 0.9)
        t1.face_id = "visitor_001"
        self.em.on_track_identified(t1, np.zeros((40, 40, 3), dtype=np.uint8))

        # Visitor 2
        self.db.insert_visitor("visitor_002", np.zeros(512, dtype=np.float32))
        t2 = STrack([60, 60, 100, 100], 0.85)
        t2.face_id = "visitor_002"
        self.em.on_track_identified(t2, np.zeros((40, 40, 3), dtype=np.uint8))

        assert self.em.get_active_count() == 2

        # End of stream flush
        flushed = self.em.flush_all_exits()
        assert len(flushed) == 2
        assert self.em.get_active_count() == 0

        all_events = self.db.get_all_events()
        assert len(all_events) == 4  # 2 ENTRYs + 2 EXITs
