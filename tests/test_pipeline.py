"""
tests/test_pipeline.py
----------------------
Unit tests for Pipeline end-to-end integration (Phase 2).
"""

import os
import shutil
import tempfile
import pytest
import numpy as np

from src.database import DatabaseManager
from src.detector import FaceDetector
from src.tracker import ByteTracker
from src.recognizer import FaceRecognizer
from src.visitor_manager import VisitorManager
from src.event_manager import EventManager
from src.pipeline import Pipeline


class TestPipeline:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmpdir, "pipeline_test.db")
        self.entry_dir = os.path.join(self.tmpdir, "entries")
        self.exit_dir = os.path.join(self.tmpdir, "exits")

        self.db = DatabaseManager(self.db_path)
        self.detector = FaceDetector(confidence=0.5, skip_frames=1)
        self.tracker = ByteTracker(max_missed_frames=5)
        self.recognizer = FaceRecognizer(similarity_threshold=0.5)
        self.recognizer._backend = "stub"  # deterministic test stub
        self.visitor_manager = VisitorManager(self.db, self.recognizer)
        self.event_manager = EventManager(self.db, self.entry_dir, self.exit_dir)

        config = {
            "video_source": "dummy.mp4",
            "detection": {"confidence_threshold": 0.5, "detection_skip_frames": 1},
            "recognition": {"similarity_threshold": 0.5},
            "tracking": {"max_missed_frames": 5},
            "storage": {"entry_directory": self.entry_dir, "exit_directory": self.exit_dir},
            "display": {"show_window": False, "save_output_video": False},
        }

        self.pipeline = Pipeline(
            config=config,
            detector=self.detector,
            tracker=self.tracker,
            recognizer=self.recognizer,
            visitor_manager=self.visitor_manager,
            event_manager=self.event_manager,
        )

    def teardown_method(self):
        self.db.close()
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_pipeline_process_frame_and_finish(self):
        # Create a dummy frame (480, 640, 3)
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 100

        res = self.pipeline.process_frame(frame, frame_idx=1)
        assert "annotated_frame" in res
        assert res["annotated_frame"].shape == frame.shape
        assert "active_tracks" in res
        assert "unique_count" in res

        # Finish pipeline
        flushed = self.pipeline.finish()
        assert isinstance(flushed, list)
