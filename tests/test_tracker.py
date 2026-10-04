"""
tests/test_tracker.py
---------------------
Unit tests for the ByteTracker and KalmanBoxTracker modules (Phase 2).
"""

import pytest
import numpy as np

from src.tracker import (
    ByteTracker,
    KalmanBoxTracker,
    STrack,
    TrackState,
    calculate_iou,
    compute_iou_cost_matrix,
    linear_assignment,
)


class TestKalmanBoxTracker:
    def test_predict_and_update(self):
        bbox = [100, 100, 200, 200]
        kt = KalmanBoxTracker(bbox)

        # Initial bbox should match input closely
        pred_box = kt.predict()
        assert len(pred_box) == 4
        assert abs(pred_box[0] - 100) < 5
        assert abs(pred_box[1] - 100) < 5

        # Update with slight motion
        new_bbox = [105, 105, 205, 205]
        updated = kt.update(new_bbox)
        assert len(updated) == 4
        assert updated[0] > 95 and updated[2] < 215


class TestIoUCalculation:
    def test_identical_boxes(self):
        b1 = [10, 10, 50, 50]
        assert calculate_iou(b1, b1) == pytest.approx(1.0)

    def test_non_overlapping_boxes(self):
        b1 = [0, 0, 10, 10]
        b2 = [20, 20, 30, 30]
        assert calculate_iou(b1, b2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        b1 = [0, 0, 20, 20]  # area 400
        b2 = [10, 0, 30, 20] # area 400, inter 10x20=200, union=600
        assert calculate_iou(b1, b2) == pytest.approx(200.0 / 600.0, rel=1e-2)


class TestByteTracker:
    def setup_method(self):
        STrack.reset_counter()
        self.tracker = ByteTracker(
            track_thresh=0.5,
            high_thresh=0.5,
            match_thresh=0.3,
            max_missed_frames=3,
        )

    def test_single_track_creation(self):
        dets = [{"bbox": [100, 100, 200, 200], "confidence": 0.85}]
        active, removed = self.tracker.update(dets)

        assert len(active) == 1
        assert len(removed) == 0
        assert active[0].track_id == 1
        assert active[0].state == TrackState.Tracked

    def test_track_continuity_across_frames(self):
        # Frame 1
        dets1 = [{"bbox": [100, 100, 200, 200], "confidence": 0.9}]
        active1, _ = self.tracker.update(dets1)
        track_id_1 = active1[0].track_id

        # Frame 2 (moving slightly)
        dets2 = [{"bbox": [104, 102, 204, 202], "confidence": 0.88}]
        active2, _ = self.tracker.update(dets2)

        assert len(active2) == 1
        assert active2[0].track_id == track_id_1  # Identity preserved!

    def test_track_survival_and_expiration(self):
        # Frame 1: Detected
        dets = [{"bbox": [100, 100, 200, 200], "confidence": 0.9}]
        active, _ = self.tracker.update(dets)
        tid = active[0].track_id

        # Frames 2, 3: Missed (occlusion / skip), max_missed_frames=3
        _, removed_2 = self.tracker.update([])
        assert len(removed_2) == 0

        _, removed_3 = self.tracker.update([])
        assert len(removed_3) == 0

        # Frame 4: Track re-appears!
        dets_reappear = [{"bbox": [108, 105, 208, 205], "confidence": 0.85}]
        active_reappear, removed_4 = self.tracker.update(dets_reappear)
        assert len(removed_4) == 0
        assert len(active_reappear) == 1
        assert active_reappear[0].track_id == tid  # Track recovered!

        # Miss 4 consecutive frames -> must expire
        for _ in range(3):
            self.tracker.update([])
        _, final_removed = self.tracker.update([])
        assert len(final_removed) == 1
        assert final_removed[0].track_id == tid
