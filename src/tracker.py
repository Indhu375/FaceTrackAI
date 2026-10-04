"""
src/tracker.py
--------------
Face tracking module for FaceTrackAI — Phase 2.

Implements ByteTrack association with a Kalman Filter to track faces across
frames, maintain temporal continuity, bridge missed detections, and associate
temporary Track IDs with persistent Face IDs.

Design & Principles
-------------------
1. Track ID != Face ID:
   - Track ID: short-lived, integer assigned per continuous trajectory.
   - Face ID: long-lived string (e.g. 'visitor_001') assigned by recognizer.
2. Two-stage ByteTrack matching:
   - Stage 1: High-confidence detections matched against confirmed active tracks.
   - Stage 2: Remaining low-confidence detections matched against unconfirmed/lost tracks
     to prevent dropping tracks during motion blur or partial occlusion.
3. Pure SciPy Hungarian algorithm (linear_sum_assignment) avoids MSVC compilation
   issues with 'lap' on Windows.
"""

from enum import Enum
from typing import List, Tuple, Dict, Any, Optional
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.logger import get_logger


class TrackState(Enum):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


# ---------------------------------------------------------------------------
# Kalman Filter for 2D Bounding Boxes
# ---------------------------------------------------------------------------

class KalmanBoxTracker:
    """
    Kalman filter tracking bounding box state:
    x = [cx, cy, s, h, v_cx, v_cy, v_s, v_h]^T
    where:
        cx, cy: center coordinates
        s     : aspect ratio (w / h)
        h     : height
        v_*   : velocity terms
    """

    def __init__(self, bbox: List[float]) -> None:
        # State dimension: 8, Measurement dimension: 4
        self.dim_x = 8
        self.dim_z = 4

        # State transition matrix F
        self.F = np.eye(self.dim_x)
        for i in range(4):
            self.F[i, i + 4] = 1.0  # dt = 1 frame

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x))
        for i in range(4):
            self.H[i, i] = 1.0

        # Measurement noise covariance R
        self.R = np.eye(self.dim_z) * 1.0
        self.R[2, 2] *= 10.0  # aspect ratio uncertainty

        # Process noise covariance Q
        self.Q = np.eye(self.dim_x)
        self.Q[4:, 4:] *= 0.01

        # Initial state covariance P
        self.P = np.eye(self.dim_x) * 10.0
        self.P[4:, 4:] *= 1000.0  # high uncertainty on initial velocities

        # Initial state x
        self.x = np.zeros((self.dim_x, 1))
        z = self._bbox_to_z(bbox)
        self.x[:4] = z

    @staticmethod
    def _bbox_to_z(bbox: List[float]) -> np.ndarray:
        """Convert [x1, y1, x2, y2] to [cx, cy, s, h]^T."""
        w = max(1.0, float(bbox[2] - bbox[0]))
        h = max(1.0, float(bbox[3] - bbox[1]))
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = w / h
        return np.array([[cx], [cy], [s], [h]], dtype=np.float32)

    @staticmethod
    def _z_to_bbox(x: np.ndarray) -> List[int]:
        """Convert [cx, cy, s, h] to [x1, y1, x2, y2]."""
        cx, cy, s, h = float(x[0, 0]), float(x[1, 0]), float(x[2, 0]), float(x[3, 0])
        h = max(1.0, h)
        s = max(0.01, s)
        w = s * h
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = int(round(cx + w / 2.0))
        y2 = int(round(cy + h / 2.0))
        return [x1, y1, x2, y2]

    def predict(self) -> List[int]:
        """Advance state by 1 time step and return predicted bounding box."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self._z_to_bbox(self.x)

    def update(self, bbox: List[float]) -> List[int]:
        """Update state with observed bounding box measurement."""
        z = self._bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self._z_to_bbox(self.x)

    def get_bbox(self) -> List[int]:
        """Return current estimated bounding box."""
        return self._z_to_bbox(self.x)


# ---------------------------------------------------------------------------
# Track Representation (STrack)
# ---------------------------------------------------------------------------

class STrack:
    """
    Represents an individual tracked face trajectory across frames.
    """

    _count = 0

    def __init__(self, bbox: List[int], score: float) -> None:
        STrack._count += 1
        self.track_id: int = STrack._count
        self.bbox: List[int] = list(map(int, bbox))
        self.score: float = float(score)

        self.kalman = KalmanBoxTracker(self.bbox)
        self.state: TrackState = TrackState.New

        self.is_activated: bool = False
        self.face_id: Optional[str] = None       # e.g. 'visitor_001'

        self.frame_id: int = 0
        self.start_frame: int = 0
        self.time_since_update: int = 0
        self.tracklet_len: int = 0

        self.first_crop: Optional[np.ndarray] = None
        self.last_crop: Optional[np.ndarray] = None

    @classmethod
    def reset_counter(cls) -> None:
        """Reset track ID counter (for tests or session restarts)."""
        cls._count = 0

    def activate(self, frame_id: int) -> None:
        """Activate a brand new track."""
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.tracklet_len = 1
        self.time_since_update = 0
        self.state = TrackState.Tracked
        self.is_activated = True

    def re_activate(self, new_bbox: List[int], new_score: float, frame_id: int) -> None:
        """Re-activate a lost track with a new measurement."""
        self.bbox = self.kalman.update(new_bbox)
        self.score = new_score
        self.frame_id = frame_id
        self.time_since_update = 0
        self.tracklet_len += 1
        self.state = TrackState.Tracked
        self.is_activated = True

    def predict(self) -> List[int]:
        """Predict the next state."""
        self.bbox = self.kalman.predict()
        self.time_since_update += 1
        return self.bbox

    def update(self, new_bbox: List[int], new_score: float, frame_id: int) -> None:
        """Update track with matching detection."""
        self.frame_id = frame_id
        self.tracklet_len += 1
        self.time_since_update = 0
        self.bbox = self.kalman.update(new_bbox)
        self.score = new_score
        self.state = TrackState.Tracked
        self.is_activated = True

    def mark_lost(self) -> None:
        """Mark track as temporarily lost."""
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        """Mark track as permanently terminated."""
        self.state = TrackState.Removed

    def to_dict(self) -> Dict[str, Any]:
        """Serialise track state to dictionary."""
        return {
            "track_id": self.track_id,
            "face_id": self.face_id,
            "bbox": self.bbox,
            "score": round(self.score, 3),
            "state": self.state.name,
            "tracklet_len": self.tracklet_len,
            "time_since_update": self.time_since_update,
        }


# ---------------------------------------------------------------------------
# Matching & Association Utilities
# ---------------------------------------------------------------------------

def calculate_iou(bbox1: List[int], bbox2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, bbox1[2] - bbox1[0]) * max(0, bbox1[3] - bbox1[1])
    area2 = max(0, bbox2[2] - bbox2[0]) * max(0, bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def compute_iou_cost_matrix(tracks: List[STrack], detections: List[Dict[str, Any]]) -> np.ndarray:
    """Compute (1 - IoU) cost matrix between tracks and detections."""
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou = calculate_iou(track.bbox, det["bbox"])
            cost_matrix[i, j] = 1.0 - iou
    return cost_matrix


def linear_assignment(
    cost_matrix: np.ndarray, thresh: float
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Perform linear sum assignment using SciPy Hungarian algorithm.

    Parameters
    ----------
    cost_matrix : np.ndarray
    thresh      : float  Maximum allowable cost (e.g. 1.0 - match_thresh)

    Returns
    -------
    matches           : List of (track_idx, det_idx)
    unmatched_tracks  : List of track indices
    unmatched_dets    : List of detection indices
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_tracks = set(range(cost_matrix.shape[0]))
    unmatched_dets = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= thresh:
            matches.append((r, c))
            unmatched_tracks.discard(r)
            unmatched_dets.discard(c)

    return matches, sorted(list(unmatched_tracks)), sorted(list(unmatched_dets))


# ---------------------------------------------------------------------------
# ByteTracker
# ---------------------------------------------------------------------------

class ByteTracker:
    """
    ByteTrack multi-object face tracker.

    Parameters
    ----------
    track_thresh       : float Minimum detection confidence to initiate a track.
    high_thresh        : float Threshold dividing high-conf from low-conf detections.
    match_thresh       : float Minimum IoU threshold to consider a match valid.
    max_missed_frames  : int   Frames to retain a track without detection before removal.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        high_thresh: float = 0.5,
        match_thresh: float = 0.3,
        max_missed_frames: int = 30,
    ) -> None:
        self.track_thresh = track_thresh
        self.high_thresh = high_thresh
        self.match_thresh = match_thresh
        self.max_missed_frames = max_missed_frames
        self.logger = get_logger()

        self.frame_id: int = 0
        self.tracked_tracks: List[STrack] = []
        self.lost_tracks: List[STrack] = []
        self.removed_tracks: List[STrack] = []

    def reset(self) -> None:
        """Reset tracker state and track IDs."""
        self.frame_id = 0
        self.tracked_tracks.clear()
        self.lost_tracks.clear()
        self.removed_tracks.clear()
        STrack.reset_counter()

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame: Optional[np.ndarray] = None,
    ) -> Tuple[List[STrack], List[STrack]]:
        """
        Update tracker with new frame detections.

        Parameters
        ----------
        detections : List of dicts, each with keys 'bbox': [x1,y1,x2,y2], 'confidence': float
        frame      : Optional full OpenCV frame (for extracting and caching crops)

        Returns
        -------
        Tuple of:
            active_tracks : List[STrack] currently active and tracked
            newly_lost    : List[STrack] tracks that just reached max_missed_frames this update
        """
        self.frame_id += 1
        newly_removed: List[STrack] = []

        # 1. Split detections into high and low confidence sets
        dets_high: List[Dict[str, Any]] = []
        dets_low: List[Dict[str, Any]] = []

        for d in detections:
            conf = float(d.get("confidence", 0.0))
            if conf >= self.high_thresh:
                dets_high.append(d)
            elif conf >= 0.1:
                dets_low.append(d)

        # 2. Predict positions for all existing tracks
        for track in self.tracked_tracks + self.lost_tracks:
            track.predict()

        # Pool of tracks to match in stage 1: active tracks + currently lost tracks
        track_pool = [t for t in self.tracked_tracks if t.is_activated] + list(self.lost_tracks)

        # 3. Stage 1: Associate high-confidence detections with active + lost tracks
        cost_matrix = compute_iou_cost_matrix(track_pool, dets_high)
        matches_1, u_tracks_1, u_dets_1 = linear_assignment(cost_matrix, 1.0 - self.match_thresh)

        for track_idx, det_idx in matches_1:
            track = track_pool[track_idx]
            det = dets_high[det_idx]
            if track.state == TrackState.Tracked:
                track.update(det["bbox"], det["confidence"], self.frame_id)
            else:
                track.re_activate(det["bbox"], det["confidence"], self.frame_id)
                if track in self.lost_tracks:
                    self.lost_tracks.remove(track)
                if track not in self.tracked_tracks:
                    self.tracked_tracks.append(track)

        # 4. Stage 2: Associate low-confidence detections with remaining unmatched tracks
        remaining_tracks = [track_pool[i] for i in u_tracks_1]
        cost_matrix_2 = compute_iou_cost_matrix(remaining_tracks, dets_low)
        matches_2, u_tracks_2, _ = linear_assignment(cost_matrix_2, 1.0 - 0.2)  # relaxed IoU threshold for low-conf

        for track_idx, det_idx in matches_2:
            track = remaining_tracks[track_idx]
            det = dets_low[det_idx]
            if track.state == TrackState.Tracked:
                track.update(det["bbox"], det["confidence"], self.frame_id)
            else:
                track.re_activate(det["bbox"], det["confidence"], self.frame_id)
                if track in self.lost_tracks:
                    self.lost_tracks.remove(track)
                if track not in self.tracked_tracks:
                    self.tracked_tracks.append(track)

        # 5. Deal with remaining unmatched tracks -> mark lost
        for track_idx in u_tracks_2:
            track = remaining_tracks[track_idx]
            if track.state != TrackState.Lost:
                track.mark_lost()
                if track in self.tracked_tracks:
                    self.tracked_tracks.remove(track)
                if track not in self.lost_tracks:
                    self.lost_tracks.append(track)

        # Check lost tracks for expiration
        for track in list(self.lost_tracks):
            if track.time_since_update > self.max_missed_frames:
                track.mark_removed()
                self.lost_tracks.remove(track)
                self.removed_tracks.append(track)
                newly_removed.append(track)
                self.logger.debug(f"TRACK_REMOVED | track_id={track.track_id} face_id={track.face_id}")

        # 6. Initialize new tracks for unmatched high-confidence detections
        for det_idx in u_dets_1:
            det = dets_high[det_idx]
            if det["confidence"] >= self.track_thresh:
                new_track = STrack(det["bbox"], det["confidence"])
                new_track.activate(self.frame_id)
                self.tracked_tracks.append(new_track)
                self.logger.debug(f"TRACK_CREATED | track_id={new_track.track_id} conf={new_track.score}")

        # Return only active tracked tracks and newly removed tracks
        active_tracks = [t for t in self.tracked_tracks if t.state == TrackState.Tracked]
        return active_tracks, newly_removed
