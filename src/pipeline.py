"""
src/pipeline.py
---------------
Full pipeline orchestration for FaceTrackAI — Phase 2.

Integrates:
    Input Frame
        |
        v
    FaceDetector (YOLOv8)
        |
        v
    ByteTracker (Kalman + Hungarian)
        |
        v
    FaceRecognizer (InsightFace ArcFace)
        |
        v
    VisitorManager (Identity resolution & SQLite registry)
        |
        v
    EventManager (ENTRY / EXIT state machine & snapshot saving)
        |
        v
    Annotator & Dashboard Overlay
"""

from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

from src.detector import FaceDetector
from src.tracker import ByteTracker, STrack
from src.recognizer import FaceRecognizer
from src.visitor_manager import VisitorManager
from src.event_manager import EventManager
from src.logger import get_logger


class Pipeline:
    """
    End-to-end orchestration pipeline for FaceTrackAI.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        detector: FaceDetector,
        tracker: ByteTracker,
        recognizer: FaceRecognizer,
        visitor_manager: VisitorManager,
        event_manager: EventManager,
    ) -> None:
        self.config = config
        self.detector = detector
        self.tracker = tracker
        self.recognizer = recognizer
        self.visitor_manager = visitor_manager
        self.event_manager = event_manager
        self.logger = get_logger()

    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Process a single video frame through the complete pipeline.

        Returns
        -------
        dict with keys:
            'annotated_frame' : np.ndarray
            'active_tracks'   : List[STrack]
            'active_visitors' : List[str]
            'unique_count'    : int
            'events'          : List[Dict]
        """
        # 1. Detection
        detections = self.detector.detect(frame)

        # 2. Tracking (ByteTrack update)
        active_tracks, newly_removed = self.tracker.update(detections, frame)

        triggered_events: List[Dict[str, Any]] = []

        # 3. Handle removed tracks -> EXIT events
        for lost_track in newly_removed:
            ev = self.event_manager.on_track_lost(lost_track)
            if ev:
                triggered_events.append(ev)

        # 4. Recognition & Identity resolution for active tracks
        for track in active_tracks:
            # If identity not yet assigned, run recognition
            if track.face_id is None:
                face_crop = FaceDetector.crop_face(frame, track.bbox)
                if face_crop is not None and face_crop.size > 0:
                    face_id = self.visitor_manager.identify(face_crop)
                    if face_id:
                        track.face_id = face_id
                        ev = self.event_manager.on_track_identified(track, face_crop)
                        if ev:
                            triggered_events.append(ev)
            else:
                # Track identity is already confirmed; update crop for snapshot tracking
                face_crop = FaceDetector.crop_face(frame, track.bbox)
                if face_crop is not None and face_crop.size > 0:
                    self.event_manager.on_track_identified(track, face_crop)

        # 5. Visual Annotation & Analytics Overlay
        annotated = self._annotate(frame, active_tracks, frame_idx)

        return {
            "annotated_frame": annotated,
            "active_tracks": active_tracks,
            "active_visitors": self.event_manager.get_active_visitors(),
            "unique_count": self.visitor_manager.unique_visitor_count(),
            "events": triggered_events,
        }

    def finish(self) -> List[Dict[str, Any]]:
        """
        Finalise pipeline at end-of-stream.
        Flushes remaining active tracks as EXIT events.
        """
        flushed = self.event_manager.flush_all_exits()
        self.logger.info(
            f"PIPELINE | Session finished | Flushed {len(flushed)} exit event(s) | "
            f"Total unique visitors={self.visitor_manager.unique_visitor_count()}"
        )
        return flushed

    # ------------------------------------------------------------------
    # Rendering & Annotation
    # ------------------------------------------------------------------

    def _annotate(
        self,
        frame: np.ndarray,
        tracks: List[STrack],
        frame_idx: int,
    ) -> np.ndarray:
        """Render modern bounding boxes, labels, and statistics dashboard."""
        out = frame.copy()
        h, w = out.shape[:2]

        # Draw bounding boxes and tags
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            if track.face_id:
                # Confirmed visitor: Green
                color = (46, 204, 113)  # Emerald green
                label = f"T:{track.track_id} | {track.face_id}"
            else:
                # Tracking but pending identification: Amber/Orange
                color = (0, 165, 255)
                label = f"T:{track.track_id} | Detecting..."

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label banner
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.5
            thickness = 1
            (lw, lh), baseline = cv2.getTextSize(label, font, scale, thickness)
            cv2.rectangle(
                out,
                (x1, max(0, y1 - lh - 8)),
                (x1 + lw + 8, y1),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (x1 + 4, y1 - 4),
                font,
                scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        # Semi-transparent top HUD dashboard
        hud_h = 42
        hud_bg = out[0:hud_h, 0:w].copy()
        dark_overlay = np.zeros_like(hud_bg)
        cv2.addWeighted(dark_overlay, 0.7, hud_bg, 0.3, 0, hud_bg)
        out[0:hud_h, 0:w] = hud_bg

        # Top border accent line
        cv2.line(out, (0, hud_h), (w, hud_h), (52, 152, 219), 2)

        # HUD Text
        unique_cnt = self.visitor_manager.unique_visitor_count()
        active_cnt = self.event_manager.get_active_count()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 1

        hud_text_left = f"FaceTrackAI Phase 2  |  Frame: {frame_idx}"
        hud_text_right = f"Active: {active_cnt}  |  Total Unique Visitors: {unique_cnt}"

        cv2.putText(
            out,
            hud_text_left,
            (14, 26),
            font,
            font_scale,
            (220, 220, 220),
            thickness,
            cv2.LINE_AA,
        )
        (rw, _), _ = cv2.getTextSize(hud_text_right, font, font_scale, thickness)
        cv2.putText(
            out,
            hud_text_right,
            (w - rw - 14, 26),
            font,
            font_scale,
            (46, 204, 113),
            thickness,
            cv2.LINE_AA,
        )

        return out
