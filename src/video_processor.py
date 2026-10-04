"""
src/video_processor.py
----------------------
Frame-level orchestration for FaceTrackAI — Phase 2.

Opens the video source (file or RTSP), feeds frames through the pipeline
(Detector -> Tracker -> Recognizer -> EventManager), and manages
preview display and video recording.
"""

import cv2
import os
import numpy as np
from typing import Any, Dict, Optional

from src.detector import FaceDetector
from src.visitor_manager import VisitorManager
from src.pipeline import Pipeline
from src.logger import get_logger


class VideoProcessor:
    """
    Drives the per-frame processing loop.

    Parameters
    ----------
    config          : dict                  Full config.json content.
    pipeline        : Optional[Pipeline]    Phase 2 end-to-end pipeline.
    detector        : Optional[FaceDetector] Phase 1 fallback.
    visitor_manager : Optional[VisitorManager] Phase 1 fallback.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        pipeline: Optional[Pipeline] = None,
        detector: Optional[FaceDetector] = None,
        visitor_manager: Optional[VisitorManager] = None,
    ) -> None:
        self.config = config
        self.pipeline = pipeline
        self.detector = detector
        self.visitor_manager = visitor_manager
        self.logger = get_logger()

        self.video_source: str = config["video_source"]
        self.show_window: bool = config.get("display", {}).get("show_window", True)
        self.save_output: bool = config.get("display", {}).get("save_output_video", True)

        self._cap: Optional[cv2.VideoCapture] = None
        self._writer: Optional[cv2.VideoWriter] = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the video source and process frames until completion or Ctrl-C."""
        self._cap = self._open_video_source()
        if self._cap is None:
            return

        if self.save_output:
            self._writer = self._create_video_writer()

        self.logger.info(f"VIDEO_PROCESSOR | Processing started | source={self.video_source}")

        try:
            self._frame_loop()
        except KeyboardInterrupt:
            self.logger.info("VIDEO_PROCESSOR | Interrupted by user")
        finally:
            if self.pipeline:
                self.pipeline.finish()
            self._release()

    def _frame_loop(self) -> None:
        """Core frame-processing loop."""
        frame_idx = 0
        while True:
            ret, frame = self._cap.read()
            if not ret or frame is None:
                self.logger.info("VIDEO_PROCESSOR | End of video stream")
                break

            frame_idx += 1

            if self.pipeline:
                # --- Phase 2: Pipeline execution --------------------------
                res = self.pipeline.process_frame(frame, frame_idx=frame_idx)
                annotated = res["annotated_frame"]
            else:
                # --- Phase 1: Fallback execution -------------------------
                detections = self.detector.detect(frame) if self.detector else []
                face_ids = []
                for det in detections:
                    bbox = det["bbox"]
                    conf = det["confidence"]
                    face_crop = FaceDetector.crop_face(frame, bbox)
                    if face_crop is None:
                        continue
                    face_id = self.visitor_manager.identify(face_crop) if self.visitor_manager else None
                    if face_id:
                        face_ids.append((face_id, bbox, conf))
                annotated = self._annotate_frame(frame.copy(), face_ids)

            # --- Display / write ----------------------------------------
            if self.show_window:
                cv2.imshow("FaceTrackAI", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.logger.info("VIDEO_PROCESSOR | 'q' pressed – stopping")
                    break

            if self._writer:
                self._writer.write(annotated)

        unique_count = (
            self.pipeline.visitor_manager.unique_visitor_count()
            if self.pipeline
            else (self.visitor_manager.unique_visitor_count() if self.visitor_manager else 0)
        )
        self.logger.info(
            f"VIDEO_PROCESSOR | Finished | unique_visitors={unique_count}"
        )

    # ------------------------------------------------------------------
    # Phase 1 Fallback Annotation
    # ------------------------------------------------------------------

    def _annotate_frame(
        self,
        frame: np.ndarray,
        face_ids: list,
    ) -> np.ndarray:
        """Draw bounding boxes, face IDs, and summary overlay on the frame."""
        for face_id, bbox, conf in face_ids:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{face_id} ({conf:.2f})"
            cv2.putText(frame, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        return frame

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def _open_video_source(self) -> Optional[cv2.VideoCapture]:
        """Open the video file or RTSP stream. Returns None on failure."""
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.logger.error(f"VIDEO_PROCESSOR | Cannot open source | {self.video_source}")
            return None
        self.logger.info(f"VIDEO_PROCESSOR | Source opened | {self.video_source}")
        return cap

    def _create_video_writer(self) -> Optional[cv2.VideoWriter]:
        """Create an output VideoWriter for saving annotated video."""
        out_dir = "outputs/annotated"
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "demo_output.mp4")

        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        self.logger.info(f"VIDEO_PROCESSOR | Output video → {out_path}")
        return writer

    def _release(self) -> None:
        """Release video capture and writer resources."""
        if self._cap:
            self._cap.release()
        if self._writer:
            self._writer.release()
        cv2.destroyAllWindows()
        self.logger.info("VIDEO_PROCESSOR | Resources released")
