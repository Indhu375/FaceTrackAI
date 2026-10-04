"""
src/detector.py
---------------
YOLO-based face detection module for FaceTrackAI.

Loads a YOLOv8 face detection model and exposes a simple detect() method
that accepts an OpenCV frame and returns a list of bounding boxes with
confidence scores.

Detection is intentionally separated from tracking and recognition so each
concern can be developed, tested, and replaced independently.
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional

from src.logger import get_logger


class FaceDetector:
    """
    Wraps a YOLO model for face detection.

    Parameters
    ----------
    model_path  : str   Path or name of the YOLO weights file.
                        e.g. 'yolov8n-face.pt'
    confidence  : float Minimum confidence threshold (0–1).
    skip_frames : int   Run detection every N frames; otherwise reuse the
                        previous result.  Reduces CPU load.
    """

    def __init__(
        self,
        model_path: str = "yolov8n-face.pt",
        confidence: float = 0.5,
        skip_frames: int = 5,
    ) -> None:
        self.model_path = model_path
        self.confidence_threshold = confidence
        self.skip_frames = skip_frames
        self.logger = get_logger()

        self._model = None
        self._frame_counter: int = 0
        self._last_detections: List[Dict[str, Any]] = []

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load the YOLO model.

        Tries ultralytics (YOLOv8) first.  Falls back to a lightweight
        OpenCV Haar cascade so the rest of the pipeline can be tested
        without GPU or large model files.
        """
        try:
            from ultralytics import YOLO  # type: ignore
            self._model = YOLO(self.model_path)
            self._backend = "ultralytics"
            self.logger.info(f"FACE_DETECTOR | YOLO model loaded | {self.model_path}")
        except Exception as e:
            self.logger.warning(
                f"FACE_DETECTOR | YOLO load failed ({e}) | Falling back to Haar cascade"
            )
            self._model = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            self._backend = "haar"
            self.logger.info("FACE_DETECTOR | Haar cascade loaded as fallback")

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in a single frame.

        Respects the skip_frames setting: detection runs only on frames
        0, N, 2N, … and the last result is returned for intermediate frames.

        Parameters
        ----------
        frame : np.ndarray  BGR OpenCV image.

        Returns
        -------
        List of detection dicts::

            [
                {"bbox": [x1, y1, x2, y2], "confidence": 0.91},
                …
            ]
        """
        self._frame_counter += 1

        # Run detection only every skip_frames frames
        if (self._frame_counter - 1) % self.skip_frames != 0:
            return self._last_detections

        detections = []

        try:
            if self._backend == "ultralytics":
                detections = self._detect_yolo(frame)
            else:
                detections = self._detect_haar(frame)
        except Exception as e:
            self.logger.error(f"FACE_DETECTOR | Detection error | {e}")
            return self._last_detections

        self._last_detections = detections
        if detections:
            self.logger.debug(f"FACE_DETECTED | {len(detections)} face(s) in frame {self._frame_counter}")
        return detections

    def _detect_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLOv8 inference and extract bounding boxes."""
        results = self._model(frame, verbose=False, conf=self.confidence_threshold)
        detections = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append({"bbox": [x1, y1, x2, y2], "confidence": round(conf, 3)})
        return detections

    def _detect_haar(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run OpenCV Haar cascade detection as a fallback."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._model.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,   # lower = more detections (less strict)
            minSize=(60, 60),
        )
        detections = []
        for (x, y, w, h) in faces:
            detections.append(
                {"bbox": [x, y, x + w, y + h], "confidence": 0.9}  # Haar has no score; use fixed
            )
        return detections

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def crop_face(frame: np.ndarray, bbox: List[int], padding: int = 30) -> Optional[np.ndarray]:
        """
        Extract and return a face crop from the frame.

        Parameters
        ----------
        frame   : np.ndarray  Full BGR frame.
        bbox    : [x1,y1,x2,y2]
        padding : int         Extra pixels around the bounding box.

        Returns
        -------
        np.ndarray or None if the crop is invalid.
        """
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def reset_frame_counter(self) -> None:
        """Reset the internal frame counter (useful for test scenarios)."""
        self._frame_counter = 0
        self._last_detections = []
