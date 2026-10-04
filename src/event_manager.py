"""
src/event_manager.py
--------------------
Entry/exit event management for FaceTrackAI — Phase 2.

Responsibilities
----------------
1. Track active visitor presence sessions (unobserved -> ACTIVE -> EXITED).
2. Detect and record ENTRY events on initial track identification.
3. Detect and record EXIT events when a track expires after max_missed_frames.
4. Save cropped face image snapshots to disk (logs/entries/ and logs/exits/).
5. Write event records into the SQLite database ('events' table).
6. Prevent duplicate ENTRY or EXIT events for active visitor sessions.
"""

import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import cv2
import numpy as np

from src.database import DatabaseManager
from src.tracker import STrack
from src.logger import get_logger


class EventManager:
    """
    State machine for visitor ENTRY and EXIT events.

    Parameters
    ----------
    db        : DatabaseManager Persistence backend.
    entry_dir : str             Directory to save entry face images.
    exit_dir  : str             Directory to save exit face images.
    """

    def __init__(
        self,
        db: DatabaseManager,
        entry_dir: str = "logs/entries",
        exit_dir: str = "logs/exits",
    ) -> None:
        self.db = db
        self.entry_dir = entry_dir
        self.exit_dir = exit_dir
        self.logger = get_logger()

        # Ensure snapshot storage directories exist
        os.makedirs(self.entry_dir, exist_ok=True)
        os.makedirs(self.exit_dir, exist_ok=True)

        # Active visitor sessions: {face_id: session_dict}
        # session_dict: {
        #     "face_id": str,
        #     "track_ids": set of active track_ids,
        #     "entry_time": str,
        #     "last_seen": str,
        #     "latest_crop": Optional[np.ndarray],
        #     "entry_image": Optional[str],
        # }
        self._active_sessions: Dict[str, Dict[str, Any]] = {}

    # ------------------------------------------------------------------
    # State inspection
    # ------------------------------------------------------------------

    def get_active_visitors(self) -> List[str]:
        """Return list of face_ids currently present in the scene."""
        return list(self._active_sessions.keys())

    def get_active_count(self) -> int:
        """Return count of currently active visitors."""
        return len(self._active_sessions)

    def is_active(self, face_id: str) -> bool:
        """Check whether a given face_id is currently inside the scene."""
        return face_id in self._active_sessions

    # ------------------------------------------------------------------
    # Event triggers
    # ------------------------------------------------------------------

    def on_track_identified(
        self,
        track: STrack,
        face_crop: Optional[np.ndarray] = None,
        timestamp: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Invoked when a track has an assigned Face ID.

        If the visitor is not currently active, an ENTRY event is triggered,
        the face snapshot is saved to logs/entries/, and the DB is updated.
        """
        face_id = track.face_id
        if not face_id:
            return None

        now = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Cache crops on the track object
        if face_crop is not None and face_crop.size > 0:
            if track.first_crop is None:
                track.first_crop = face_crop.copy()
            track.last_crop = face_crop.copy()

        # Case 1: Brand new session -> Trigger ENTRY
        if face_id not in self._active_sessions:
            img_path = self._save_snapshot(
                crop=face_crop,
                face_id=face_id,
                directory=self.entry_dir,
                timestamp=now,
                tag="entry",
            )

            self.db.insert_event(
                face_id=face_id,
                event_type="ENTRY",
                timestamp=now,
                image_path=img_path,
            )

            self._active_sessions[face_id] = {
                "face_id": face_id,
                "track_ids": {track.track_id},
                "entry_time": now,
                "last_seen": now,
                "latest_crop": face_crop.copy() if face_crop is not None else None,
                "entry_image": img_path,
            }

            self.logger.info(
                f"EVENT | ENTRY | {face_id} | track_id={track.track_id} | time={now} | img={img_path}"
            )
            return {
                "event_type": "ENTRY",
                "face_id": face_id,
                "track_id": track.track_id,
                "timestamp": now,
                "image_path": img_path,
            }

        # Case 2: Already active session -> Update last seen
        session = self._active_sessions[face_id]
        session["track_ids"].add(track.track_id)
        session["last_seen"] = now
        if face_crop is not None and face_crop.size > 0:
            session["latest_crop"] = face_crop.copy()

        return None

    def on_track_lost(
        self,
        track: STrack,
        timestamp: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Invoked when a track expires after exceeding max_missed_frames.

        If no other active tracks belong to this visitor, an EXIT event is
        triggered, the exit face snapshot is saved to logs/exits/, and the DB
        is updated.
        """
        face_id = track.face_id
        if not face_id or face_id not in self._active_sessions:
            return None

        session = self._active_sessions[face_id]
        session["track_ids"].discard(track.track_id)

        # If other tracks are still active for this visitor, they haven't exited
        if len(session["track_ids"]) > 0:
            return None

        now = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        exit_crop = track.last_crop if track.last_crop is not None else session.get("latest_crop")
        img_path = self._save_snapshot(
            crop=exit_crop,
            face_id=face_id,
            directory=self.exit_dir,
            timestamp=now,
            tag="exit",
        )

        self.db.insert_event(
            face_id=face_id,
            event_type="EXIT",
            timestamp=now,
            image_path=img_path,
        )

        del self._active_sessions[face_id]

        self.logger.info(
            f"EVENT | EXIT | {face_id} | track_id={track.track_id} | time={now} | img={img_path}"
        )
        return {
            "event_type": "EXIT",
            "face_id": face_id,
            "track_id": track.track_id,
            "timestamp": now,
            "image_path": img_path,
        }

    def flush_all_exits(self, timestamp: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Close all active visitor sessions at the end of the video stream or shutdown.
        Ensures every entry has a matching exit record in the database.
        """
        exits = []
        now = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for face_id in list(self._active_sessions.keys()):
            session = self._active_sessions[face_id]
            img_path = self._save_snapshot(
                crop=session.get("latest_crop"),
                face_id=face_id,
                directory=self.exit_dir,
                timestamp=now,
                tag="exit_flush",
            )

            self.db.insert_event(
                face_id=face_id,
                event_type="EXIT",
                timestamp=now,
                image_path=img_path,
            )

            exits.append(
                {
                    "event_type": "EXIT",
                    "face_id": face_id,
                    "timestamp": now,
                    "image_path": img_path,
                }
            )
            self.logger.info(
                f"EVENT | EXIT (flush) | {face_id} | time={now} | img={img_path}"
            )

        self._active_sessions.clear()
        return exits

    # ------------------------------------------------------------------
    # Image storage helper
    # ------------------------------------------------------------------

    def _save_snapshot(
        self,
        crop: Optional[np.ndarray],
        face_id: str,
        directory: str,
        timestamp: str,
        tag: str,
    ) -> Optional[str]:
        """Save a face crop to disk and return the relative filepath."""
        if crop is None or crop.size == 0:
            return None

        # Sanitize timestamp for Windows/cross-platform path
        clean_ts = timestamp.replace(":", "-").replace(" ", "_")
        filename = f"{face_id}_{tag}_{clean_ts}.jpg"
        filepath = os.path.join(directory, filename)

        try:
            cv2.imwrite(filepath, crop)
            return filepath
        except Exception as e:
            self.logger.warning(f"EVENT_MANAGER | Failed to save snapshot {filepath}: {e}")
            return None
