"""
src/visitor_manager.py
----------------------
Identity management layer for FaceTrackAI — Phase 1.

This module bridges detection (detector.py), recognition (recognizer.py),
and persistence (database.py).  It is the single source of truth for:

* Deciding whether a face is new or returning.
* Assigning persistent Face IDs.
* Keeping the in-memory registry in sync with the database.

Phase 1 scope
-------------
Only registration and recognition are implemented here.
Entry/exit state management is added in Phase 2 (event_manager.py).
"""

import numpy as np
from datetime import datetime
from typing import Optional

from src.database import DatabaseManager
from src.recognizer import FaceRecognizer
from src.logger import get_logger


class VisitorManager:
    """
    Handles all face identity decisions for FaceTrackAI.

    Parameters
    ----------
    db      : DatabaseManager   Persistence layer.
    recognizer : FaceRecognizer  Embedding generation + similarity search.
    """

    def __init__(self, db: DatabaseManager, recognizer: FaceRecognizer) -> None:
        self.db = db
        self.recognizer = recognizer
        self.logger = get_logger()

        # Load existing visitors into the recognizer's in-memory registry
        self._bootstrap()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _bootstrap(self) -> None:
        """
        Load all stored embeddings from the database into memory so that
        recognition works without querying the DB on every frame.
        """
        visitors = self.db.get_all_visitors()
        self.recognizer.load_from_database(visitors)
        self.logger.info(
            f"VISITOR_MANAGER | Bootstrapped with {len(visitors)} known visitor(s)"
        )

    # ------------------------------------------------------------------
    # Core identity resolution
    # ------------------------------------------------------------------

    def identify(self, face_crop: np.ndarray) -> Optional[str]:
        """
        Identify a face from its crop image.

        This is the main entry point called per detected face.  It:
        1. Generates an ArcFace embedding.
        2. Searches the in-memory registry for a match.
        3. If a match is found → returns existing face_id.
        4. If no match → registers a new visitor and returns the new face_id.

        Parameters
        ----------
        face_crop : np.ndarray  BGR face crop.

        Returns
        -------
        str   – the resolved face_id (e.g. 'visitor_001').
        None  – if embedding generation fails.
        """
        # Step 1: generate embedding
        embedding = self.recognizer.generate_embedding(face_crop)
        if embedding is None:
            h, w = face_crop.shape[:2] if face_crop is not None and face_crop.size > 0 else (0, 0)
            self.logger.warning(
                f"VISITOR_MANAGER | Could not generate embedding – skipping frame "
                f"(crop size: {w}x{h})"
            )
            return None

        self.logger.debug("EMBEDDING_GENERATED | embedding produced for face crop")

        # Step 2: try to match against registered faces
        face_id = self.recognizer.find_match(embedding)

        if face_id is not None:
            # Returning visitor
            self.logger.info(f"FACE_RECOGNIZED | {face_id}")
            self.db.update_visitor_last_seen(face_id)
            return face_id

        # Step 3: new visitor – register
        face_id = self._register_new_visitor(embedding)
        return face_id

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def _register_new_visitor(self, embedding: np.ndarray) -> str:
        """
        Create a new visitor record in the database and in-memory registry.

        Parameters
        ----------
        embedding : np.ndarray  ArcFace embedding for the new visitor.

        Returns
        -------
        str  – newly assigned face_id.
        """
        face_id = self.db.get_next_face_id()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Persist to database
        self.db.insert_visitor(face_id, embedding, first_seen=now)

        # Keep in-memory registry up-to-date so future frames match immediately
        self.recognizer.register_embedding(face_id, embedding)

        self.logger.info(f"FACE_REGISTERED | {face_id} | first_seen={now}")
        return face_id

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def unique_visitor_count(self) -> int:
        """Return the number of unique visitors from the database."""
        return self.db.get_visitor_count()

    def is_known(self, face_id: str) -> bool:
        """Return True if face_id already exists in the database."""
        return self.db.face_id_exists(face_id)
