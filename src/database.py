"""
src/database.py
---------------
SQLite data access layer for FaceTrackAI.

Provides a clean DatabaseManager class that isolates all SQL from the
rest of the application.  The design intentionally keeps the interface
generic so the backend can be swapped for PostgreSQL later with minimal
changes.

Tables
------
visitors  – one row per unique person, stores embedding as BLOB.
events    – one row per ENTRY or EXIT event.
"""

import sqlite3
import pickle
import os
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any

import numpy as np

from src.logger import get_logger


class DatabaseManager:
    """
    Manages all database interactions for FaceTrackAI.

    Parameters
    ----------
    db_path : str
        Path to the SQLite file (e.g. 'data/visitors.db').
    """

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self.logger = get_logger()

        # Ensure the parent directory exists
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else "data", exist_ok=True)

        self._connection: Optional[sqlite3.Connection] = None
        self._initialise_schema()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_connection(self) -> sqlite3.Connection:
        """Return (or lazily create) a persistent SQLite connection."""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
            self._connection.row_factory = sqlite3.Row  # dict-like row access
        return self._connection

    def _initialise_schema(self) -> None:
        """Create tables if they do not already exist."""
        ddl_visitors = """
        CREATE TABLE IF NOT EXISTS visitors (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id    TEXT    UNIQUE NOT NULL,
            first_seen TEXT    NOT NULL,
            last_seen  TEXT    NOT NULL,
            embedding  BLOB    NOT NULL
        );
        """

        ddl_events = """
        CREATE TABLE IF NOT EXISTS events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            face_id    TEXT    NOT NULL,
            event_type TEXT    NOT NULL,
            timestamp  TEXT    NOT NULL,
            image_path TEXT,
            FOREIGN KEY(face_id) REFERENCES visitors(face_id)
        );
        """

        conn = self._get_connection()
        with conn:
            conn.execute(ddl_visitors)
            conn.execute(ddl_events)
        self.logger.debug("DATABASE_SCHEMA | Tables verified/created")

    # ------------------------------------------------------------------
    # Visitor CRUD
    # ------------------------------------------------------------------

    def insert_visitor(
        self,
        face_id: str,
        embedding: np.ndarray,
        first_seen: Optional[str] = None,
    ) -> None:
        """
        Insert a brand-new visitor record.

        Parameters
        ----------
        face_id   : str        Unique identifier, e.g. 'visitor_001'.
        embedding : np.ndarray ArcFace embedding vector.
        first_seen: str        ISO timestamp string (defaults to now).
        """
        now = first_seen or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        embedding_blob = pickle.dumps(embedding)

        conn = self._get_connection()
        try:
            with conn:
                conn.execute(
                    "INSERT INTO visitors (face_id, first_seen, last_seen, embedding) VALUES (?, ?, ?, ?)",
                    (face_id, now, now, embedding_blob),
                )
            self.logger.info(f"DATABASE_INSERT | Visitor registered | {face_id}")
        except sqlite3.IntegrityError:
            self.logger.warning(f"DATABASE | Duplicate face_id skipped | {face_id}")

    def update_visitor_last_seen(self, face_id: str, timestamp: Optional[str] = None) -> None:
        """Update the last_seen timestamp for a returning visitor."""
        now = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        with conn:
            conn.execute(
                "UPDATE visitors SET last_seen = ? WHERE face_id = ?",
                (now, face_id),
            )

    def get_all_visitors(self) -> List[Dict[str, Any]]:
        """
        Retrieve every visitor with their stored embedding.

        Returns
        -------
        List of dicts: {face_id, first_seen, last_seen, embedding (np.ndarray)}
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT face_id, first_seen, last_seen, embedding FROM visitors")
        rows = cursor.fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "face_id": row["face_id"],
                    "first_seen": row["first_seen"],
                    "last_seen": row["last_seen"],
                    "embedding": pickle.loads(row["embedding"]),
                }
            )
        return result

    def get_visitor_count(self) -> int:
        """Return the count of distinct visitors (unique visitor count)."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(DISTINCT face_id) FROM visitors")
        return cursor.fetchone()[0]

    def face_id_exists(self, face_id: str) -> bool:
        """Check whether a face_id is already in the database."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT 1 FROM visitors WHERE face_id = ? LIMIT 1", (face_id,))
        return cursor.fetchone() is not None

    def get_next_face_id(self) -> str:
        """
        Generate the next sequential Face ID in the format 'visitor_NNN'.

        Returns
        -------
        str  e.g. 'visitor_001', 'visitor_002', …
        """
        conn = self._get_connection()
        cursor = conn.execute("SELECT COUNT(*) FROM visitors")
        count = cursor.fetchone()[0]
        return f"visitor_{count + 1:03d}"

    # ------------------------------------------------------------------
    # Event CRUD
    # ------------------------------------------------------------------

    def insert_event(
        self,
        face_id: str,
        event_type: str,
        timestamp: Optional[str] = None,
        image_path: Optional[str] = None,
    ) -> None:
        """
        Insert an ENTRY or EXIT event record.

        Parameters
        ----------
        face_id    : str  Visitor identifier.
        event_type : str  'ENTRY' or 'EXIT'.
        timestamp  : str  ISO timestamp (defaults to now).
        image_path : str  Path to the saved face crop (optional).
        """
        now = timestamp or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn = self._get_connection()
        with conn:
            conn.execute(
                "INSERT INTO events (face_id, event_type, timestamp, image_path) VALUES (?, ?, ?, ?)",
                (face_id, event_type, now, image_path),
            )
        self.logger.info(f"DATABASE_INSERT | Event | {event_type} | {face_id} | {now}")

    def get_events_for_visitor(self, face_id: str) -> List[Dict[str, Any]]:
        """Return all events for a given visitor."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT face_id, event_type, timestamp, image_path FROM events WHERE face_id = ? ORDER BY timestamp",
            (face_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_all_events(self) -> List[Dict[str, Any]]:
        """Return all events ordered by timestamp."""
        conn = self._get_connection()
        cursor = conn.execute(
            "SELECT face_id, event_type, timestamp, image_path FROM events ORDER BY timestamp"
        )
        return [dict(row) for row in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the SQLite connection cleanly."""
        if self._connection:
            self._connection.close()
            self._connection = None
            self.logger.debug("DATABASE | Connection closed")
