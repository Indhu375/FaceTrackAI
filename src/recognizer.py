"""
src/recognizer.py
-----------------
InsightFace-based face recognition module for FaceTrackAI.

Responsibilities
----------------
1. Load an InsightFace ArcFace model.
2. Generate a 512-d embedding from a face crop.
3. Compare a new embedding against all registered embeddings using
   cosine similarity.
4. Return the best-matching Face ID when similarity >= threshold,
   or None when no match is found (new person).

Design note
-----------
The recognizer does NOT touch the database.  It only handles embeddings.
The visitor_manager.py layer is responsible for deciding when to register
or update the database.
"""

import numpy as np
from typing import Dict, List, Optional

import cv2

from src.logger import get_logger


class FaceRecognizer:
    """
    Generates ArcFace embeddings and performs similarity-based face matching.

    Parameters
    ----------
    similarity_threshold : float
        Cosine-similarity cutoff (0–1).  A score >= threshold counts as a
        match.  Typical good values: 0.4–0.6.
    """

    def __init__(self, similarity_threshold: float = 0.5) -> None:
        self.similarity_threshold = similarity_threshold
        self.logger = get_logger()

        # In-memory store: {face_id: embedding_vector}
        # Populated from the database at startup and updated as new faces register.
        self._registered: Dict[str, np.ndarray] = {}

        self._app = None          # InsightFace FaceAnalysis app
        self._backend = "none"

        self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load InsightFace (primary) with graceful fallback for environments
        where the package or model files are not yet available.
        """
        try:
            import insightface  # type: ignore
            from insightface.app import FaceAnalysis  # type: ignore

            self._app = FaceAnalysis(
                name="buffalo_l",          # ArcFace-based model bundle
                providers=["CPUExecutionProvider"],
            )
            # Use a smaller det_size so InsightFace can detect faces in
            # cropped images (Haar crops are typically 50–200 px wide).
            # 160×160 is a good balance between accuracy and speed on CPU.
            self._app.prepare(ctx_id=0, det_size=(160, 160))
            self._rec_model = getattr(self._app, "models", {}).get("recognition", None)
            self._backend = "insightface"
            self.logger.info("FACE_RECOGNIZER | InsightFace (ArcFace) loaded successfully")
        except Exception as e:
            self.logger.warning(
                f"FACE_RECOGNIZER | InsightFace load failed ({e}) | "
                "Using random-embedding stub for pipeline testing"
            )
            self._backend = "stub"
            self._rec_model = None

    # ------------------------------------------------------------------
    # Embedding generation
    # ------------------------------------------------------------------

    def generate_embedding(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Generate a normalized ArcFace embedding from a face crop.

        Parameters
        ----------
        face_crop : np.ndarray  BGR face crop (output of detector.crop_face).

        Returns
        -------
        np.ndarray of shape (512,) – the face embedding vector.
        Returns None if the crop is unusable.
        """
        if face_crop is None or face_crop.size == 0:
            self.logger.warning("FACE_RECOGNIZER | Empty face crop received – skipping")
            return None

        try:
            if self._backend == "insightface":
                return self._embed_insightface(face_crop)
            else:
                return self._embed_stub(face_crop)
        except Exception as e:
            self.logger.error(f"FACE_RECOGNIZER | Embedding error | {e}")
            return None

    def _embed_insightface(self, face_crop: np.ndarray) -> Optional[np.ndarray]:
        """
        Use InsightFace to produce a real ArcFace embedding.

        First attempts detection and alignment with app.get().
        If detection on the crop finds no face, falls back to direct
        ArcFace feature extraction on a 112x112 normalized crop.
        """
        MIN_SIDE = 160  # px — minimum short-edge size for reliable detection

        h, w = face_crop.shape[:2]
        scaled = face_crop
        if h < MIN_SIDE or w < MIN_SIDE:
            scale = MIN_SIDE / min(h, w)
            new_w = max(MIN_SIDE, int(w * scale))
            new_h = max(MIN_SIDE, int(h * scale))
            scaled = cv2.resize(face_crop, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Method 1: FaceAnalysis detection + landmark alignment
        try:
            faces = self._app.get(scaled)
            if faces:
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                if face.normed_embedding is not None:
                    self.logger.debug("EMBEDDING_GENERATED | InsightFace landmark-aligned embedding produced")
                    return face.normed_embedding.astype(np.float32)
        except Exception as e:
            self.logger.debug(f"InsightFace app.get error: {e}")

        # Method 2: Direct ArcFace feature extraction on 112x112 crop
        if self._rec_model is not None:
            try:
                aligned = cv2.resize(face_crop, (112, 112), interpolation=cv2.INTER_LINEAR)
                feats = self._rec_model.get_feat(aligned)
                feat = feats.flatten()
                norm = np.linalg.norm(feat)
                if norm > 0:
                    feat = feat / norm
                    self.logger.debug("EMBEDDING_GENERATED | Direct ArcFace crop embedding produced")
                    return feat.astype(np.float32)
            except Exception as e:
                self.logger.warning(f"Direct ArcFace extraction failed: {e}")

        return None

    def _embed_stub(self, face_crop: np.ndarray) -> np.ndarray:
        """
        Deterministic stub embedding based on image pixel mean.
        Useful for pipeline testing when InsightFace is not installed.
        The embedding is normalised so cosine-similarity comparisons work.
        """
        # Use a small resized version to create a reproducible pseudo-embedding
        resized = cv2.resize(face_crop, (64, 64)).astype(np.float32) / 255.0
        flat = resized.flatten()[:512]
        if len(flat) < 512:
            flat = np.pad(flat, (0, 512 - len(flat)))
        norm = np.linalg.norm(flat)
        if norm > 0:
            flat = flat / norm
        return flat.astype(np.float32)

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------

    def find_match(self, embedding: np.ndarray) -> Optional[str]:
        """
        Compare an embedding against all registered embeddings.

        Parameters
        ----------
        embedding : np.ndarray  Query embedding (512-d, normalised).

        Returns
        -------
        str   – face_id of the best match if similarity >= threshold.
        None  – no match; caller should register as a new visitor.
        """
        if not self._registered:
            return None

        best_face_id: Optional[str] = None
        best_score: float = -1.0

        for face_id, registered_embedding in self._registered.items():
            score = self._cosine_similarity(embedding, registered_embedding)
            if score > best_score:
                best_score = score
                best_face_id = face_id

        if best_score >= self.similarity_threshold:
            self.logger.debug(
                f"FACE_RECOGNIZED | {best_face_id} | similarity={best_score:.3f}"
            )
            return best_face_id

        self.logger.debug(f"FACE_RECOGNIZER | No match | best_score={best_score:.3f}")
        return None

    # ------------------------------------------------------------------
    # Registry management
    # ------------------------------------------------------------------

    def register_embedding(self, face_id: str, embedding: np.ndarray) -> None:
        """
        Add (or update) an embedding in the in-memory registry.

        Called by visitor_manager when a new face is registered or when
        embeddings are loaded from the database at startup.

        Parameters
        ----------
        face_id   : str          Persistent visitor identifier.
        embedding : np.ndarray   512-d ArcFace vector.
        """
        self._registered[face_id] = embedding
        self.logger.debug(f"FACE_RECOGNIZER | Registered in memory | {face_id}")

    def load_from_database(self, visitors: List[Dict]) -> None:
        """
        Bulk-load all stored embeddings into memory at startup.

        Parameters
        ----------
        visitors : list of dicts  [{face_id, embedding, …}, …]
                   Typically from DatabaseManager.get_all_visitors().
        """
        for visitor in visitors:
            self._registered[visitor["face_id"]] = visitor["embedding"]
        self.logger.info(f"FACE_RECOGNIZER | Loaded {len(visitors)} embedding(s) from database")

    def registered_count(self) -> int:
        """Return how many faces are currently registered in memory."""
        return len(self._registered)

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors.

        Both vectors should be L2-normalised (which InsightFace guarantees).
        For normalised vectors, dot product == cosine similarity.

        Returns
        -------
        float in [-1, 1]; higher means more similar.
        """
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
