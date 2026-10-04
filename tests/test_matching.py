"""
tests/test_matching.py
-----------------------
Unit tests for FaceRecognizer embedding generation and similarity matching.

These tests use the built-in stub backend so they work without InsightFace
installed.  The stub produces deterministic embeddings based on pixel values.

InsightFace is mocked out before import so no model download occurs.
"""

import sys
import os
import unittest
import types
import numpy as np
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# --- Patch insightface before it is imported so model download never fires ---
# Create a minimal mock that satisfies the import in recognizer.py
_mock_insightface = types.ModuleType("insightface")
_mock_app_mod = types.ModuleType("insightface.app")

class _MockFaceAnalysis:
    def __init__(self, *a, **kw): pass
    def prepare(self, *a, **kw): pass
    def get(self, img): return []

_mock_app_mod.FaceAnalysis = _MockFaceAnalysis
_mock_insightface.app = _mock_app_mod
sys.modules.setdefault("insightface", _mock_insightface)
sys.modules.setdefault("insightface.app", _mock_app_mod)
# -----------------------------------------------------------------------------

from src.logger import setup_logger
setup_logger(log_file="logs/events.log")

from src.recognizer import FaceRecognizer


def _make_face_image(seed: int = 0, size: int = 64) -> np.ndarray:
    """Create a deterministic synthetic face image for testing."""
    rng = np.random.default_rng(seed)
    img = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    return img


class TestFaceRecognizer(unittest.TestCase):
    """Tests for embedding generation and similarity-based matching."""

    def setUp(self) -> None:
        # Force stub backend regardless of whether InsightFace is installed
        self.recognizer = FaceRecognizer(similarity_threshold=0.5)
        # Override backend to stub for deterministic tests
        self.recognizer._backend = "stub"
        self.recognizer._registered = {}

    # ------------------------------------------------------------------
    # Embedding tests
    # ------------------------------------------------------------------

    def test_embedding_shape(self) -> None:
        img = _make_face_image(seed=1)
        emb = self.recognizer.generate_embedding(img)
        self.assertIsNotNone(emb)
        self.assertEqual(emb.shape, (512,))

    def test_embedding_is_normalised(self) -> None:
        img = _make_face_image(seed=2)
        emb = self.recognizer.generate_embedding(img)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=4)

    def test_same_image_same_embedding(self) -> None:
        """The stub must be deterministic: same pixels → same embedding."""
        img = _make_face_image(seed=42)
        emb1 = self.recognizer.generate_embedding(img)
        emb2 = self.recognizer.generate_embedding(img)
        np.testing.assert_array_almost_equal(emb1, emb2)

    def test_different_image_different_embedding(self) -> None:
        img_a = _make_face_image(seed=10)
        img_b = _make_face_image(seed=99)
        emb_a = self.recognizer.generate_embedding(img_a)
        emb_b = self.recognizer.generate_embedding(img_b)
        similarity = float(np.dot(emb_a, emb_b))
        # Different images should not be identical
        self.assertFalse(np.allclose(emb_a, emb_b))

    def test_empty_crop_returns_none(self) -> None:
        emb = self.recognizer.generate_embedding(np.array([]))
        self.assertIsNone(emb)

    # ------------------------------------------------------------------
    # Matching tests
    # ------------------------------------------------------------------

    def test_no_match_when_registry_empty(self) -> None:
        img = _make_face_image(seed=5)
        emb = self.recognizer.generate_embedding(img)
        result = self.recognizer.find_match(emb)
        self.assertIsNone(result)

    def test_same_face_returns_same_id(self) -> None:
        """Embedding registered as visitor_001 must match itself."""
        img = _make_face_image(seed=7)
        emb = self.recognizer.generate_embedding(img)
        self.recognizer.register_embedding("visitor_001", emb)
        match = self.recognizer.find_match(emb)
        self.assertEqual(match, "visitor_001")

    def test_different_face_returns_none_or_different_id(self) -> None:
        """Two clearly different faces should not match each other."""
        img_a = _make_face_image(seed=11)
        img_b = _make_face_image(seed=222)
        emb_a = self.recognizer.generate_embedding(img_a)
        emb_b = self.recognizer.generate_embedding(img_b)

        self.recognizer.register_embedding("visitor_001", emb_a)

        # Set a high threshold so the different embedding does not match
        self.recognizer.similarity_threshold = 0.99
        match = self.recognizer.find_match(emb_b)
        self.assertNotEqual(match, "visitor_001")

    def test_closest_match_selected(self) -> None:
        """When multiple visitors are registered, the closest one wins."""
        img_a = _make_face_image(seed=1)
        img_b = _make_face_image(seed=200)
        emb_a = self.recognizer.generate_embedding(img_a)
        emb_b = self.recognizer.generate_embedding(img_b)

        self.recognizer.register_embedding("visitor_001", emb_a)
        self.recognizer.register_embedding("visitor_002", emb_b)
        self.recognizer.similarity_threshold = 0.0  # accept any match

        match_a = self.recognizer.find_match(emb_a)
        match_b = self.recognizer.find_match(emb_b)
        self.assertEqual(match_a, "visitor_001")
        self.assertEqual(match_b, "visitor_002")

    def test_register_and_count(self) -> None:
        for i in range(3):
            emb = np.random.rand(512).astype(np.float32)
            emb /= np.linalg.norm(emb)
            self.recognizer.register_embedding(f"visitor_{i:03d}", emb)
        self.assertEqual(self.recognizer.registered_count(), 3)


if __name__ == "__main__":
    unittest.main()
