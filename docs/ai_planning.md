# FaceTrackAI — AI Planning Document

## 1. Problem Understanding

Build a real-time face tracking system that:
- Detects faces in video/RTSP streams.
- Assigns unique persistent IDs to each person.
- Recognises returning visitors to avoid duplicate counts.
- Logs entry and exit events.
- Maintains an accurate unique visitor count.

## 2. Requirements Summary

| Requirement | Implementation |
|---|---|
| Face detection | YOLOv8 face model |
| Face recognition | InsightFace ArcFace |
| Tracking | ByteTrack (Phase 2) |
| Persistence | SQLite |
| Configuration | config.json |
| Logging | Python logging module |
| Unique counting | COUNT(DISTINCT face_id) |

## 3. Proposed Architecture

See `docs/architecture.md` for the full pipeline diagram.

## 4. Model Selection

### 4.1 Why YOLO?
- State-of-the-art real-time object detection.
- YOLOv8n (nano) is fast enough for CPU execution.
- Purpose-trained face variants (yolov8n-face.pt) exist.
- Simple ultralytics API, easy to replace.

### 4.2 Why InsightFace?
- Production-grade ArcFace embeddings.
- 512-d normalised vectors ideal for cosine similarity.
- Outperforms `face_recognition` on benchmark datasets.
- Does not depend on dlib (easier to install on Windows).
- Well maintained with ONNX support for CPU-only environments.

### 4.3 Why ByteTrack?
- Maintains object identities across frames without requiring re-ID.
- Handles temporary occlusions using Kalman filter prediction.
- Lightweight: does not require a separate appearance model.
- Widely used in surveillance and people-counting systems.

### 4.4 Why SQLite?
- Zero-configuration, file-based database.
- Sufficient for development and demonstration.
- Can be swapped for PostgreSQL via DatabaseManager interface.

## 5. Recognition Strategy

1. At startup, load all stored embeddings into RAM.
2. For each detected face crop:
   a. Generate 512-d ArcFace embedding.
   b. Compute cosine similarity against all registered embeddings.
   c. If max similarity ≥ threshold → returning visitor.
   d. If max similarity < threshold → new visitor; register and assign ID.
3. Threshold is configurable (default 0.5).

## 6. Tracking Strategy

- ByteTrack assigns Track IDs per frame using IoU + motion prediction.
- Track ID is temporary and resets between runs.
- Face ID (visitor_NNN) is permanent and persists in the database.
- Mapping: `track_id → face_id` maintained in memory during a session.

## 7. Entry / Exit Strategy

- ENTRY: logged the first time a face_id is seen in a session.
- EXIT: logged after `max_missed_frames` consecutive frames without detection.
- State machine per active visitor prevents duplicate events.

## 8. Unique Visitor Strategy

- Each unique face_id in the database = one unique visitor.
- `SELECT COUNT(DISTINCT face_id) FROM visitors` gives the accurate count.
- Re-identification ensures the same person is never double-counted.

## 9. Error Handling

- Missing video file → log error, exit cleanly.
- Empty frame → skip, continue loop.
- Embedding failure → skip face, log warning.
- Database error → log error, attempt graceful continue.
- Keyboard interrupt → save state, release resources.

## 10. Performance Optimization

- `detection_skip_frames` reduces YOLO calls (default: every 5 frames).
- Embeddings cached in RAM; DB queried only at startup and registration.
- Lightweight YOLOv8n model chosen for CPU compatibility.
- Embedding generated only when detection fires, not every frame.

## 11. Testing Strategy

- `test_config.py`: config.json validation.
- `test_database.py`: CRUD, round-trip, unique count.
- `test_matching.py`: embedding determinism, matching logic.
- End-to-end test: run on sample video, verify output artifacts.

## 12. Future Improvements

- GPU acceleration with CUDA for both YOLO and InsightFace.
- PostgreSQL backend for multi-node deployments.
- REST API for remote monitoring and control.
- Alert system for unauthorized access detection.
- Dashboard for real-time visitor analytics.
- RTSP multi-camera support.
