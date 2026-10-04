# FaceTrackAI — Architecture

## System Architecture

```
Input Layer          → Video file or RTSP stream (configurable via config.json)
       ↓
Detection Layer      → YOLOv8 face detector (src/detector.py)
                       • Runs every N frames (detection_skip_frames)
                       • Returns [bbox, confidence] per face
       ↓
Tracking Layer       → ByteTrack (src/tracker.py) — Phase 2
                       • Assigns temporary Track IDs
                       • Maintains identity across missed frames
       ↓
Recognition Layer    → InsightFace ArcFace (src/recognizer.py)
                       • Generates 512-d normalised embedding per face crop
                       • Cosine-similarity search against registered embeddings
       ↓
Identity Management  → VisitorManager (src/visitor_manager.py)
                       • New face → visitor_NNN (auto-registered)
                       • Known face → existing visitor_NNN (re-identified)
                       • In-memory embedding cache for fast lookup
       ↓
Event Management     → EventManager (src/event_manager.py) — Phase 2
                       • ENTRY detected on first appearance
                       • EXIT detected after max_missed_frames
                       • State machine prevents duplicate events
       ↓
Persistence Layer    → DatabaseManager (src/database.py)
                       • visitors table: face_id, timestamps, embedding BLOB
                       • events table: face_id, event_type, timestamp, image_path
       ↓
Analytics            → Unique visitor count (SELECT COUNT(DISTINCT face_id))
```

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `detector.py` | YOLO inference, bounding boxes, face crops |
| `recognizer.py` | ArcFace embeddings, cosine similarity, in-memory registry |
| `tracker.py` | ByteTrack temporal track IDs (Phase 2) |
| `visitor_manager.py` | Registration, re-ID, ID assignment |
| `event_manager.py` | Entry/exit state machine (Phase 2) |
| `database.py` | SQLite CRUD, schema management |
| `logger.py` | Centralized structured logging |
| `video_processor.py` | Frame loop, annotation, I/O |
| `pipeline.py` | End-to-end orchestration (Phase 2/3) |
| `app.py` | Entry point, config loading, component wiring |

## Key Design Decisions

1. **Track ID ≠ Face ID** — ByteTrack Track IDs are temporary; Face IDs are persistent.
2. **In-memory embeddings** — All registered embeddings are kept in RAM during processing to avoid DB queries per frame.
3. **Configurable skip frames** — Detection runs every N frames to reduce CPU load.
4. **Fallback backends** — Haar cascade fallback for detector; stub embedding for recognizer when models are unavailable.
5. **SQLite → PostgreSQL** — DatabaseManager interface is designed for easy backend swap.
