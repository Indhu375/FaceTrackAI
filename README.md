# FaceTrackAI — Intelligent Multi-Face Tracking & Visitor Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-44%20Passing-brightgreen.svg)]()

**FaceTrackAI** is a high-performance computer vision system for real-time face detection, multi-person temporal tracking, ArcFace biometric re-identification, entry/exit event management, and business intelligence foot-traffic analytics.

---

## 🏛️ System Architecture

```
                       Input Stream (Video File / RTSP Camera)
                                         |
                                         v
                 [Phase 1] Detection Layer: YOLOv8 Face Detector
                         (yolov8n-face.pt / Haar Cascade fallback)
                                         |
                                         v
                 [Phase 2] Tracking Layer: ByteTrack Multi-Object Tracker
                         (Kalman Filter + Hungarian IoU Association)
                                         |
                                         v
                 [Phase 1] Recognition Layer: InsightFace ArcFace
                         (512-d L2-normalized embeddings + cosine similarity)
                                         |
                                         v
                 [Phase 1] Identity Management: VisitorManager
                         (In-memory caching + persistent SQLite registry)
                                         |
                                         v
                 [Phase 2] Event Management: EventManager
                         (ENTRY / EXIT state machine + face snapshot archival)
                                         |
                                         v
                 [Phase 3] Analytics & Live Intelligence Suite
                         (Dwell time, hourly traffic, exports & Web Dashboard)
```

---

## 🚀 Key Features by Phase

### Phase 1: Detection & Recognition
- **YOLOv8 Nano Face Model**: High-accuracy, real-time bounding box detection optimized for CPU execution.
- **InsightFace (ArcFace)**: 512-dimensional deep facial embeddings with dual-path alignment and feature extraction.
- **Persistent Biometric Registry**: In-memory embedding cache synced with an embedded SQLite database.
- **Configurable Thresholds**: Tunable detection confidence and cosine similarity cutoffs.

### Phase 2: Multi-Object Tracking & Events
- **ByteTrack Tracking**: Kalman filter motion prediction + SciPy Hungarian assignment (`linear_sum_assignment`).
- **Temporary Track IDs vs. Persistent Face IDs**: Maintains continuity across occlusions and skips re-embedding once recognized.
- **Automated Entry / Exit State Machine**:
  - `ENTRY`: Captured on initial appearance; image saved to `logs/entries/`.
  - `EXIT`: Triggered after `max_missed_frames` (default: 30 frames); image saved to `logs/exits/`.
  - Duplicate event prevention and clean shutdown flush.

### Phase 3: Analytics, Reporting & Live Dashboard
- **Foot-Traffic Analytics Engine** (`src/analytics.py`):
  - Total unique visitor count
  - Dwell time / duration calculation (average, min, max per visitor)
  - Hourly traffic distribution and peak occupancy
  - Visitor retention & frequency analysis (single vs. returning visitors)
- **Multi-Format Export**:
  - `outputs/reports/visitors_summary_<timestamp>.csv`
  - `outputs/reports/events_log_<timestamp>.csv`
  - `outputs/reports/analytics_report_<timestamp>.json`
  - `outputs/reports/visitor_report_<timestamp>.html` (Self-contained executive report)
- **Live Web Dashboard** (`dashboard.py`):
  - Real-time glassmorphic UI with auto-refresh polling (3s)
  - Live KPI cards, interactive SVG/Canvas traffic charts, and searchable visitor directory
  - Event stream with hoverable face thumbnail previews and one-click report exports.

---

## 📁 Repository Structure

```
FaceTrackAI/
├── app.py                     # Main application entry point (CLI & Video Pipeline)
├── dashboard.py               # Stand-alone Web Dashboard launcher
├── config.json                # Central system configuration
├── requirements.txt           # Python dependencies
├── yolov8n-face.pt            # Downloaded YOLO face detection weights
├── data/
│   └── visitors.db            # SQLite database (visitors & events)
├── docs/
│   ├── ai_planning.md         # Architecture and design documentation
│   ├── architecture.md        # Pipeline dataflow specifications
│   └── compute_estimation.md  # CPU/GPU compute sizing
├── logs/
│   ├── events.log             # Structured application log
│   ├── entries/               # Saved ENTRY face snapshots
│   └── exits/                 # Saved EXIT face snapshots
├── outputs/
│   ├── annotated/             # Rendered preview videos with HUD overlay
│   └── reports/               # Exported CSV, JSON, and HTML reports
├── src/
│   ├── analytics.py           # Dwell time and reporting engine (Phase 3)
│   ├── dashboard.py           # Zero-dependency HTTP server & web app (Phase 3)
│   ├── database.py            # SQLite schema and data access layer (Phase 1)
│   ├── detector.py            # YOLOv8 face detector wrapper (Phase 1)
│   ├── event_manager.py       # ENTRY/EXIT state machine and snapshot saver (Phase 2)
│   ├── logger.py              # Structured logging utility
│   ├── pipeline.py            # End-to-end processing pipeline orchestrator (Phase 2)
│   ├── recognizer.py          # InsightFace ArcFace embedding generator (Phase 1)
│   ├── tracker.py             # ByteTrack + KalmanBoxTracker implementation (Phase 2)
│   ├── video_processor.py     # Frame I/O, display, and video writer loop
│   └── visitor_manager.py     # Identity decision layer and DB sync (Phase 1)
└── tests/
    ├── test_analytics.py      # Analytics & reporting unit tests (5 tests)
    ├── test_config.py         # Config validation unit tests (8 tests)
    ├── test_database.py       # SQLite CRUD & events unit tests (9 tests)
    ├── test_events.py         # EventManager state machine unit tests (4 tests)
    ├── test_matching.py       # ArcFace embedding & cosine similarity tests (10 tests)
    ├── test_pipeline.py       # Pipeline integration unit tests (1 test)
    └── test_tracker.py        # ByteTrack & Kalman filter unit tests (7 tests)
```

---

## ⚡ Quick Start Guide

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/Indhu375/FaceTrackAI.git
cd FaceTrackAI

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Main Video Tracking Application

```bash
# Process default video (sample_video/input.mp4)
python app.py

# Process custom video or RTSP camera stream
python app.py --source path/to/video.mp4
python app.py --source "rtsp://user:pass@192.168.1.100:554/stream"
```

### 3. Launch the Live Analytics Web Dashboard

```bash
# Launch on default port 8000
python dashboard.py

# Open your browser at:
# http://127.0.0.1:8000
```

### 4. Generate Analytics Reports (CLI)

```bash
# Compute metrics and generate CSV/JSON/HTML reports into outputs/reports/
python app.py --report
```

---

## 🧪 Testing

The test suite includes **44 automated unit tests** covering configuration, database operations, recognition matching, tracking, event handling, and analytics:

```bash
python -m pytest tests/ -v
```

---

## ⚙️ Configuration (`config.json`)

```json
{
    "video_source": "sample_video/input.mp4",
    "detection": {
        "model": "yolov8n-face.pt",
        "confidence_threshold": 0.5,
        "detection_skip_frames": 5
    },
    "recognition": {
        "similarity_threshold": 0.5
    },
    "tracking": {
        "max_missed_frames": 30
    },
    "database": {
        "path": "data/visitors.db"
    },
    "logging": {
        "log_file": "logs/events.log"
    },
    "storage": {
        "entry_directory": "logs/entries",
        "exit_directory": "logs/exits"
    },
    "display": {
        "show_window": true,
        "save_output_video": true
    }
}
```

---

## 📜 License

MIT License. Developed for intelligent surveillance and automated visitor analytics.
