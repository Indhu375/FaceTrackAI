# FaceTrackAI — Intelligent Multi-Face Tracking & Visitor Analytics

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-44%20Passing-brightgreen.svg)]()

**FaceTrackAI** is a real-time computer vision and visitor intelligence platform that performs multi-face detection, continuous temporal tracking, ArcFace biometric re-identification, entry/exit event management, and business intelligence foot-traffic analytics.

---

## 📹 Video Demonstration & Walkthrough

> **[Click Here to Watch the Video Demonstration & Solution Walkthrough](https://www.youtube.com/watch?v=YOUR_VIDEO_ID_HERE)**  
> *(Alternative Loom Link: [https://www.loom.com/share/YOUR_LOOM_ID_HERE](https://www.loom.com/share/YOUR_LOOM_ID_HERE))*
> 
> *Note: Please replace the placeholder link above with your recorded YouTube or Loom presentation link.*

---

## 🏛️ System Architecture

```
                       Input Stream (Video File / RTSP Camera)
                                         │
                                         ▼
                 [Phase 1] Detection Layer: YOLOv8 Face Detector
                         (yolov8n-face.pt / Haar Cascade fallback)
                                         │
                                         ▼
                 [Phase 2] Tracking Layer: ByteTrack Multi-Object Tracker
                         (Kalman Filter + Hungarian IoU Association)
                                         │
                                         ▼
                 [Phase 1] Recognition Layer: InsightFace ArcFace
                         (512-d L2-normalized embeddings + cosine similarity)
                                         │
                                         ▼
                 [Phase 1] Identity Management: VisitorManager
                         (In-memory caching + persistent SQLite registry)
                                         │
                                         ▼
                 [Phase 2] Event Management: EventManager
                         (ENTRY / EXIT state machine + face snapshot archival)
                                         │
                                         ▼
                 [Phase 3] Analytics & Live Intelligence Suite
                         (Dwell time, hourly traffic, exports & Web Dashboard)
```

### Module Data Flow & Responsibilities

| Layer | Module | Responsibility |
|---|---|---|
| **Detection** | [`src/detector.py`](file:///f:/My%20Project/FaceTrackAI/src/detector.py) | YOLOv8 nano face detection running on configurable frame skips (`detection_skip_frames`). |
| **Tracking** | [`src/tracker.py`](file:///f:/My%20Project/FaceTrackAI/src/tracker.py) | ByteTrack multi-object tracker using Kalman filter state prediction and SciPy Hungarian matching. |
| **Recognition** | [`src/recognizer.py`](file:///f:/My%20Project/FaceTrackAI/src/recognizer.py) | InsightFace ArcFace 512-d normalized embeddings with dual-path alignment and fallback. |
| **Visitor Registry** | [`src/visitor_manager.py`](file:///f:/My%20Project/FaceTrackAI/src/visitor_manager.py) | Assigns persistent Face IDs (`visitor_001`, `visitor_002`), manages in-memory embedding cache. |
| **Event State** | [`src/event_manager.py`](file:///f:/My%20Project/FaceTrackAI/src/event_manager.py) | ENTRY / EXIT state machine, duplicate prevention, and face crop snapshot archival. |
| **Analytics Engine**| [`src/analytics.py`](file:///f:/My%20Project/FaceTrackAI/src/analytics.py) | Dwell time computation, hourly traffic, retention rates, CSV/JSON/HTML report exports. |
| **Web Dashboard** | [`src/dashboard.py`](file:///f:/My%20Project/FaceTrackAI/src/dashboard.py) | Zero-dependency real-time glassmorphic analytics dashboard with 3s live polling. |
| **Persistence** | [`src/database.py`](file:///f:/My%20Project/FaceTrackAI/src/database.py) | SQLite storage with `visitors` and `events` tables. |

---

## 🧠 AI Planning Document

### 1. Problem Understanding
Surveillance and retail environments require tracking individuals across space and time to distinguish between first-time and returning visitors without double-counting, log arrivals and departures, compute dwell times, and provide actionable foot-traffic intelligence.

### 2. Model Selection Rationale
- **YOLOv8 Nano (`yolov8n-face.pt`)**: Purpose-trained for face detection. Nano architecture ensures high FPS execution on standard CPU hardware without mandatory GPU requirements.
- **InsightFace ArcFace (`buffalo_l`)**: Industry standard for deep facial recognition. Generates 512-dimensional unit vectors where cosine similarity directly measures identity equivalence. Robust to pose variations and lighting changes.
- **ByteTrack Tracking**: Operates by associating both high-confidence and low-confidence detection boxes across frames using Kalman motion estimation and IoU matching. Eliminates ID switches during brief occlusions.
- **SciPy Linear Assignment**: Built using pure SciPy (`scipy.optimize.linear_sum_assignment`), avoiding fragile C++ compilation issues (such as `lap` on Windows).

### 3. Identity & Tracking Strategy
- **Track ID $\neq$ Face ID**:
  - `track_id` is an ephemeral integer (e.g. `1`, `2`, `3`) maintaining spatial continuity frame-to-frame.
  - `face_id` is a persistent string (e.g. `visitor_001`) stored in the database.
- **Performance Optimization**: Once a `track_id` is matched with a confirmed `face_id`, computationally intensive ArcFace feature extraction is skipped on subsequent frames for that track, yielding massive CPU efficiency.

### 4. Entry / Exit State Machine
- **ENTRY**: Triggered on first confirmed recognition of a visitor trajectory; persists an entry snapshot image to `logs/entries/` and inserts a database row.
- **ACTIVE**: Visitor remains active; updates `last_seen` timestamp without generating redundant entry events.
- **EXIT**: Triggered when a track has been lost for `max_missed_frames` (default: 30 frames); persists an exit snapshot image to `logs/exits/` and inserts a database row.
- **FLUSH**: On stream termination or graceful shutdown, all currently active visitors are closed with an exit event.

---

## ⚙️ Assumptions Made

1. **Camera Angle & Visibility**:
   - The input video or RTSP stream provides a frontal or semi-profile view of human faces with minimum resolution of at least $40 \times 40$ pixels for reliable ArcFace feature extraction.
2. **Biometric Similarity Cutoff**:
   - Cosine similarity threshold of `0.5` is assumed as the operational balance between False Accepts and False Rejects. Faces with similarity $\ge 0.5$ are identified as the same visitor; below $0.5$ triggers registration of a new visitor.
3. **Session Expiry & Dwell Time**:
   - If a person is undetected for more than `max_missed_frames` (default: 30 frames $\approx$ 1–2 seconds depending on frame rate), they are assumed to have left the camera field of view, closing their visit session. If they return later, a new visit session (ENTRY) is recorded under their existing persistent `face_id`.
4. **Hardware & Environment**:
   - The system is architected for CPU execution using ONNX Runtime and OpenCV, but automatically leverages CUDA if available.
5. **Storage & Persistence**:
   - SQLite provides single-file local persistence with zero configuration. Schema design follows modular repository patterns to allow painless migration to PostgreSQL.

---

## 🛠️ Setup Instructions

### 1. Prerequisites
- **Python 3.10+** (Python 3.10, 3.11, 3.12, or 3.13)
- Windows, macOS, or Linux

### 2. Installation

```bash
# Clone the repository
git clone https://github.com/Indhu375/FaceTrackAI.git
cd FaceTrackAI

# Create and activate a virtual environment (recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

### 3. Model Weights
- **YOLOv8 Face Model**: The pre-trained model `yolov8n-face.pt` is already provided in the repository root.
- **InsightFace Model Pack**: The `buffalo_l` ArcFace bundle auto-downloads on first run to `~/.insightface/models/buffalo_l/`.

---

## 🚀 How to Run

### Mode 1: Main Video Tracking Pipeline
Process the sample video or live RTSP camera stream:

```bash
# Run with default config (sample_video/input.mp4)
python app.py

# Run with a custom video source or RTSP stream
python app.py --source "sample_video/input.mp4"
python app.py --source "rtsp://user:pass@192.168.1.100:554/stream"
```
*Outputs:*
- Live preview window with bounding boxes (`T:<track_id> | <visitor_id>`) and HUD dashboard.
- Annotated output video saved to `outputs/annotated/demo_output.mp4`.
- Saved face crops in `logs/entries/` and `logs/exits/`.
- Automated analytics report generated upon completion into `outputs/reports/`.

---

### Mode 2: Live Analytics Web Dashboard
Launch the standalone web dashboard:

```bash
python dashboard.py
```
Open your browser at: **`http://127.0.0.1:8000`**

*Features:*
- **Live Sync**: Auto-refreshes every 3 seconds to reflect real-time camera tracking.
- **Interactive KPIs**: Total Unique Visitors, Active Right Now, Average Dwell Time, Event Counts.
- **Hourly Traffic Chart**: Zero-dependency visual canvas chart.
- **Visitor Directory & Event Stream**: Searchable tables with hoverable face thumbnails.
- **One-Click Export**: Download Visitors CSV, Events CSV, and Analytics JSON directly.

---

### Mode 3: Generate Analytics Reports (CLI)
Generate executive reports directly from the database without opening a video stream:

```bash
python app.py --report
```
*Generated in `outputs/reports/`:*
- `visitors_summary_<timestamp>.csv`
- `events_log_<timestamp>.csv`
- `analytics_report_<timestamp>.json`
- `visitor_report_<timestamp>.html` (Interactive standalone visual report)

---

## 🧪 Running Unit Tests

Run the full automated test suite (44 unit tests):

```bash
python -m pytest tests/ -v
```

---

## 📋 Sample `config.json` Structure

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

### Parameter Explanations:
- `video_source`: Path to local MP4/AVI video file or RTSP stream URL.
- `detection.model`: Path to YOLO weights (`yolov8n-face.pt`).
- `detection.confidence_threshold`: Minimum confidence score (0.0 – 1.0) to register a face box.
- `detection.detection_skip_frames`: Detection runs every $N$ frames to save CPU cycles.
- `recognition.similarity_threshold`: Cosine similarity cutoff (0.0 – 1.0) for matching ArcFace embeddings.
- `tracking.max_missed_frames`: Consecutive unobserved frames before a track is closed and an EXIT event is fired.
- `database.path`: Location of SQLite database file.
- `storage.entry_directory` / `exit_directory`: Target folders for saved face crop images.
- `display.show_window`: Set `true` to show live OpenCV preview window (`q` to quit).
- `display.save_output_video`: Set `true` to record annotated output video to `outputs/annotated/demo_output.mp4`.

---

This project is a part of a hackathon run by https://katomaran.com
