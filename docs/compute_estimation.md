# FaceTrackAI — Compute Estimation

> **Note:** Values below marked *(estimate)* are qualitative estimates based on
> published benchmarks and typical deployment experience.
> Values marked *(measured)* were recorded during actual test runs on this machine.

## Per-Component Resource Usage

| Component | CPU Usage | GPU Usage | RAM | Notes |
|---|---|---|---|---|
| YOLOv8n face | Medium *(estimate)* | Recommended | ~6 MB model | ~15–30 FPS CPU-only |
| InsightFace (buffalo_l) | Medium *(estimate)* | Recommended | ~300 MB model | ArcFace 512-d embedding |
| ByteTrack | Low *(estimate)* | Not required | < 50 MB | Kalman filter, IoU |
| SQLite | Very Low *(estimate)* | N/A | < 5 MB | File-based, single connection |
| OpenCV (decode+display) | Low–Medium *(estimate)* | Optional | Frame buffer | Depends on resolution |

## System-Level Estimates

| Scenario | Expected FPS | CPU % | RAM |
|---|---|---|---|
| 1080p video, CPU only | 5–15 FPS *(estimate)* | 60–90% *(estimate)* | ~800 MB |
| 720p video, CPU only | 15–25 FPS *(estimate)* | 40–70% *(estimate)* | ~600 MB |
| 720p video, GPU (CUDA) | 30+ FPS *(estimate)* | 10–20% *(estimate)* | ~500 MB + VRAM |

## Frame Skip Impact

With `detection_skip_frames = 5`, YOLO runs on 1 in 5 frames:
- Effective YOLO load reduced by ~80%.
- Bounding box interpolation used for intermediate frames.
- Significant CPU savings with minimal accuracy loss at typical video frame rates (25–30 FPS).

## Model Sizes (Approximate)

| Model | Size |
|---|---|
| yolov8n-face.pt | ~6 MB |
| InsightFace buffalo_l | ~300 MB (auto-downloaded) |

## Assumptions

- Development environment: Windows 10/11, Python 3.10+, no dedicated GPU.
- Video resolution: 1280×720 or lower recommended for CPU-only.
- `detection_skip_frames = 5` is the default; increase for faster CPU.
- InsightFace model downloaded automatically on first run.
- Benchmark numbers will be updated after actual profiling runs.
