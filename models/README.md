# Place YOLO and InsightFace Model Files Here

This directory holds downloaded model weights.

## YOLO Face Detection

Download `yolov8n-face.pt`:

```bash
pip install ultralytics
# The model auto-downloads on first run, or manually:
# https://github.com/ultralytics/assets/releases/
```

## InsightFace (ArcFace)

InsightFace downloads the `buffalo_l` model pack automatically on first run.

Default location: `~/.insightface/models/buffalo_l/`

## Notes

- Model files are excluded from Git (see `.gitignore`).
- Place any custom `.pt` or `.onnx` files here if needed.
