"""
app.py
------
Entry point for FaceTrackAI — Phase 2.

Usage
-----
    python app.py
    python app.py --source sample_video/input.mp4
    python app.py --source "rtsp://user:pass@192.168.1.1/stream"

The application:
1. Loads config.json.
2. Sets up the logger.
3. Initialises the SQLite database.
4. Loads YOLO (face detection).
5. Loads ByteTracker (face tracking + temporal continuity).
6. Loads InsightFace (face recognition / ArcFace embedding).
7. Initialises VisitorManager (persistent identity resolution).
8. Initialises EventManager (ENTRY/EXIT state machine & snapshot archival).
9. Wires all modules into an end-to-end Pipeline.
10. Processes video frames, renders preview HUD, writes output video.
11. Records ENTRY/EXIT events into SQLite and saves face images.
12. Releases all resources and flushes exit sessions on shutdown.
"""

import argparse
import json
import sys
import os

# Ensure the project root is on sys.path so 'src' can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.logger import setup_logger, get_logger
from src.database import DatabaseManager
from src.detector import FaceDetector
from src.tracker import ByteTracker
from src.recognizer import FaceRecognizer
from src.visitor_manager import VisitorManager
from src.event_manager import EventManager
from src.pipeline import Pipeline
from src.video_processor import VideoProcessor


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.json") -> dict:
    """
    Load and return config.json as a dictionary.

    Parameters
    ----------
    config_path : str  Path to the JSON configuration file.

    Returns
    -------
    dict  Parsed configuration.
    """
    if not os.path.exists(config_path):
        print(f"[ERROR] Configuration file not found: {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError as e:
        print(f"[ERROR] Invalid JSON in {config_path}: {e}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse optional command-line arguments."""
    parser = argparse.ArgumentParser(
        description="FaceTrackAI — Intelligent Face Tracking & Unique Visitor Analytics"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override video_source in config.json (path or RTSP URL)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to the configuration file (default: config.json)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Load configuration
    config = load_config(args.config)

    # Override video source from CLI if provided
    if args.source:
        config["video_source"] = args.source

    # 2. Set up logging
    log_file = config.get("logging", {}).get("log_file", "logs/events.log")
    logger = setup_logger(log_file)
    logger.info("=" * 60)
    logger.info("FaceTrackAI | Starting — Phase 2 (Tracking + Events + Analytics)")
    logger.info(f"FaceTrackAI | Config: {args.config}")
    logger.info(f"FaceTrackAI | Source: {config['video_source']}")
    logger.info("=" * 60)

    # 3. Initialise database
    db_path = config.get("database", {}).get("path", "data/visitors.db")
    logger.info(f"DATABASE | Initialising | {db_path}")
    db = DatabaseManager(db_path)

    # 4. Load YOLO face detector
    det_cfg = config.get("detection", {})
    detector = FaceDetector(
        model_path=det_cfg.get("model", "yolov8n-face.pt"),
        confidence=det_cfg.get("confidence_threshold", 0.5),
        skip_frames=det_cfg.get("detection_skip_frames", 5),
    )

    # 5. Load ByteTracker
    track_cfg = config.get("tracking", {})
    tracker = ByteTracker(
        max_missed_frames=track_cfg.get("max_missed_frames", 30),
        track_thresh=det_cfg.get("confidence_threshold", 0.5),
    )

    # 6. Load InsightFace recognizer
    rec_cfg = config.get("recognition", {})
    recognizer = FaceRecognizer(
        similarity_threshold=rec_cfg.get("similarity_threshold", 0.5)
    )

    # 7. Initialise visitor manager (bootstraps in-memory embeddings from DB)
    visitor_manager = VisitorManager(db=db, recognizer=recognizer)

    # 8. Initialise event manager (handles ENTRY/EXIT state & saves crops)
    storage_cfg = config.get("storage", {})
    event_manager = EventManager(
        db=db,
        entry_dir=storage_cfg.get("entry_directory", "logs/entries"),
        exit_dir=storage_cfg.get("exit_directory", "logs/exits"),
    )

    # 9. Assemble end-to-end Pipeline
    pipeline = Pipeline(
        config=config,
        detector=detector,
        tracker=tracker,
        recognizer=recognizer,
        visitor_manager=visitor_manager,
        event_manager=event_manager,
    )

    # 10. Initialise and run video processor
    processor = VideoProcessor(
        config=config,
        pipeline=pipeline,
    )

    try:
        processor.run()
    finally:
        # 11. Graceful shutdown
        unique_count = visitor_manager.unique_visitor_count()
        logger.info(f"FaceTrackAI | Finished | Unique Visitors = {unique_count}")
        db.close()
        logger.info("FaceTrackAI | Shutdown complete")


if __name__ == "__main__":
    main()
