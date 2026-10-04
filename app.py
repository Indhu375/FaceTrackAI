"""
app.py
------
Entry point for FaceTrackAI — Phase 3 (Tracking, Analytics & Live Intelligence).

Usage
-----
    python app.py
    python app.py --source sample_video/input.mp4
    python app.py --report                 # Generate analytics reports and exit
    python app.py --dashboard              # Launch live web analytics dashboard
    python app.py --dashboard --port 8080
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
from src.analytics import AnalyticsEngine
from src.dashboard import DashboardServer


# ---------------------------------------------------------------------------
# Configuration loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.json") -> dict:
    """Load and return config.json as a dictionary."""
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
        description="FaceTrackAI — Intelligent Face Tracking, Recognition & Analytics"
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
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate analytics reports (CSV/JSON/HTML) and exit",
    )
    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Launch the live web analytics dashboard",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the web dashboard (default: 8000)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1. Load configuration
    config = load_config(args.config)
    if args.source:
        config["video_source"] = args.source

    # 2. Set up logging
    log_file = config.get("logging", {}).get("log_file", "logs/events.log")
    logger = setup_logger(log_file)

    # 3. Initialise database
    db_path = config.get("database", {}).get("path", "data/visitors.db")
    db = DatabaseManager(db_path)

    # 4. Handle dedicated Dashboard mode
    if args.dashboard:
        logger.info("FaceTrackAI | Launching Analytics Web Dashboard")
        server = DashboardServer(db=db, port=args.port)
        try:
            server.start()
        finally:
            db.close()
        return

    # 5. Handle dedicated Report Generation mode
    if args.report:
        logger.info("FaceTrackAI | Generating Analytics Reports")
        analytics = AnalyticsEngine(db)
        metrics = analytics.compute_metrics()
        reports = analytics.export_all(prefix="manual")
        print("\n=======================================================")
        print("  FaceTrackAI Analytics Summary")
        print(f"  Total Unique Visitors: {metrics['summary']['total_unique_visitors']}")
        print(f"  Total Activity Events: {metrics['summary']['total_events']}")
        print(f"  Average Dwell Time:    {metrics['summary']['avg_dwell_time_formatted']}")
        print(f"  Visitor Return Rate:   {metrics['retention']['return_rate_percentage']}%")
        print("-------------------------------------------------------")
        print(f"  Visitors CSV: {reports['visitors_csv']}")
        print(f"  Events CSV:   {reports['events_csv']}")
        print(f"  JSON Report:  {reports['json_report']}")
        print(f"  HTML Report:  {reports['html_report']}")
        print("=======================================================\n")
        db.close()
        return

    # 6. Standard Video Tracking Mode (Phase 3 Pipeline)
    logger.info("=" * 60)
    logger.info("FaceTrackAI | Starting — Phase 3 (Tracking, Analytics & Live Intelligence)")
    logger.info(f"FaceTrackAI | Config: {args.config}")
    logger.info(f"FaceTrackAI | Source: {config['video_source']}")
    logger.info("=" * 60)

    det_cfg = config.get("detection", {})
    detector = FaceDetector(
        model_path=det_cfg.get("model", "yolov8n-face.pt"),
        confidence=det_cfg.get("confidence_threshold", 0.5),
        skip_frames=det_cfg.get("detection_skip_frames", 5),
    )

    track_cfg = config.get("tracking", {})
    tracker = ByteTracker(
        max_missed_frames=track_cfg.get("max_missed_frames", 30),
        track_thresh=det_cfg.get("confidence_threshold", 0.5),
    )

    rec_cfg = config.get("recognition", {})
    recognizer = FaceRecognizer(
        similarity_threshold=rec_cfg.get("similarity_threshold", 0.5)
    )

    visitor_manager = VisitorManager(db=db, recognizer=recognizer)

    storage_cfg = config.get("storage", {})
    event_manager = EventManager(
        db=db,
        entry_dir=storage_cfg.get("entry_directory", "logs/entries"),
        exit_dir=storage_cfg.get("exit_directory", "logs/exits"),
    )

    pipeline = Pipeline(
        config=config,
        detector=detector,
        tracker=tracker,
        recognizer=recognizer,
        visitor_manager=visitor_manager,
        event_manager=event_manager,
    )

    processor = VideoProcessor(
        config=config,
        pipeline=pipeline,
    )

    try:
        processor.run()
    finally:
        # Automated Phase 3 Report Generation after video run
        unique_count = visitor_manager.unique_visitor_count()
        logger.info(f"FaceTrackAI | Finished | Unique Visitors = {unique_count}")

        analytics = AnalyticsEngine(db)
        reports = analytics.export_all(prefix="auto")
        logger.info(f"ANALYTICS | Automated reports generated: {reports['html_report']}")

        db.close()
        logger.info("FaceTrackAI | Shutdown complete")


if __name__ == "__main__":
    main()
