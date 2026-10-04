"""
dashboard.py
------------
Stand-alone entry point to launch the FaceTrackAI Live Analytics Web Dashboard.

Usage
-----
    python dashboard.py
    python dashboard.py --port 8080 --host 0.0.0.0
"""

import argparse
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import DatabaseManager
from src.dashboard import DashboardServer
from src.logger import setup_logger
from app import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FaceTrackAI — Live Intelligence & Visitor Dashboard"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the web server (default: 8000)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.json",
        help="Path to configuration file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    log_file = config.get("logging", {}).get("log_file", "logs/events.log")
    setup_logger(log_file)

    db_path = config.get("database", {}).get("path", "data/visitors.db")
    db = DatabaseManager(db_path)

    server = DashboardServer(db=db, port=args.port, host=args.host)
    try:
        server.start()
    finally:
        db.close()


if __name__ == "__main__":
    main()
