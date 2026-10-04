"""
src/logger.py
-------------
Centralized logging configuration for FaceTrackAI.

Sets up both file and console handlers so every module can import
`get_logger()` and write structured log lines to logs/events.log.
"""

import logging
import os
from typing import Optional


# Custom log format matching the spec:
#   2026-10-04 13:10:21 | INFO | FACE_DETECTED | visitor_001
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(log_file: str = "logs/events.log", level: int = logging.INFO) -> logging.Logger:
    """
    Initialise the root 'facetrack' logger with both a rotating file handler
    and a stream (console) handler.

    Parameters
    ----------
    log_file : str
        Path to the log file (read from config.json at startup).
    level : int
        Logging level (default INFO).

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    # Ensure the directory exists before opening the file
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else "logs", exist_ok=True)

    logger = logging.getLogger("facetrack")
    logger.setLevel(level)

    # Prevent duplicate handlers when the function is called more than once
    if logger.handlers:
        return logger

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # --- File handler ---------------------------------------------------
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    # --- Console handler ------------------------------------------------
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_logger() -> logging.Logger:
    """
    Return the existing 'facetrack' logger.
    Must be called after setup_logger() has been invoked once at startup.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger("facetrack")
