"""
logger.py
---------
Central logger for the audio transcription filler project.
Every log entry includes: timestamp, log level, filename, line number, and message.
Log is saved to /logs/app.log and also printed to console.
"""

import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
LOG_FILE = os.path.join(LOG_DIR, "app.log")

# Create logs directory if not exists
os.makedirs(LOG_DIR, exist_ok=True)


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance with file + console handlers.
    Each log line includes: time | level | filename:line_number | message

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("This is a log message")
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Format: 2024-01-15 10:23:45,123 | INFO     | services/transcriber.py:42 | message
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # --- File Handler (saves every log to app.log) ---
    file_handler = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Force every single log call to flush to disk immediately instead of
    # sitting in the OS write buffer. Without this, log lines can appear
    # to "stop updating" under Flask's debug/reloader process even though
    # they're still being written — they just land in bursts instead of
    # in real time.
    file_handler.flush = lambda: file_handler.stream.flush()
    original_emit = file_handler.emit

    def emit_and_flush(record):
        original_emit(record)
        file_handler.flush()

    file_handler.emit = emit_and_flush

    # --- Console Handler (prints to terminal) ---
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def log_separator(logger: logging.Logger, label: str = ""):
    """Logs a visual separator line. Useful between major steps."""
    line = "-" * 60
    if label:
        logger.info(f"{line} [ {label} ] {line}")
    else:
        logger.info(line * 2)