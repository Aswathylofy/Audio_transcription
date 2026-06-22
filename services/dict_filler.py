"""
dict_filler.py
--------------
Fills null/failed pages in the OCR dictionary with transcribed audio text.
Handles status tracking: pending, transcribed, confirmed, skipped.
"""

import json
import os
from datetime import datetime
from utils.logger import get_logger
from config.settings import DATA_OUTPUT_DIR, OUTPUT_JSON_FILENAME

logger = get_logger(__name__)


def build_status_tracker(ocr_dict: dict, null_pages: list) -> dict:
    """
    Creates a status tracker for all null pages.

    Structure:
    {
        "page_2": {
            "status": "pending",       # pending | transcribed | confirmed | skipped
            "original_value": "",
            "transcribed_text": None,
            "confirmed_text": None,
            "timestamp": None
        },
        ...
    }
    """
    logger.info(f"Building status tracker for {len(null_pages)} null pages")

    tracker = {}
    for page_key in null_pages:
        tracker[page_key] = {
            "status": "pending",
            "original_value": ocr_dict.get(page_key),
            "transcribed_text": None,
            "confirmed_text": None,
            "timestamp": None
        }
        logger.debug(f"Tracker entry created for '{page_key}' — status: pending")

    logger.info("Status tracker built successfully")
    return tracker


def fill_page(ocr_dict: dict, tracker: dict, page_key: str, transcribed_text: str) -> dict:
    """
    Fills a single null page slot with transcribed text.
    Updates both the OCR dictionary and the tracker.

    Status set to 'transcribed' — waiting for user confirmation.
    """
    logger.info(f"Filling page '{page_key}' with transcribed text")

    if page_key not in tracker:
        logger.warning(f"Page '{page_key}' not found in tracker — skipping fill")
        return tracker

    tracker[page_key]["transcribed_text"] = transcribed_text
    tracker[page_key]["status"] = "transcribed"
    tracker[page_key]["timestamp"] = datetime.now().isoformat()

    logger.debug(f"Tracker updated for '{page_key}' — status: transcribed")
    logger.debug(f"Text preview: '{transcribed_text[:80]}...'")
    return tracker


def confirm_page(ocr_dict: dict, tracker: dict, page_key: str) -> dict:
    """
    User confirmed the transcription is correct.
    Saves the text into the OCR dictionary permanently.
    Status set to 'confirmed'.
    """
    logger.info(f"User confirmed transcription for page '{page_key}'")

    if tracker[page_key]["status"] != "transcribed":
        logger.warning(
            f"Cannot confirm '{page_key}' — current status is "
            f"'{tracker[page_key]['status']}', expected 'transcribed'"
        )
        return tracker

    confirmed_text = tracker[page_key]["transcribed_text"]
    ocr_dict[page_key] = confirmed_text
    tracker[page_key]["confirmed_text"] = confirmed_text
    tracker[page_key]["status"] = "confirmed"
    tracker[page_key]["timestamp"] = datetime.now().isoformat()

    logger.info(f"Page '{page_key}' confirmed and saved to OCR dictionary")
    return tracker


def skip_page(tracker: dict, page_key: str) -> dict:
    """
    User chose to skip this page (can't read / no audio available).
    Status set to 'skipped'. OCR dictionary value remains as-is.
    """
    logger.info(f"Page '{page_key}' skipped by user")

    tracker[page_key]["status"] = "skipped"
    tracker[page_key]["timestamp"] = datetime.now().isoformat()

    logger.debug(f"Tracker updated for '{page_key}' — status: skipped")
    return tracker


def get_progress_summary(tracker: dict) -> dict:
    """
    Returns a summary of current progress across all null pages.
    """
    summary = {
        "total": len(tracker),
        "pending": 0,
        "transcribed": 0,
        "confirmed": 0,
        "skipped": 0
    }

    for page_key, info in tracker.items():
        status = info["status"]
        summary[status] = summary.get(status, 0) + 1

    logger.info(
        f"Progress — Total: {summary['total']} | "
        f"Confirmed: {summary['confirmed']} | "
        f"Pending: {summary['pending']} | "
        f"Transcribed(unconfirmed): {summary['transcribed']} | "
        f"Skipped: {summary['skipped']}"
    )
    return summary


def save_result(ocr_dict: dict, tracker: dict):
    """
    Saves the final filled OCR dictionary and tracker to output JSON files.
    """
    os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

    # Save filled OCR dictionary
    ocr_output_path = os.path.join(DATA_OUTPUT_DIR, OUTPUT_JSON_FILENAME)
    with open(ocr_output_path, "w", encoding="utf-8") as f:
        json.dump(ocr_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Filled OCR dictionary saved to: {ocr_output_path}")

    # Save tracker log
    tracker_path = os.path.join(DATA_OUTPUT_DIR, "transcription_tracker.json")
    with open(tracker_path, "w", encoding="utf-8") as f:
        json.dump(tracker, f, ensure_ascii=False, indent=2)
    logger.info(f"Transcription tracker saved to: {tracker_path}")