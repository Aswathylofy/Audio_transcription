"""
null_detector.py
----------------
Detects which pages in the OCR dictionary have failed or empty content.
Uses rule-based checks: None, empty string, too short, or garbage characters.
"""

import re
from utils.logger import get_logger
from config.settings import NULL_MIN_TEXT_LENGTH, NULL_MAX_GARBAGE_RATIO

logger = get_logger(__name__)


def is_null_page(page_key: str, value) -> bool:
    """
    Returns True if a page value is considered null/failed.

    Rules applied in order:
    1. Value is None
    2. Value is empty string
    3. Value is whitespace only
    4. Value is too short (< NULL_MIN_TEXT_LENGTH chars)
    5. Value has too many garbage/non-alphanumeric characters
    """
    logger.debug(f"Checking page '{page_key}' for null status")

    # Rule 1: None
    if value is None:
        logger.debug(f"  [{page_key}] FAILED — value is None")
        return True

    # Rule 2: Empty string
    if value == "":
        logger.debug(f"  [{page_key}] FAILED — value is empty string")
        return True

    # Rule 3: Whitespace only
    if value.strip() == "":
        logger.debug(f"  [{page_key}] FAILED — value is whitespace only")
        return True

    # Rule 4: Too short
    if len(value.strip()) < NULL_MIN_TEXT_LENGTH:
        logger.debug(f"  [{page_key}] FAILED — too short ({len(value.strip())} chars)")
        return True

    # Rule 5: Garbage OCR (too many non-alphanumeric characters)
    # \u0D00-\u0D7F is the Unicode block for Malayalam script —
    # included so valid Malayalam text is never flagged as garbage.
    total_chars = len(value)
    garbage_chars = len(re.sub(r'[a-zA-Z0-9\u0D00-\u0D7F\s]', '', value))
    garbage_ratio = garbage_chars / total_chars if total_chars > 0 else 1.0

    if garbage_ratio > NULL_MAX_GARBAGE_RATIO:
        logger.debug(
            f"  [{page_key}] FAILED — garbage ratio too high "
            f"({garbage_ratio:.0%} non-alphanumeric)"
        )
        return True

    logger.debug(f"  [{page_key}] PASSED — valid OCR content")
    return False


def detect_null_pages(ocr_dict: dict) -> list:
    """
    Scans the full OCR dictionary and returns a list of keys
    whose pages are null/failed and need audio transcription.

    Args:
        ocr_dict: dict like {"page_1": "text...", "page_2": None, ...}

    Returns:
        List of page keys that need audio input, e.g. ["page_2", "page_5"]
    """
    logger.info("Starting null page detection")
    logger.info(f"Total pages in OCR dictionary: {len(ocr_dict)}")

    null_pages = []

    for page_key, value in ocr_dict.items():
        if is_null_page(page_key, value):
            null_pages.append(page_key)

    logger.info(f"Null pages detected: {len(null_pages)} out of {len(ocr_dict)}")
    logger.info(f"Null page keys: {null_pages}")

    return null_pages