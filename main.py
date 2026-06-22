"""
main.py
-------
Main entry point for the Audio Transcription Filler pipeline.
Orchestrates: null detection → audio transcription → dictionary fill → save.

Usage:
    python main.py
"""

import os
os.environ["HF_HOME"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
from utils.logger import get_logger, log_separator
from services.null_detector import detect_null_pages
from services.transcriber import transcribe_audio
from services.dict_filler import (
    build_status_tracker,
    fill_page,
    confirm_page,
    skip_page,
    get_progress_summary,
    save_result
)

logger = get_logger(__name__)


def run_pipeline(ocr_dict: dict, audio_map: dict) -> dict:
    """
    Full pipeline:
    1. Detect null pages
    2. For each null page, transcribe its audio
    3. Fill the dictionary
    4. Save the result

    Args:
        ocr_dict  : The OCR result dictionary {"page_1": "text", "page_2": None, ...}
        audio_map : Mapping of page keys to audio file paths
                    {"page_2": "/path/to/page_2.mp3", "page_5": "/path/to/page_5.wav"}

    Returns:
        Updated OCR dictionary with null pages filled
    """
    log_separator(logger, "PIPELINE START")
    logger.info(f"OCR dictionary received with {len(ocr_dict)} pages")
    logger.info(f"Audio map received with {len(audio_map)} entries")

    # ── Step 1: Detect null pages ──────────────────────────────
    log_separator(logger, "STEP 1: NULL DETECTION")
    null_pages = detect_null_pages(ocr_dict)

    if not null_pages:
        logger.info("No null pages found. OCR dictionary is complete. Exiting pipeline.")
        return ocr_dict

    # ── Step 2: Build status tracker ──────────────────────────
    log_separator(logger, "STEP 2: BUILD TRACKER")
    tracker = build_status_tracker(ocr_dict, null_pages)

    # ── Step 3: Transcribe and fill each null page ─────────────
    log_separator(logger, "STEP 3: TRANSCRIBE & FILL")

    for page_key in null_pages:
        logger.info(f"Processing null page: '{page_key}'")

        # Check if audio is provided for this page
        if page_key not in audio_map:
            logger.warning(f"No audio file provided for '{page_key}' — skipping")
            skip_page(tracker, page_key)
            continue

        audio_path = audio_map[page_key]
        logger.info(f"Audio file for '{page_key}': {audio_path}")

        # Transcribe
        transcribed_text = transcribe_audio(audio_path)

        if transcribed_text is None:
            logger.error(f"Transcription failed for '{page_key}' — skipping")
            skip_page(tracker, page_key)
            continue

        # Fill into tracker (status = transcribed, waiting for confirm)
        fill_page(ocr_dict, tracker, page_key, transcribed_text)

        # Auto-confirm (set to False if you want manual confirmation step)
        # In the UI phase, this will be replaced by user button click
        AUTO_CONFIRM = True
        if AUTO_CONFIRM:
            logger.info(f"Auto-confirming '{page_key}'")
            confirm_page(ocr_dict, tracker, page_key)

    # ── Step 4: Progress Summary ───────────────────────────────
    log_separator(logger, "STEP 4: PROGRESS SUMMARY")
    get_progress_summary(tracker)

    # ── Step 5: Save Results ───────────────────────────────────
    log_separator(logger, "STEP 5: SAVE RESULTS")
    save_result(ocr_dict, tracker)

    log_separator(logger, "PIPELINE COMPLETE")
    return ocr_dict


# ── Example / Test Run ─────────────────────────────────────────
if __name__ == "__main__":

    logger.info("Running main.py in test mode")

    # Simulated OCR dictionary (as it would come from the big project)
    sample_ocr_dict = {
        "page_1": "ചെറുപ്പം മുതൽക്കേ സംഗീതത്തിന്റെയും നൃത്തത്തിന്റെയും ലോകത്തായിരുന്നു ശ്രീവിദ്യ വളർന്നത്.",
        "page_2": "",         # failed OCR — needs audio
        "page_3": "അമൃതം‌ ഗമയ എന്ന ചിത്രത്തിലെ ഡോക്ടറുടെ കഥാപാത്രം ഇതിലൊന്നാണ്.",
        "page_4": "ആഗോളതലത്തിൽ ദരിദ്ര രാജ്യങ്ങൾക്കായി ബദൽ സാമ്പത്തിക പരിഷ്കരണ നിർദ്ദേശങ്ങൾ നൽകിയും ചാവെസ് ശ്രദ്ധ നേടി.",            
        "page_5": "ആരെങ്കിലും ആ പൊതുനിലപാട് സ്വീകരിച്ചില്ലെന്നത് സമ്മാനിതനാവാനുള്ള കാരണമല്ല.",     
        "page_6": "ആരെങ്കിലും ആ പൊതുനിലപാട് സ്വീകരിച്ചില്ലെന്നത് സമ്മാനിതനാവാനുള്ള കാരണമല്ല.",      
        "page_7": "മംഗളം എന്ന വാക്കാൽ വിവക്ഷിക്കാവുന്ന ഒന്നിലധികം കാര്യങ്ങളുണ്ട്.",       
    }

    # Audio map — in real usage, UI provides these paths
    # For testing, point to actual audio files in data/input/
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sample_audio_map = {
        "page_2": os.path.join(BASE_DIR, "data", "input", "page_2.wav"),
        
    }
    
    logger.info("Sample OCR dict and audio map prepared")
    result = run_pipeline(sample_ocr_dict, sample_audio_map)
    logger.info("Pipeline finished. Final dictionary:")

    for k, v in result.items():
        preview = str(v)[:60] if v else "NULL/EMPTY"
        logger.info(f"  {k}: {preview}")