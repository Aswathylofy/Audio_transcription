"""
settings.py
-----------
Central configuration for the project.
All paths, model settings, and thresholds defined here.
"""

import os

# ── Base Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_INPUT_DIR  = os.path.join(BASE_DIR, "data", "input")
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# ── Whisper Models ─────────────────────────────────────────────
# Malayalam fine-tuned model — used only when detected language is Malayalam
HF_MODEL_ID_MALAYALAM = "thennal/whisper-medium-ml"

# Generic multilingual model — used for language detection on every file,
# and for transcription whenever the detected language is NOT Malayalam
HF_MODEL_ID_MULTILINGUAL = "openai/whisper-medium"

# Languages this project actively supports. Anything detected outside
# this set still gets transcribed (via the multilingual model) but is
# logged as "unsupported — best effort" since it wasn't explicitly tested.
SUPPORTED_LANGUAGES = {
    "ml": "Malayalam",
    "ta": "Tamil",
    "en": "English",
    "hi": "Hindi",
}

# Local cache directory — both models downloaded here on first run.
# Subsequent runs load from cache (no re-download).
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models")

# ── Null Page Detection Rules ─────────────────────────────────
NULL_MIN_TEXT_LENGTH   = 10    # less than 10 chars → treat as null
NULL_MAX_GARBAGE_RATIO = 0.6   # >60% non-alphanumeric chars → garbage

# ── Audio Settings ────────────────────────────────────────────
SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]

# ── Output Settings ───────────────────────────────────────────
OUTPUT_JSON_FILENAME = "filled_ocr_result.json"