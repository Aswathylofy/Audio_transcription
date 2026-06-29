"""
settings.py
-----------
Central configuration for the project.
All paths, model settings, and thresholds defined here.
"""

import os
from dotenv import load_dotenv

load_dotenv()
# ── Base Paths ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_INPUT_DIR  = os.path.join(BASE_DIR, "data", "input")
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
LOGS_DIR        = os.path.join(BASE_DIR, "logs")

# ── Transcription Models ────────────────────────────────────────
# Malayalam fine-tuned Whisper — best accuracy specifically for Malayalam
HF_MODEL_ID_MALAYALAM = "thennal/whisper-medium-ml"

# Generic multilingual Whisper — covers English, Hindi, Tamil, and
# anything else outside the Malayalam fine-tune's scope
HF_MODEL_ID_MULTILINGUAL = "openai/whisper-medium"

# IndicConformer — AI4Bharat's Conformer-based model covering 22 Indian
# languages with CTC and RNNT decoding. NOTE: does not support English.
HF_MODEL_ID_INDIC_CONFORMER = "ai4bharat/indic-conformer-600m-multilingual"

# Languages this project actively supports across the three models.
# Used to populate the language dropdown in the UI and for logging.
# Not every language is valid for every model — see IndicConformer note above.
SUPPORTED_LANGUAGES = {
    "ml": "Malayalam",
    "ta": "Tamil",
    "en": "English",
    "hi": "Hindi",
}

# Local cache directory — all three models downloaded here on first run.
# Subsequent runs load from cache (no re-download).
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "models")

# ── Null Page Detection Rules ─────────────────────────────────
NULL_MIN_TEXT_LENGTH   = 10    # less than 10 chars → treat as null
NULL_MAX_GARBAGE_RATIO = 0.6   # >60% non-alphanumeric chars → garbage

# ── Audio Settings ────────────────────────────────────────────
SUPPORTED_AUDIO_FORMATS = [".mp3", ".wav", ".m4a", ".ogg", ".flac"]

# ── Output Settings ───────────────────────────────────────────
OUTPUT_JSON_FILENAME = "filled_ocr_result.json"

# ── HF token ────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN")