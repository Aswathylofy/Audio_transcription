# """
# app.py
# ------
# Flask web app for the Audio Transcription Filler.

# Workflow:
# 1. Load OCR dictionary (sample or uploaded JSON)
# 2. Show null pages that need audio
# 3. User uploads audio for a specific page
# 4. Backend transcribes it using transcriber.py
# 5. Show preview to user
# 6. User confirms -> saved into ocr_dict
# 7. User can download final filled JSON

# Run:
#     uv run python app.py
# Then open:
#     http://127.0.0.1:5000
# """

# import os
# import json
# from flask import Flask, render_template, request, jsonify, send_file

# from utils.logger import get_logger, log_separator
# from services.null_detector import detect_null_pages
# from services.transcriber import transcribe_audio
# from services.dict_filler import (
#     build_status_tracker,
#     fill_page,
#     confirm_page,
#     skip_page,
#     get_progress_summary,
#     save_result
# )
# from config.settings import DATA_INPUT_DIR, DATA_OUTPUT_DIR

# logger = get_logger(__name__)

# app = Flask(__name__)
# app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload (long audio files)

# os.makedirs(DATA_INPUT_DIR, exist_ok=True)
# os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# # ── In-memory session state (single-user local tool) ───────────
# # For a multi-user deployment this would move to a database,
# # but for a local single-user tool, memory is enough.
# STATE = {
#     "ocr_dict": {},
#     "tracker": {}
# }


# # ── Sample data for first-time testing ──────────────────────────
# SAMPLE_OCR_DICT = {
#     "page_1": "ചെറുപ്പം മുതൽക്കേ സംഗീതത്തിന്റെയും നൃത്തത്തിന്റെയും ലോകത്തായിരുന്നു ശ്രീവിദ്യ വളർന്നത്.",
#     "page_2": None,
#     "page_3": "അമൃതം ഗമയ എന്ന ചിത്രത്തിലെ ഡോക്ടറുടെ കഥാപാത്രം ഇതിലൊന്നാണ്.",
#     "page_4": "",
#     "page_5": "ആരെങ്കിലും ആ പൊതുനിലപാട് സ്വീകരിച്ചില്ലെന്നത് സമ്മാനിതനാവാനുള്ള കാരണമല്ല.",
#     "page_6": "ആഗോളതലത്തിൽ ദരിദ്ര രാജ്യങ്ങൾക്കായി ബദൽ സാമ്പത്തിക പരിഷ്കരണ നിർദ്ദേശങ്ങൾ നൽകിയും ചാവെസ് ശ്രദ്ധ നേടി.",
#     "page_7": "മംഗളം എന്ന വാക്കാൽ വിവക്ഷിക്കാവുന്ന ഒന്നിലധികം കാര്യങ്ങളുണ്ട്.",
# }


# def init_state(ocr_dict: dict):
#     """Initializes STATE with a fresh OCR dictionary and builds the tracker."""
#     log_separator(logger, "INIT STATE")
#     logger.info(f"Initializing state with {len(ocr_dict)} pages")

#     STATE["ocr_dict"] = ocr_dict
#     null_pages = detect_null_pages(ocr_dict)
#     STATE["tracker"] = build_status_tracker(ocr_dict, null_pages)

#     logger.info(f"State initialized — {len(null_pages)} null pages need audio")


# # Initialize with sample data at startup so the UI has something to show
# init_state(dict(SAMPLE_OCR_DICT))


# # ── Routes ────────────────────────────────────────────────────

# @app.route("/")
# def index():
#     """Renders the main UI page."""
#     logger.info("Serving index page")
#     return render_template("index.html")


# @app.route("/api/status", methods=["GET"])
# def api_status():
#     """
#     Returns the current state: all pages, their content, and tracker status.
#     Used by frontend to render the page list.
#     """
#     logger.debug("API: /api/status called")

#     pages = []
#     for page_key, value in STATE["ocr_dict"].items():
#         tracker_info = STATE["tracker"].get(page_key)
#         pages.append({
#             "page_key": page_key,
#             "content": value if value else None,
#             "is_null": page_key in STATE["tracker"],
#             "status": tracker_info["status"] if tracker_info else "ok",
#             "transcribed_text": tracker_info["transcribed_text"] if tracker_info else None
#         })

#     summary = get_progress_summary(STATE["tracker"]) if STATE["tracker"] else {
#         "total": 0, "pending": 0, "transcribed": 0, "confirmed": 0, "skipped": 0
#     }

#     return jsonify({
#         "pages": pages,
#         "summary": summary
#     })


# @app.route("/api/upload_json", methods=["POST"])
# def api_upload_json():
#     """
#     Accepts an uploaded OCR dictionary JSON file and re-initializes state.
#     """
#     log_separator(logger, "API: UPLOAD JSON")

#     if "file" not in request.files:
#         logger.error("No file part in upload request")
#         return jsonify({"error": "No file uploaded"}), 400

#     file = request.files["file"]
#     if file.filename == "":
#         logger.error("Empty filename in upload request")
#         return jsonify({"error": "No file selected"}), 400

#     try:
#         ocr_dict = json.load(file.stream)
#         logger.info(f"Uploaded OCR dictionary parsed — {len(ocr_dict)} pages")
#     except Exception as e:
#         logger.error(f"Failed to parse uploaded JSON: {e}")
#         return jsonify({"error": f"Invalid JSON file: {e}"}), 400

#     init_state(ocr_dict)
#     return jsonify({"message": "OCR dictionary loaded successfully", "page_count": len(ocr_dict)})


# @app.route("/api/transcribe/<page_key>", methods=["POST"])
# def api_transcribe(page_key):
#     """
#     Accepts an audio file upload for a specific null page,
#     runs transcription, and returns the preview text.
#     Does NOT save into ocr_dict yet — waits for user confirmation.
#     """
#     log_separator(logger, f"API: TRANSCRIBE {page_key}")

#     if page_key not in STATE["tracker"]:
#         logger.error(f"Page '{page_key}' is not a null page or does not exist")
#         return jsonify({"error": "Invalid page key"}), 400

#     if "audio" not in request.files:
#         logger.error("No audio file in request")
#         return jsonify({"error": "No audio file uploaded"}), 400

#     audio_file = request.files["audio"]
#     if audio_file.filename == "":
#         logger.error("Empty audio filename")
#         return jsonify({"error": "No audio file selected"}), 400

#     # Save uploaded audio to data/input/
#     ext = os.path.splitext(audio_file.filename)[1].lower()
#     saved_path = os.path.join(DATA_INPUT_DIR, f"{page_key}{ext}")
#     audio_file.save(saved_path)
#     logger.info(f"Audio saved for '{page_key}': {saved_path}")

#     # Run transcription (this may take a while for long audio)
#     transcribed_text = transcribe_audio(saved_path)

#     if transcribed_text is None:
#         logger.error(f"Transcription failed for '{page_key}'")
#         return jsonify({"error": "Transcription failed. Check server logs."}), 500

#     # Store as 'transcribed' status — waiting for user confirmation
#     fill_page(STATE["ocr_dict"], STATE["tracker"], page_key, transcribed_text)

#     logger.info(f"Transcription preview ready for '{page_key}'")
#     return jsonify({
#         "page_key": page_key,
#         "transcribed_text": transcribed_text,
#         "status": "transcribed"
#     })


# @app.route("/api/confirm/<page_key>", methods=["POST"])
# def api_confirm(page_key):
#     """
#     User confirmed the transcription preview is correct.
#     Saves it permanently into the OCR dictionary.
#     Accepts optional edited text from the user in JSON body: {"text": "..."}
#     """
#     log_separator(logger, f"API: CONFIRM {page_key}")

#     if page_key not in STATE["tracker"]:
#         logger.error(f"Page '{page_key}' not found in tracker")
#         return jsonify({"error": "Invalid page key"}), 400

#     # Allow user to edit text before confirming
#     body = request.get_json(silent=True) or {}
#     edited_text = body.get("text")

#     if edited_text:
#         logger.info(f"User edited text for '{page_key}' before confirming")
#         STATE["tracker"][page_key]["transcribed_text"] = edited_text

#     confirm_page(STATE["ocr_dict"], STATE["tracker"], page_key)

#     logger.info(f"Page '{page_key}' confirmed")
#     return jsonify({"page_key": page_key, "status": "confirmed"})


# @app.route("/api/skip/<page_key>", methods=["POST"])
# def api_skip(page_key):
#     """User chose to skip this page — no audio available."""
#     log_separator(logger, f"API: SKIP {page_key}")

#     if page_key not in STATE["tracker"]:
#         logger.error(f"Page '{page_key}' not found in tracker")
#         return jsonify({"error": "Invalid page key"}), 400

#     skip_page(STATE["tracker"], page_key)
#     logger.info(f"Page '{page_key}' skipped")
#     return jsonify({"page_key": page_key, "status": "skipped"})


# @app.route("/api/save", methods=["POST"])
# def api_save():
#     """
#     Saves the final OCR dictionary and tracker to data/output/ as JSON files.
#     """
#     log_separator(logger, "API: SAVE RESULTS")

#     save_result(STATE["ocr_dict"], STATE["tracker"])
#     logger.info("Results saved successfully")

#     return jsonify({"message": "Saved successfully"})


# @app.route("/api/download")
# def api_download():
#     """Lets the user download the final filled_ocr_result.json file."""
#     logger.info("Download requested for filled_ocr_result.json")

#     filepath = os.path.join(DATA_OUTPUT_DIR, "filled_ocr_result.json")
#     if not os.path.exists(filepath):
#         logger.error("filled_ocr_result.json does not exist yet — save first")
#         return jsonify({"error": "No saved result yet. Click Save first."}), 404

#     return send_file(filepath, as_attachment=True)


# if __name__ == "__main__":
#     log_separator(logger, "FLASK APP START")
#     logger.info("Starting Flask app on http://127.0.0.1:5000")
#     app.run(debug=True, host="127.0.0.1", port=5000)



"""
app.py
------
Flask web app for the Audio Transcription Filler.

Workflow:
1. Load OCR dictionary (sample or uploaded JSON)
2. Show null pages that need audio
3. User uploads audio for a specific page
4. Backend transcribes it using transcriber.py
5. Show preview to user
6. User confirms -> saved into ocr_dict
7. User can download final filled JSON

Run:
    uv run python app.py
Then open:
    http://127.0.0.1:5000
"""

import os
import json
import subprocess
from flask import Flask, render_template, request, jsonify, send_file

from utils.logger import get_logger, log_separator
from services.null_detector import detect_null_pages
from services.transcriber import (
    transcribe_audio,
    MODEL_WHISPER_MALAYALAM,
    MODEL_INDIC_CONFORMER,
)
from services.dict_filler import (
    build_status_tracker,
    fill_page,
    confirm_page,
    skip_page,
    get_progress_summary,
    save_result
)
from config.settings import DATA_INPUT_DIR, DATA_OUTPUT_DIR

logger = get_logger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500MB max upload (long audio files)

os.makedirs(DATA_INPUT_DIR, exist_ok=True)
os.makedirs(DATA_OUTPUT_DIR, exist_ok=True)

# ── In-memory session state (single-user local tool) ───────────
# For a multi-user deployment this would move to a database,
# but for a local single-user tool, memory is enough.
STATE = {
    "ocr_dict": {},
    "tracker": {}
}


# ── Sample data for first-time testing ──────────────────────────
SAMPLE_OCR_DICT = {
    "page_1": "ചെറുപ്പം മുതൽക്കേ സംഗീതത്തിന്റെയും നൃത്തത്തിന്റെയും ലോകത്തായിരുന്നു ശ്രീവിദ്യ വളർന്നത്.",
    "page_2": None,
    "page_3": "അമൃതം ഗമയ എന്ന ചിത്രത്തിലെ ഡോക്ടറുടെ കഥാപാത്രം ഇതിലൊന്നാണ്.",
    "page_4": "",
    "page_5": "ആരെങ്കിലും ആ പൊതുനിലപാട് സ്വീകരിച്ചില്ലെന്നത് സമ്മാനിതനാവാനുള്ള കാരണമല്ല.",
    "page_6": "ആഗോളതലത്തിൽ ദരിദ്ര രാജ്യങ്ങൾക്കായി ബദൽ സാമ്പത്തിക പരിഷ്കരണ നിർദ്ദേശങ്ങൾ നൽകിയും ചാവെസ് ശ്രദ്ധ നേടി.",
    "page_7": "മംഗളം എന്ന വാക്കാൽ വിവക്ഷിക്കാവുന്ന ഒന്നിലധികം കാര്യങ്ങളുണ്ട്.",
}


def normalize_audio_to_wav(input_path: str) -> str | None:
    """
    Converts any incoming audio file to 16kHz mono WAV using ffmpeg.

    This handles browser-recorded audio (webm/opus, ogg/opus from
    MediaRecorder) as well as uploaded formats (mp3, m4a, etc.) that
    torchaudio's soundfile backend can't reliably decode on its own.

    Returns the path to the normalized .wav file, or None on failure.
    """
    output_path = os.path.splitext(input_path)[0] + "_normalized.wav"

    cmd = [
        "ffmpeg",
        "-y",                 # overwrite output if it already exists
        "-i", input_path,
        "-ar", "16000",       # 16kHz sample rate (Whisper requirement)
        "-ac", "1",           # mono
        output_path
    ]

    logger.info(f"Normalizing audio with ffmpeg: {input_path} -> {output_path}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        logger.error(f"ffmpeg conversion failed: {result.stderr.strip()}")
        return None

    logger.debug("ffmpeg conversion succeeded")
    return output_path


def init_state(ocr_dict: dict):
    """Initializes STATE with a fresh OCR dictionary and builds the tracker."""
    log_separator(logger, "INIT STATE")
    logger.info(f"Initializing state with {len(ocr_dict)} pages")

    STATE["ocr_dict"] = ocr_dict
    null_pages = detect_null_pages(ocr_dict)
    STATE["tracker"] = build_status_tracker(ocr_dict, null_pages)

    logger.info(f"State initialized — {len(null_pages)} null pages need audio")


# Initialize with sample data at startup so the UI has something to show
init_state(dict(SAMPLE_OCR_DICT))


# ── Routes ────────────────────────────────────────────────────

@app.route("/api/models", methods=["GET"])
def api_models():
    """
    Returns the two user-facing model choices for the dropdown in the UI.

    Both options auto-detect language per chunk in the background and
    fall back to a general multilingual Whisper model for anything
    that isn't Malayalam — they only differ in which model handles
    Malayalam specifically. No language or decoding-mode picker is
    needed in the UI anymore; both are now fully automatic.
    """
    logger.debug("API: /api/models called")

    return jsonify({
        "models": [
            {
                "id": MODEL_WHISPER_MALAYALAM,
                "label": "Thennal Whisper (Malayalam fine-tuned)",
            },
            {
                "id": MODEL_INDIC_CONFORMER,
                "label": "IndicConformer (AI4Bharat)",
            },
        ],
    })


@app.route("/")
def index():
    """Renders the main UI page."""
    logger.info("Serving index page")
    return render_template("index.html")


@app.route("/api/status", methods=["GET"])
def api_status():
    """
    Returns the current state: all pages, their content, and tracker status.
    Used by frontend to render the page list.
    """
    logger.debug("API: /api/status called")

    pages = []
    for page_key, value in STATE["ocr_dict"].items():
        tracker_info = STATE["tracker"].get(page_key)
        pages.append({
            "page_key": page_key,
            "content": value if value else None,
            "is_null": page_key in STATE["tracker"],
            "status": tracker_info["status"] if tracker_info else "ok",
            "transcribed_text": tracker_info["transcribed_text"] if tracker_info else None
        })

    summary = get_progress_summary(STATE["tracker"]) if STATE["tracker"] else {
        "total": 0, "pending": 0, "transcribed": 0, "confirmed": 0, "skipped": 0
    }

    return jsonify({
        "pages": pages,
        "summary": summary
    })


@app.route("/api/upload_json", methods=["POST"])
def api_upload_json():
    """
    Accepts an uploaded OCR dictionary JSON file and re-initializes state.
    """
    log_separator(logger, "API: UPLOAD JSON")

    if "file" not in request.files:
        logger.error("No file part in upload request")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        logger.error("Empty filename in upload request")
        return jsonify({"error": "No file selected"}), 400

    try:
        ocr_dict = json.load(file.stream)
        logger.info(f"Uploaded OCR dictionary parsed — {len(ocr_dict)} pages")
    except Exception as e:
        logger.error(f"Failed to parse uploaded JSON: {e}")
        return jsonify({"error": f"Invalid JSON file: {e}"}), 400

    init_state(ocr_dict)
    return jsonify({"message": "OCR dictionary loaded successfully", "page_count": len(ocr_dict)})


@app.route("/api/transcribe/<page_key>", methods=["POST"])
def api_transcribe(page_key):
    """
    Accepts an audio file upload for a specific null page, along with
    the user's chosen "Malayalam handler" model, runs transcription,
    and returns the preview text.
    Does NOT save into ocr_dict yet — waits for user confirmation.

    Language detection and model routing (including the general
    multilingual fallback for non-Malayalam audio) all happen
    automatically inside transcribe_audio() — no language or decoding
    mode needs to be sent from the frontend anymore.

    Expected form field alongside the 'audio' file:
        model_choice : "whisper_malayalam" | "indic_conformer"
    """
    log_separator(logger, f"API: TRANSCRIBE {page_key}")

    if page_key not in STATE["tracker"]:
        logger.error(f"Page '{page_key}' is not a null page or does not exist")
        return jsonify({"error": "Invalid page key"}), 400

    if "audio" not in request.files:
        logger.error("No audio file in request")
        return jsonify({"error": "No audio file uploaded"}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        logger.error("Empty audio filename")
        return jsonify({"error": "No audio file selected"}), 400

    # Read the user's model selection from the form data.
    # Default preserves old behavior if the frontend hasn't sent this yet.
    model_choice = request.form.get("model_choice", MODEL_WHISPER_MALAYALAM)

    logger.info(f"Malayalam handler choice for '{page_key}': {model_choice}")

    # Save uploaded audio to data/input/
    # No extension fallback: browser MediaRecorder blobs may not carry
    # a clean extension in the filename, so default to .webm.
    ext = os.path.splitext(audio_file.filename)[1].lower() or ".webm"
    saved_path = os.path.join(DATA_INPUT_DIR, f"{page_key}{ext}")
    audio_file.save(saved_path)
    logger.info(f"Audio saved for '{page_key}': {saved_path}")

    # Normalize to 16kHz mono WAV via ffmpeg — handles recorded
    # webm/opus audio and uploaded mp3/m4a/etc. uniformly, and
    # sidesteps soundfile's limited codec support.
    normalized_path = normalize_audio_to_wav(saved_path)
    if normalized_path is None:
        logger.error(f"Audio normalization failed for '{page_key}'")
        return jsonify({"error": "Audio conversion failed. Check server logs."}), 500

    # Run transcription — language detection and model routing for
    # each chunk happen automatically inside transcribe_audio()
    transcribed_text = transcribe_audio(
        normalized_path,
        model_choice=model_choice,
    )

    if transcribed_text is None:
        logger.error(f"Transcription failed for '{page_key}'")
        return jsonify({"error": "Transcription failed. Check server logs."}), 500

    # Store as 'transcribed' status — waiting for user confirmation
    fill_page(STATE["ocr_dict"], STATE["tracker"], page_key, transcribed_text)

    logger.info(f"Transcription preview ready for '{page_key}'")
    return jsonify({
        "page_key": page_key,
        "transcribed_text": transcribed_text,
        "status": "transcribed",
        "model_used": model_choice,
    })


@app.route("/api/confirm/<page_key>", methods=["POST"])
def api_confirm(page_key):
    """
    User confirmed the transcription preview is correct.
    Saves it permanently into the OCR dictionary.
    Accepts optional edited text from the user in JSON body: {"text": "..."}
    """
    log_separator(logger, f"API: CONFIRM {page_key}")

    if page_key not in STATE["tracker"]:
        logger.error(f"Page '{page_key}' not found in tracker")
        return jsonify({"error": "Invalid page key"}), 400

    # Allow user to edit text before confirming
    body = request.get_json(silent=True) or {}
    edited_text = body.get("text")

    if edited_text:
        logger.info(f"User edited text for '{page_key}' before confirming")
        STATE["tracker"][page_key]["transcribed_text"] = edited_text

    confirm_page(STATE["ocr_dict"], STATE["tracker"], page_key)

    logger.info(f"Page '{page_key}' confirmed")
    return jsonify({"page_key": page_key, "status": "confirmed"})


@app.route("/api/skip/<page_key>", methods=["POST"])
def api_skip(page_key):
    """User chose to skip this page — no audio available."""
    log_separator(logger, f"API: SKIP {page_key}")

    if page_key not in STATE["tracker"]:
        logger.error(f"Page '{page_key}' not found in tracker")
        return jsonify({"error": "Invalid page key"}), 400

    skip_page(STATE["tracker"], page_key)
    logger.info(f"Page '{page_key}' skipped")
    return jsonify({"page_key": page_key, "status": "skipped"})


@app.route("/api/save", methods=["POST"])
def api_save():
    """
    Saves the final OCR dictionary and tracker to data/output/ as JSON files.
    """
    log_separator(logger, "API: SAVE RESULTS")

    save_result(STATE["ocr_dict"], STATE["tracker"])
    logger.info("Results saved successfully")

    return jsonify({"message": "Saved successfully"})


@app.route("/api/download")
def api_download():
    """Lets the user download the final filled_ocr_result.json file."""
    logger.info("Download requested for filled_ocr_result.json")

    filepath = os.path.join(DATA_OUTPUT_DIR, "filled_ocr_result.json")
    if not os.path.exists(filepath):
        logger.error("filled_ocr_result.json does not exist yet — save first")
        return jsonify({"error": "No saved result yet. Click Save first."}), 404

    return send_file(filepath, as_attachment=True)


if __name__ == "__main__":
    log_separator(logger, "FLASK APP START")
    logger.info("Starting Flask app on http://127.0.0.1:5000")
    # use_reloader=False — Flask's reloader spawns a second child process
    # that re-imports this whole module. That second process was the
    # actual cause of logs appearing to "stop updating": both processes
    # were writing to app.log, but only one of them was the one you were
    # watching/testing against, so writes looked inconsistent or delayed.
    # debug=True is kept so you still get Flask's error pages.
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=5000)