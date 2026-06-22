# Audio Transcription Filler

Audio Transcription Filler is a local Flask web app that helps fill missing OCR text pages using Malayalam speech transcription.

The app detects null or failed OCR pages in a JSON dictionary, accepts audio input for missing pages, transcribes the audio with a Malayalam Whisper model, and saves the filled JSON output.

## Features

- Detects null OCR pages from a dictionary-style JSON file
- Accepts audio uploads for missing pages
- Normalizes uploaded audio to 16kHz mono WAV using `ffmpeg`
- Transcribes Malayalam audio using `thennal/whisper-medium-ml`
- Supports long audio by chunking into 30-second segments
- Tracks page status: pending, transcribed, confirmed, skipped
- Saves filled OCR results and a transcription tracker JSON
- Includes a simple browser UI for upload, review, confirm, and download

## Requirements

- Python 3.12+
- `ffmpeg` installed and available on `PATH`

## Install

From the project root:

```bash
uv install
```

If you are not using `uv`, you can create a virtual environment manually and install dependencies from `pyproject.toml`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

## Run the app

Start the local Flask web server:

```bash
uv run python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Usage

1. Open the app in your browser.
2. Upload an OCR dictionary JSON file or use the sample data.
3. The app displays null pages that need transcription.
4. Upload audio for a page and preview the transcription.
5. Confirm the transcription to save it into the OCR dictionary.
6. Download the completed JSON when ready.

## Sample command-line pipeline

A simple command-line pipeline is available in `main.py` for testing or batch use.

```bash
python main.py
```

This script demonstrates null page detection, transcription, auto-confirmation, and saving results.

## Project structure

- `app.py` — Flask application and API endpoints
- `main.py` — pipeline driver for CLI/test use
- `config/settings.py` — project paths and model/audio settings
- `services/transcriber.py` — Whisper-based Malayalam audio transcription
- `services/dict_filler.py` — tracker, fill, confirm, skip, save logic
- `services/null_detector.py` — null OCR page detection rules
- `templates/index.html` — browser UI
- `static/app.js` — frontend app logic
- `static/style.css` — UI styles
- `data/input/` — input files directory
- `data/output/` — generated output files
- `logs/` — log files directory
- `models/` — local Hugging Face model cache

## Output files

- `data/output/filled_ocr_result.json` — filled OCR dictionary
- `data/output/transcription_tracker.json` — page-by-page tracker state

## Notes

- The first transcription run downloads the Whisper model (~1.5GB).
- `ffmpeg` is required to normalize browser audio uploads and unsupported formats.

## License

This project does not include a license file. Add one if you want to publish or share it more broadly.
