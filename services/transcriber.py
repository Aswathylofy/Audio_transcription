"""
transcriber.py
--------------
Handles multilingual audio transcription using two models, manually
selectable by the user in the UI:
    1. thennal/whisper-medium-ml                     (Malayalam fine-tuned Whisper)
    2. ai4bharat/indic-conformer-600m-multilingual    (IndicConformer, 22 Indian languages)

For every chunk, the spoken language is first auto-detected using a
third model, openai/whisper-medium, which runs invisibly in the
background and is never shown as a selectable option in the UI.

Based on the detected language, transcription is then routed as follows:
    - If the detected language is Malayalam:
        - "thennal" selected   -> Malayalam Whisper transcribes it
        - "indic_conformer" selected -> IndicConformer transcribes it
    - If the detected language is anything else (Tamil, English, Hindi, ...):
        - Either selection falls back to the same general multilingual
          Whisper model, since neither Thennal (Malayalam-only) nor
          IndicConformer's strengths apply outside Malayalam in this setup.

In other words, the two UI options only differ in which model handles
Malayalam specifically — everything else is handled identically by the
general multilingual model regardless of which option is selected.

IndicConformer decoding mode is fixed to RNNT. CTC decoding was tried
and found to produce noticeably worse results in practice, so it has
been removed as an option entirely.

Long audio files are split into 30-second chunks (Whisper's processing
window limit) and transcribed sequentially, then joined together.
IndicConformer does not have this 30s limit, but the same chunking is
applied for consistency and comparable per-chunk logging.

All models are downloaded on first run and cached locally in /models.
Subsequent runs load from cache — no internet needed after first run.
"""

import os
import torch
import torchaudio
from utils.logger import get_logger
from config.settings import (
    HF_MODEL_ID_MALAYALAM,
    HF_MODEL_ID_MULTILINGUAL,
    HF_MODEL_ID_INDIC_CONFORMER,
    SUPPORTED_LANGUAGES,
    MODEL_CACHE_DIR,
    SUPPORTED_AUDIO_FORMATS,
    HF_TOKEN,
)

logger = get_logger(__name__)

# Global model components — loaded once, reused across calls.
# All three models (including the hidden language detector / fallback
# transcriber) are kept in memory simultaneously so switching between
# them per-chunk has zero reload delay.
_components = None

# Whisper's fixed processing window — do not change.
# This is a model architecture limit, not a configurable setting.
CHUNK_DURATION_SEC = 30
SAMPLE_RATE = 16000
SAMPLES_PER_CHUNK = CHUNK_DURATION_SEC * SAMPLE_RATE

# Model identifiers for the UI dropdown — only these two are user-facing.
# The generic multilingual Whisper model still loads internally (see
# load_models / detect_language) but is never exposed as a choice here —
# it works invisibly as both the language detector and the fallback
# transcriber for any non-Malayalam audio under either selection.
MODEL_WHISPER_MALAYALAM = "whisper_malayalam"
MODEL_INDIC_CONFORMER = "indic_conformer"

# IndicConformer decoding mode is no longer user-selectable — CTC was
# tested and performed noticeably worse, so RNNT is now the only mode used.
INDIC_CONFORMER_DECODING_MODE = "rnnt"


def load_models():
    """
    Loads ALL THREE models into memory at startup:
        1. thennal/whisper-medium-ml                     — Malayalam fine-tuned Whisper (user-selectable)
        2. openai/whisper-medium                         — generic multilingual Whisper
                                                             (NOT user-selectable — used only as a
                                                             hidden language detector for IndicConformer)
        3. ai4bharat/indic-conformer-600m-multilingual    — IndicConformer (user-selectable)

    Keeping all three loaded simultaneously means switching between
    them per-file has zero reload delay during transcription.

    Downloads on first run (~1.5GB + ~1.5GB + ~2.4GB), then loads from
    local cache on every run after that.
    """
    global _components

    if _components is not None:
        logger.debug("Models already loaded, reusing cached instances")
        return _components

    logger.info("Loading transcription models...")
    logger.info(f"Cache directory: {MODEL_CACHE_DIR}")
    logger.info("First run will download all three models (~5.5GB total) — please wait...")

    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoModel

        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        # ── Malayalam fine-tuned Whisper ──────────────────────
        logger.info(f"Loading Malayalam model: '{HF_MODEL_ID_MALAYALAM}'")
        ml_processor = WhisperProcessor.from_pretrained(
            HF_MODEL_ID_MALAYALAM,
            cache_dir=MODEL_CACHE_DIR
        )
        ml_model = WhisperForConditionalGeneration.from_pretrained(
            HF_MODEL_ID_MALAYALAM,
            cache_dir=MODEL_CACHE_DIR
        )
        ml_model.eval()
        logger.info("Malayalam model loaded successfully")

        # ── Generic multilingual Whisper ──────────────────────
        logger.info(f"Loading multilingual model: '{HF_MODEL_ID_MULTILINGUAL}'")
        multi_processor = WhisperProcessor.from_pretrained(
            HF_MODEL_ID_MULTILINGUAL,
            cache_dir=MODEL_CACHE_DIR
        )
        multi_model = WhisperForConditionalGeneration.from_pretrained(
            HF_MODEL_ID_MULTILINGUAL,
            cache_dir=MODEL_CACHE_DIR
        )
        multi_model.eval()
        logger.info("Multilingual model loaded successfully")

        # ── IndicConformer ─────────────────────────────────────
        # Loaded via plain transformers.AutoModel (trust_remote_code=True
        # is required because AI4Bharat ships custom model code alongside
        # the weights). No NeMo toolkit install needed for this variant —
        # that's a separate, heavier route AI4Bharat also offers, but this
        # transformers-native checkpoint avoids that entirely.
        #
        # This specific model is GATED on HuggingFace — unlike the two
        # Whisper models above, it requires an authenticated token even
        # though the repo is publicly visible. The token is read from
        # HF_TOKEN (set via .env or the environment) and passed explicitly
        # here rather than relying on a prior `hf auth login` having been
        # run on this machine.
        if not HF_TOKEN:
            logger.warning(
                "HF_TOKEN is not set — IndicConformer is a gated model and "
                "will fail to download without it. Set HF_TOKEN in a .env "
                "file (see .env.example) or run `hf auth login`."
            )

        logger.info(f"Loading IndicConformer model: '{HF_MODEL_ID_INDIC_CONFORMER}'")
        indic_model = AutoModel.from_pretrained(
            HF_MODEL_ID_INDIC_CONFORMER,
            trust_remote_code=True,
            cache_dir=MODEL_CACHE_DIR,
            token=HF_TOKEN,
        )
        indic_model.eval()
        logger.info("IndicConformer model loaded successfully")

        _components = {
            "malayalam": {"processor": ml_processor, "model": ml_model},
            "multilingual": {"processor": multi_processor, "model": multi_model},
            "indic_conformer": {"model": indic_model},  # no separate processor — model() takes raw waveform directly
        }

        logger.info("All three models loaded and ready")

    except ImportError:
        logger.error("transformers package not installed. Run: uv add transformers")
        raise
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        raise

    return _components


def validate_audio_file(audio_path: str) -> bool:
    """
    Validates the audio file exists and has a supported format.
    """
    logger.debug(f"Validating audio file: {audio_path}")

    if not os.path.exists(audio_path):
        logger.error(f"Audio file not found: {audio_path}")
        return False

    ext = os.path.splitext(audio_path)[1].lower()
    if ext not in SUPPORTED_AUDIO_FORMATS:
        logger.error(
            f"Unsupported format: '{ext}'. "
            f"Supported: {SUPPORTED_AUDIO_FORMATS}"
        )
        return False

    file_size_kb = os.path.getsize(audio_path) / 1024
    logger.debug(f"Audio valid — format: '{ext}', size: {file_size_kb:.1f} KB")
    return True


def trim_silence_vad(audio: "np.ndarray", sample_rate: int = SAMPLE_RATE) -> "np.ndarray":
    """
    Uses webrtcvad (Google's voice-activity-detection library, originally
    built for WebRTC) to find where actual speech starts and ends, and
    trims everything else.

    This is more reliable than amplitude-threshold trimming because VAD
    looks at the actual frequency characteristics of speech, not just
    loudness — so it correctly tells apart "quiet speech" from "silence
    with background hiss", which a simple volume threshold cannot.

    webrtcvad requires 16-bit PCM mono audio at 8k/16k/32k/48kHz,
    processed in 10/20/30ms frames — so audio is converted to that
    exact format first, VAD runs frame-by-frame, then the result is
    converted back to float32 for the rest of the pipeline.
    """
    import webrtcvad
    import numpy as np

    vad = webrtcvad.Vad(2)  # aggressiveness 0-3; 2 = moderately aggressive

    frame_ms = 30
    frame_len = int(sample_rate * frame_ms / 1000)

    # Convert float32 [-1, 1] audio to 16-bit PCM bytes, which is what
    # webrtcvad expects internally.
    pcm16 = (audio * 32767).astype(np.int16)

    speech_frames = []
    num_frames = len(pcm16) // frame_len

    for i in range(num_frames):
        frame = pcm16[i * frame_len: (i + 1) * frame_len]
        frame_bytes = frame.tobytes()

        try:
            is_speech = vad.is_speech(frame_bytes, sample_rate)
        except Exception:
            # Malformed frame (can happen on the very last partial frame) — skip it
            is_speech = False

        speech_frames.append(is_speech)

    if not any(speech_frames):
        logger.warning("VAD found no speech frames — returning audio unchanged")
        return audio

    first_speech = speech_frames.index(True)
    last_speech = len(speech_frames) - 1 - speech_frames[::-1].index(True)

    # Keep a small padding (~0.2s) around detected speech so words
    # right at the edge aren't accidentally clipped
    pad_frames = int(0.2 * 1000 / frame_ms)
    start_frame = max(0, first_speech - pad_frames)
    end_frame = min(num_frames, last_speech + 1 + pad_frames)

    start_sample = start_frame * frame_len
    end_sample = min(len(audio), end_frame * frame_len)

    trimmed = audio[start_sample:end_sample]

    logger.debug(
        f"VAD trimming — {len(audio) / sample_rate:.2f}s -> {len(trimmed) / sample_rate:.2f}s "
        f"({sum(speech_frames)}/{num_frames} frames had speech)"
    )

    return trimmed


def normalize_waveform(waveform: "torch.Tensor") -> "torch.Tensor":
    """
    Improves recorded-audio quality before transcription using
    webrtcvad (for silence/non-speech trimming) and librosa (for
    proper peak normalization).

    1. Trims leading/trailing silence using real voice-activity
       detection instead of a simple volume threshold — this correctly
       distinguishes quiet speech from background noise/hiss, which a
       naive amplitude cutoff cannot.
    2. Normalizes peak volume using librosa, leaving a small amount of
       headroom to avoid clipping.

    NOTE: A noisereduce-based noise-reduction step was tried here
    earlier but has been removed. It was found to alter the audio's
    frequency characteristics enough to cause language detection to
    misidentify Malayalam speech as other languages (e.g. Japanese),
    which broke transcription entirely. Silence trimming and
    normalization alone are kept since they don't carry that risk.

    Applied to every audio file (recorded or uploaded) before chunking
    and transcription.
    """
    import numpy as np
    import librosa

    audio = waveform.numpy().astype(np.float32)

    # ── Step 1: VAD-based silence trimming ───────────────────
    try:
        audio = trim_silence_vad(audio, SAMPLE_RATE)
    except Exception as e:
        logger.warning(f"VAD trimming failed, continuing without it: {e}")

    # ── Step 2: Peak normalization via librosa ───────────────
    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0
    if peak > 0:
        target_peak = 0.95  # librosa convention — normalize close to full scale
        audio = librosa.util.normalize(audio) * target_peak
        logger.debug(f"Volume normalized via librosa — original peak was {peak:.3f}")

    return torch.from_numpy(audio).float()


def load_and_preprocess_audio(audio_path: str):
    """
    Loads audio file, resamples to 16kHz, converts to mono,
    then normalizes volume and trims silence.

    Returns:
        1D torch tensor of audio samples at 16kHz, or None if failed
    """
    logger.info("Loading and preprocessing audio...")

    try:
        waveform, sample_rate = torchaudio.load(audio_path, backend="soundfile")
        logger.debug(f"Audio loaded — sample rate: {sample_rate}Hz, shape: {waveform.shape}")

        # Whisper requires 16000Hz sample rate
        if sample_rate != SAMPLE_RATE:
            logger.debug(f"Resampling from {sample_rate}Hz to {SAMPLE_RATE}Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Convert stereo to mono by averaging channels
        if waveform.shape[0] > 1:
            logger.debug("Converting stereo to mono")
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)

        # Normalize volume + trim silence — primarily helps recorded
        # (mic) audio, which tends to be quieter and have dead air
        # at the start/end compared to uploaded files.
        waveform = normalize_waveform(waveform)

        duration_sec = waveform.shape[0] / SAMPLE_RATE
        logger.info(f"Audio preprocessed — duration: {duration_sec:.1f}s")

        return waveform

    except Exception as e:
        logger.error(f"Audio preprocessing failed: {e}")
        return None


def detect_language(waveform_chunk, multi_processor, multi_model) -> str:
    """
    Detects the spoken language of a single audio chunk using the
    generic multilingual model's built-in language-detection method.

    Whisper computes a probability distribution over all languages it
    was trained on as part of its standard decoding setup.
    `model.detect_language()` runs just that detection step (no full
    transcription) and returns the single most likely language's
    special token id (e.g. the token for "<|ml|>").

    Args:
        waveform_chunk: 1D torch tensor, audio samples for this chunk only
        multi_processor: WhisperProcessor for the generic multilingual model
        multi_model: WhisperForConditionalGeneration for the generic model

    Returns:
        ISO language code string, e.g. "ml", "ta", "en", "hi"
    """
    inputs = multi_processor(
        waveform_chunk.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    with torch.no_grad():
        lang_token_ids = multi_model.detect_language(
            input_features=inputs["input_features"]
        )

    # Token looks like "<|ml|>" once decoded — strip the markers to get "ml"
    lang_token = multi_processor.tokenizer.decode(lang_token_ids[0])
    lang_code = lang_token.strip("<|>")

    return lang_code


def transcribe_chunk(waveform_chunk, processor, model, language_code: str) -> str:
    """
    Transcribes a single audio chunk (max 30 seconds) using the given
    model, forcing output into the specified language's script.

    Args:
        waveform_chunk: 1D torch tensor, audio samples for this chunk only
        processor: WhisperProcessor instance for the model being used
        model: WhisperForConditionalGeneration instance for the model being used
        language_code: ISO code like "ml", "ta", "en", "hi" — forces both
                        the spoken-language assumption and the output script

    Returns:
        Transcribed text for this chunk (string, may be empty on failure)
    """
    inputs = processor(
        waveform_chunk.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=language_code,
        task="transcribe"
    )

    with torch.no_grad():
        predicted_ids = model.generate(
            inputs["input_features"],
            forced_decoder_ids=forced_decoder_ids
        )

    chunk_text = processor.batch_decode(
        predicted_ids,
        skip_special_tokens=True
    )[0].strip()

    return chunk_text


def transcribe_chunk_indic_conformer(waveform_chunk, indic_model, language_code: str, decoding_mode: str) -> str:
    """
    Transcribes a single audio chunk using IndicConformer.

    Unlike the Whisper models, IndicConformer takes the raw waveform
    tensor directly (no separate processor/feature-extraction step) and
    is called as model(wav, language_code, decoding_mode) — matching
    AI4Bharat's documented usage for this checkpoint.

    Args:
        waveform_chunk: 1D torch tensor, audio samples for this chunk only
        indic_model: the loaded IndicConformer model
        language_code: ISO code like "ml", "ta", "hi" — IndicConformer
                        covers 22 Indian languages, see SUPPORTED_LANGUAGES
        decoding_mode: "ctc" (faster) or "rnnt" (generally more accurate) —
                        chosen by the user in the UI

    Returns:
        Transcribed text for this chunk (string, may be empty on failure)
    """
    # IndicConformer expects shape (1, num_samples) — batch dimension first
    wav = waveform_chunk.unsqueeze(0)

    with torch.no_grad():
        chunk_text = indic_model(wav, language_code, decoding_mode)

    # Some versions return a string directly, others a list — handle both
    if isinstance(chunk_text, list):
        chunk_text = chunk_text[0] if chunk_text else ""

    return str(chunk_text).strip()


def transcribe_audio(
    audio_path: str,
    model_choice: str = MODEL_WHISPER_MALAYALAM,
) -> str | None:
    """
    Transcribes an audio file of any length using a manually chosen
    "preferred Malayalam model", with automatic per-chunk language
    detection and fallback handling.

    For every 30-second chunk:
        1. Detect the spoken language using the hidden general
           multilingual Whisper model.
        2. If the detected language is Malayalam:
             - model_choice == MODEL_WHISPER_MALAYALAM  -> Thennal Malayalam Whisper
             - model_choice == MODEL_INDIC_CONFORMER    -> IndicConformer (RNNT)
        3. If the detected language is anything else (Tamil, English,
           Hindi, etc.) -> the general multilingual Whisper model
           transcribes it directly, regardless of model_choice.

    This means the two model_choice options only differ in which model
    handles Malayalam specifically; every other language is handled
    identically either way.

    Args:
        audio_path: Full path to the audio file (.wav, .mp3, etc.)
        model_choice: MODEL_WHISPER_MALAYALAM or MODEL_INDIC_CONFORMER —
                      which model should handle Malayalam audio.

    Returns:
        Full transcribed text, or None if failed
    """
    logger.info(f"Starting transcription: {os.path.basename(audio_path)}")
    logger.info(f"Malayalam handler selected: {model_choice}")

    # Step 1: Validate audio file
    if not validate_audio_file(audio_path):
        return None

    # Step 2: Load all models
    try:
        components = load_models()
        ml_processor = components["malayalam"]["processor"]
        ml_model = components["malayalam"]["model"]
        multi_processor = components["multilingual"]["processor"]
        multi_model = components["multilingual"]["model"]
        indic_model = components["indic_conformer"]["model"]
    except Exception:
        logger.error("Cannot transcribe — models failed to load")
        return None

    # Step 3: Load and preprocess full audio
    waveform = load_and_preprocess_audio(audio_path)
    if waveform is None:
        return None

    total_samples = waveform.shape[0]
    total_duration_sec = total_samples / SAMPLE_RATE

    # Step 4: Calculate number of 30-second chunks needed
    # (IndicConformer has no hard 30s limit, but chunking is kept
    # consistent across all models for comparable logging/behavior)
    num_chunks = (total_samples // SAMPLES_PER_CHUNK) + (
        1 if total_samples % SAMPLES_PER_CHUNK > 0 else 0
    )
    if num_chunks == 0:
        num_chunks = 1

    logger.info(
        f"Audio duration: {total_duration_sec:.1f}s — "
        f"splitting into {num_chunks} chunk(s) of {CHUNK_DURATION_SEC}s each"
    )

    # Step 5: Detect language + transcribe each chunk sequentially
    full_transcript_parts = []
    failed_chunks = 0
    language_log = []

    for i in range(num_chunks):
        start_sample = i * SAMPLES_PER_CHUNK
        end_sample = min(start_sample + SAMPLES_PER_CHUNK, total_samples)
        chunk = waveform[start_sample:end_sample]

        start_sec = start_sample / SAMPLE_RATE
        end_sec = end_sample / SAMPLE_RATE

        if chunk.shape[0] == 0:
            logger.debug(f"Chunk {i + 1}/{num_chunks} is empty — skipping")
            continue

        try:
            # ── Auto-detect language for this chunk ──────────
            lang_code = detect_language(chunk, multi_processor, multi_model)
            lang_name = SUPPORTED_LANGUAGES.get(lang_code, f"unsupported ({lang_code})")
            language_log.append(lang_code)

            # ── Route based on detected language + model_choice ──
            if lang_code == "ml" and model_choice == MODEL_WHISPER_MALAYALAM:
                logger.info(
                    f"Transcribing chunk {i + 1}/{num_chunks} "
                    f"[{start_sec:.1f}s - {end_sec:.1f}s] — "
                    f"detected: {lang_name} — using Thennal Malayalam Whisper ..."
                )
                chunk_text = transcribe_chunk(chunk, ml_processor, ml_model, "malayalam")

            elif lang_code == "ml" and model_choice == MODEL_INDIC_CONFORMER:
                logger.info(
                    f"Transcribing chunk {i + 1}/{num_chunks} "
                    f"[{start_sec:.1f}s - {end_sec:.1f}s] — "
                    f"detected: {lang_name} — using IndicConformer (RNNT) ..."
                )
                chunk_text = transcribe_chunk_indic_conformer(
                    chunk, indic_model, "ml", INDIC_CONFORMER_DECODING_MODE
                )

            else:
                # Any non-Malayalam language falls back to the general
                # multilingual Whisper model, regardless of model_choice.
                logger.info(
                    f"Transcribing chunk {i + 1}/{num_chunks} "
                    f"[{start_sec:.1f}s - {end_sec:.1f}s] — "
                    f"detected: {lang_name} — using general multilingual Whisper ..."
                )
                chunk_text = transcribe_chunk(chunk, multi_processor, multi_model, lang_code)

            full_transcript_parts.append(chunk_text)
            logger.info(f"Chunk {i + 1}/{num_chunks} done — '{chunk_text[:60]}...'")

        except Exception as e:
            failed_chunks += 1
            logger.error(f"Chunk {i + 1}/{num_chunks} failed: {e}")
            continue

    # Step 6: Join all chunks into final transcript
    transcribed_text = " ".join(full_transcript_parts).strip()

    # Step 7: Summary logging
    unique_languages = sorted(set(language_log))
    lang_summary = ", ".join(
        f"{SUPPORTED_LANGUAGES.get(code, code)} ({code})" for code in unique_languages
    )

    logger.info(
        f"Transcription complete — {num_chunks - failed_chunks}/{num_chunks} "
        f"chunks succeeded, {len(transcribed_text)} total characters"
    )
    logger.info(f"Language(s) detected across this file: {lang_summary or 'none'}")

    if failed_chunks > 0:
        logger.warning(f"{failed_chunks} chunk(s) failed during transcription")

    if not transcribed_text:
        logger.error("Final transcript is empty — all chunks failed")
        return None

    return transcribed_text