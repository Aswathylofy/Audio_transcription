"""
transcriber.py
--------------
Handles Malayalam audio transcription using the HuggingFace model:
    thennal/whisper-medium-ml  (Malayalam fine-tuned Whisper)

Uses WhisperProcessor + WhisperForConditionalGeneration directly
to force Malayalam language and script output.

Long audio files are split into 30-second chunks (Whisper's processing
window limit) and transcribed sequentially, then joined together.

Model is downloaded on first run and cached locally in /models folder.
Subsequent runs load from cache — no internet needed after first run.
"""

import os
import torch
import torchaudio
from utils.logger import get_logger
from config.settings import HF_MODEL_ID, MODEL_CACHE_DIR, SUPPORTED_AUDIO_FORMATS

logger = get_logger(__name__)

# Global model components — loaded once, reused across calls
_components = None

# Whisper's fixed processing window — do not change.
# This is a model architecture limit, not a configurable setting.
CHUNK_DURATION_SEC = 30
SAMPLE_RATE = 16000
SAMPLES_PER_CHUNK = CHUNK_DURATION_SEC * SAMPLE_RATE


def load_model():
    """
    Loads WhisperProcessor and WhisperForConditionalGeneration
    from thennal/whisper-medium-ml.

    Downloads on first run (~1.5GB), then loads from local cache.
    Stored in _components dict — loaded only once per session.
    """
    global _components

    if _components is not None:
        logger.debug("Model already loaded, reusing cached instance")
        return _components

    logger.info(f"Loading model: '{HF_MODEL_ID}'")
    logger.info(f"Cache directory: {MODEL_CACHE_DIR}")
    logger.info("First run will download the model (~1.5GB) — please wait...")

    try:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        processor = WhisperProcessor.from_pretrained(
            HF_MODEL_ID,
            cache_dir=MODEL_CACHE_DIR
        )

        model = WhisperForConditionalGeneration.from_pretrained(
            HF_MODEL_ID,
            cache_dir=MODEL_CACHE_DIR
        )

        # Set to evaluation mode (disables dropout etc.)
        model.eval()

        _components = {
            "processor": processor,
            "model": model
        }

        logger.info(f"Model '{HF_MODEL_ID}' loaded successfully")

    except ImportError:
        logger.error("transformers package not installed. Run: uv add transformers")
        raise
    except Exception as e:
        logger.error(f"Failed to load model '{HF_MODEL_ID}': {e}")
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
    webrtcvad (for silence/non-speech trimming) and librosa
    (for proper peak normalization).

    1. Trims leading/trailing silence using real voice-activity
       detection instead of a simple volume threshold — this correctly
       distinguishes quiet speech from background noise/hiss, which a
       naive amplitude cutoff cannot.
    2. Normalizes peak volume using librosa, leaving a small amount of
       headroom to avoid clipping.

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


def transcribe_chunk(waveform_chunk, processor, model, forced_decoder_ids) -> str:
    """
    Transcribes a single audio chunk (max 30 seconds) using the model.

    Args:
        waveform_chunk: 1D torch tensor, audio samples for this chunk only
        processor: WhisperProcessor instance
        model: WhisperForConditionalGeneration instance
        forced_decoder_ids: pre-computed Malayalam language/task tokens

    Returns:
        Transcribed text for this chunk (string, may be empty on failure)
    """
    inputs = processor(
        waveform_chunk.numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
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


def transcribe_audio(audio_path: str) -> str | None:
    """
    Transcribes a Malayalam audio file of ANY length using thennal/whisper-medium-ml.

    Long audio is automatically split into 30-second chunks (Whisper's
    fixed processing window), each chunk transcribed separately, then
    all chunk results are joined into one final transcript.

    Forces Malayalam language and Malayalam script (not Devanagari)
    using forced_decoder_ids from WhisperProcessor.

    Args:
        audio_path: Full path to the audio file (.wav, .mp3, etc.)

    Returns:
        Full transcribed Malayalam text, or None if failed
    """
    logger.info(f"Starting transcription: {os.path.basename(audio_path)}")

    # Step 1: Validate audio file
    if not validate_audio_file(audio_path):
        return None

    # Step 2: Load model
    try:
        components = load_model()
        processor = components["processor"]
        model = components["model"]
    except Exception:
        logger.error("Cannot transcribe — model failed to load")
        return None

    # Step 3: Load and preprocess full audio
    waveform = load_and_preprocess_audio(audio_path)
    if waveform is None:
        return None

    total_samples = waveform.shape[0]
    total_duration_sec = total_samples / SAMPLE_RATE

    # Step 4: Calculate number of 30-second chunks needed
    num_chunks = (total_samples // SAMPLES_PER_CHUNK) + (
        1 if total_samples % SAMPLES_PER_CHUNK > 0 else 0
    )
    if num_chunks == 0:
        num_chunks = 1

    logger.info(
        f"Audio duration: {total_duration_sec:.1f}s — "
        f"splitting into {num_chunks} chunk(s) of {CHUNK_DURATION_SEC}s each"
    )

    # Step 5: Pre-compute Malayalam language/task tokens (same for all chunks)
    forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="malayalam",
        task="transcribe"
    )

    # Step 6: Transcribe each chunk sequentially
    full_transcript_parts = []
    failed_chunks = 0

    for i in range(num_chunks):
        start_sample = i * SAMPLES_PER_CHUNK
        end_sample = min(start_sample + SAMPLES_PER_CHUNK, total_samples)
        chunk = waveform[start_sample:end_sample]

        start_sec = start_sample / SAMPLE_RATE
        end_sec = end_sample / SAMPLE_RATE

        # Skip empty chunks (can happen on exact boundary)
        if chunk.shape[0] == 0:
            logger.debug(f"Chunk {i + 1}/{num_chunks} is empty — skipping")
            continue

        logger.info(
            f"Transcribing chunk {i + 1}/{num_chunks} "
            f"[{start_sec:.1f}s - {end_sec:.1f}s] ..."
        )

        try:
            chunk_text = transcribe_chunk(chunk, processor, model, forced_decoder_ids)
            full_transcript_parts.append(chunk_text)

            logger.info(f"Chunk {i + 1}/{num_chunks} done — '{chunk_text[:60]}...'")

        except Exception as e:
            failed_chunks += 1
            logger.error(f"Chunk {i + 1}/{num_chunks} failed: {e}")
            continue

    # Step 7: Join all chunks into final transcript
    transcribed_text = " ".join(full_transcript_parts).strip()

    logger.info(
        f"Transcription complete — {num_chunks - failed_chunks}/{num_chunks} "
        f"chunks succeeded, {len(transcribed_text)} total characters"
    )

    if failed_chunks > 0:
        logger.warning(f"{failed_chunks} chunk(s) failed during transcription")

    if not transcribed_text:
        logger.error("Final transcript is empty — all chunks failed")
        return None

    return transcribed_text