"""Speech-to-Text (STT) using faster-whisper (local, CPU-optimized).

This module provides a minimal, *academic-style* STT pipeline wrapper around
`faster-whisper`.

Why faster-whisper?
- It uses CTranslate2 for efficient Whisper inference.
- It supports CPU quantization (e.g., int8) for faster local execution.
- It can ingest common audio formats via ffmpeg decoding.

STT pipeline (conceptual overview)
1) Audio decoding: load the audio file (wav/mp3/flac/...) into a waveform.
2) Resampling/normalization: convert to the sampling rate expected by Whisper.
3) Feature extraction: compute log-Mel spectrogram features.
4) Acoustic-linguistic inference: run the Whisper encoder/decoder to predict tokens.
5) Decoding: convert token probabilities into text (greedy/beam search).
6) Post-processing: stitch segments, normalize whitespace/punctuation.

Note:
- faster-whisper performs steps (1)-(5) internally when you pass a file path.
- This module focuses on stable boundaries and a single public function.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


def _env(name: str, default: str) -> str:
    """Read environment variables at runtime.

    Why: `.env` may be loaded after module import depending on the entrypoint.
    Reading env vars at runtime avoids "frozen" defaults.
    """

    return os.environ.get(name, default)


# CPU-friendly defaults.
# - `int8` quantization reduces memory bandwidth and can be significantly faster
#   on typical CPUs for Whisper inference.
# - Beam size 1 == greedy decoding (fastest). You can increase later for quality.
_DEFAULT_DEVICE = "cpu"


@lru_cache(maxsize=1)
def _get_model():
    """Load and cache the Whisper model (singleton-style).

    Academic note:
    Model initialization includes loading weights and creating runtime kernels.
    This is an expensive operation, so we do it once and reuse it across calls.
    """

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "faster-whisper is not installed. Add it to your environment (uv) and retry."
        ) from exc

    cpu_threads = os.cpu_count() or 4

    model_name = _env("VOICE_AGENT_WHISPER_MODEL", "small")
    compute_type = _env("VOICE_AGENT_WHISPER_COMPUTE_TYPE", "int8")

    # `num_workers` controls internal dataloader/decoder workers.
    # Keep modest by default; adjust later based on profiling.
    return WhisperModel(
        model_name,
        device=_DEFAULT_DEVICE,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
    )


def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file into text.

    Parameters
    ----------
    file_path:
        Path to an audio file (e.g., wav/mp3/flac). faster-whisper uses ffmpeg
        to decode most formats; ensure ffmpeg is available on PATH.

    Returns
    -------
    str
        The concatenated transcription text.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist.
    RuntimeError
        If the model cannot load or transcription fails.

    Academic note:
    Whisper is a sequence-to-sequence model that emits text tokens conditioned on
    acoustic features. The output is typically produced in time-stamped segments
    that we then concatenate to form a single transcript.
    """

    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model()

    try:
        beam_size = int(_env("VOICE_AGENT_WHISPER_BEAM_SIZE", "1"))
    except ValueError:
        beam_size = 1

    language = _env("VOICE_AGENT_WHISPER_LANGUAGE", "").strip() or None
    task = _env("VOICE_AGENT_WHISPER_TASK", "transcribe").strip() or "transcribe"

    try:
        segments, _info = model.transcribe(
            str(audio_path),
            beam_size=beam_size,
            vad_filter=True,
            language=language,
            task=task,
        )
    except Exception as exc:  # pragma: no cover
        # Common causes: missing ffmpeg, unsupported codec, invalid file.
        raise RuntimeError(
            "Transcription failed. Ensure the audio file is valid and ffmpeg is installed/available."
        ) from exc

    # Concatenate segments into a single string.
    # Segmentation is produced by the model's decoding process (often aligned to
    # pauses/silence via VAD and internal heuristics).
    text_parts: list[str] = []
    for segment in segments:
        segment_text = (segment.text or "").strip()
        if segment_text:
            text_parts.append(segment_text)

    return " ".join(text_parts).strip()
