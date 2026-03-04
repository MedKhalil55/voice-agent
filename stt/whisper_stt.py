"""Speech-to-Text (STT) using faster-whisper (local, GPU-accelerated with fallback).

This module provides a minimal, *academic-style* STT pipeline wrapper around
`faster-whisper`.

Why faster-whisper?
- It uses CTranslate2 for efficient Whisper inference.
- It supports CUDA acceleration on NVIDIA GPUs for real-time transcription.
- It supports CPU quantization (e.g., int8) as a robust fallback for local execution.
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
import re
from functools import lru_cache
from pathlib import Path


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _load_wav_16k_mono_float32(audio_path: Path):
    """Load a PCM16 WAV (16 kHz) into a mono float32 NumPy array.

    If the file isn't PCM16/16kHz, return None and let the caller fall back to
    passing the path to faster-whisper (ffmpeg decode).
    """

    try:
        import numpy as np  # type: ignore
        import wave
    except Exception:
        return None

    try:
        with wave.open(str(audio_path), "rb") as wav_file:
            framerate = int(wav_file.getframerate())
            channels = int(wav_file.getnchannels())
            sampwidth = int(wav_file.getsampwidth())
            frames = int(wav_file.getnframes())

            if framerate != 16_000 or sampwidth != 2 or frames <= 0:
                return None

            raw = wav_file.readframes(frames)
    except Exception:
        return None

    pcm = np.frombuffer(raw, dtype=np.int16)
    if pcm.size == 0:
        return None

    if channels > 1:
        pcm = pcm.reshape(-1, channels)[:, 0]

    audio = pcm.astype(np.float32) / 32768.0
    return audio


# Runtime override used to force CPU after a CUDA failure (e.g., missing DLLs).
_RUNTIME_FORCE_DEVICE: str | None = None


def _env(name: str, default: str) -> str:
    """Read environment variables at runtime.

    Why: `.env` may be loaded after module import depending on the entrypoint.
    Reading env vars at runtime avoids "frozen" defaults.
    """

    return os.environ.get(name, default)


def _cuda_is_available() -> bool:
    """Return True if CTranslate2 reports at least one CUDA device.

    Academic note:
    faster-whisper uses CTranslate2 under the hood. The most reliable way to
    detect CUDA availability is to ask CTranslate2 directly.
    """

    try:
        import ctranslate2  # type: ignore

        get_count = getattr(ctranslate2, "get_cuda_device_count", None)
        if callable(get_count):
            return int(get_count()) > 0
    except Exception:
        return False

    return False


def _select_device_and_compute_type() -> tuple[str, str]:
    """Select an execution device and compute type.

    Defaults (production-ready banking transcription):
    - Prefer GPU: device=cuda, compute_type=float16 (fast + accurate on NVIDIA)
    - Fallback: device=cpu, compute_type=int8 (robust on any machine)

    Env overrides:
    - VOICE_AGENT_WHISPER_DEVICE: 'cuda' or 'cpu'
    - VOICE_AGENT_WHISPER_COMPUTE_TYPE: ctranslate2 compute type
    """

    requested_device = os.environ.get("VOICE_AGENT_WHISPER_DEVICE", "").strip().lower()
    requested_compute = os.environ.get("VOICE_AGENT_WHISPER_COMPUTE_TYPE", "").strip()

    cuda_ok = _cuda_is_available()

    # Device selection (prefer CUDA), with a runtime safety override.
    forced = (_RUNTIME_FORCE_DEVICE or "").strip().lower()
    if forced in {"cuda", "cpu"}:
        device = forced
    elif requested_device in {"cuda", "cpu"}:
        device = requested_device
    else:
        device = "cuda" if cuda_ok else "cpu"

    # Requirement: if CUDA is not available, fall back automatically.
    if device == "cuda" and not cuda_ok:
        device = "cpu"

    # Compute type defaults.
    default_compute = "float16" if device == "cuda" else "int8"
    compute_type = requested_compute or default_compute

    # Make CPU fallback robust even if the user left a GPU-oriented compute type.
    if device == "cpu" and compute_type.lower() in {"float16", "int8_float16"}:
        compute_type = "int8"

    return device, compute_type


def _looks_like_cuda_runtime_error(exc: Exception) -> bool:
    """Heuristically detect missing CUDA runtime dependencies.

    On Windows, this often manifests as missing DLLs like cublas64_12.dll.
    """

    message = str(exc).lower()
    needles = (
        "cublas",
        "cudnn",
        "cudart",
        "cuda",
        "dll is not found",
        "cannot be loaded",
    )
    return any(n in message for n in needles)


def _parse_int_env(name: str, default: int) -> int:
    raw = _env(name, str(default)).strip()
    try:
        return int(raw)
    except ValueError:
        return default


def _parse_float_env(name: str, default: float) -> float:
    raw = _env(name, str(default)).strip()
    try:
        return float(raw)
    except ValueError:
        return default


def _postprocess_french_banking_text(text: str) -> str:
    """Normalize and correct common French banking transcription artifacts.

    Design goals:
    - Fix common lexical errors on banking vocabulary.
    - Normalize whitespace/punctuation for downstream NLU/LLM.
    - Preserve numeric and currency expressions.
    """

    value = (text or "").strip()
    if not value:
        return ""

    # Normalize whitespace first.
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = re.sub(r"\s+", " ", value).strip()

    # Common ASR vocabulary corrections (French banking domain).
    # Keep replacements conservative to avoid altering unrelated phrasing.
    corrections: list[tuple[str, str]] = [
        (r"\bcarte\s+bleu\b", "carte bleue"),
        (r"\bviremant\b", "virement"),
        (r"\bvirment\b", "virement"),
        (r"\béchéansier\b", "échéancier"),
        (r"\becheansier\b", "échéancier"),
        (r"\bmensualitee\b", "mensualité"),
        (r"\bmensualitee\b", "mensualité"),
        (r"\bprelevement\b", "prélèvement"),
        (r"\bprelevements\b", "prélèvements"),
        (r"\bprélèvement\s+auto\b", "prélèvement automatique"),
        (r"\binteret\b", "intérêt"),
        (r"\bagios\b", "agios"),
        (r"\bdecuvert\b", "découvert"),
        (r"\bdecouvert\b", "découvert"),
        (r"\bsepa\b", "SEPA"),
        (r"\bi\s*ban\b", "IBAN"),
        (r"\bb\s*i\s*c\b", "BIC"),
        (r"\br\s*i\s*b\b", "RIB"),
    ]
    for pattern, replacement in corrections:
        value = re.sub(pattern, replacement, value, flags=re.IGNORECASE)

    # Preserve/normalize currency spacing without altering numeric content.
    value = re.sub(r"(\d)\s*€", r"\1 €", value)
    value = re.sub(r"€\s*(\d)", r"€ \1", value)
    value = re.sub(r"\b(euro|euros)\b", "euros", value, flags=re.IGNORECASE)

    # Normalize punctuation spacing carefully (do not break numeric formats).
    value = re.sub(r"\s+([;:!?])", r"\1", value)
    value = re.sub(r"([;:!?])(\S)", r"\1 \2", value)
    value = re.sub(r"\s{2,}", " ", value).strip()

    return value


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

    # Why GPU?
    # - CUDA provides higher throughput and lower latency for real-time voice agents.
    # - float16 is typically the best accuracy/speed trade-off on NVIDIA GPUs.
    # - We keep an automatic CPU int8 fallback to stay production-robust.
    device, compute_type = _select_device_and_compute_type()

    # Why 'medium'?
    # - 'medium' is a strong quality baseline for French in noisy call audio.
    # - It is lighter than very large models, improving latency for voice UX.
    model_name = _env("VOICE_AGENT_WHISPER_MODEL", "medium")

    log_device = _env("VOICE_AGENT_STT_LOG_DEVICE", "true").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }
    if log_device:
        print(
            f"[stt] faster-whisper: model={model_name}, device={device}, compute_type={compute_type}"
        )

    # `num_workers` controls internal dataloader/decoder workers.
    # Keep modest by default; adjust later based on profiling.
    try:
        return WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=1,
        )
    except Exception as exc:
        # If CUDA path fails at runtime (missing drivers, incompatible build,
        # etc.), fallback to CPU int8 automatically.
        if device == "cuda":
            if log_device:
                print("[stt] CUDA init failed; falling back to CPU (int8).")
            return WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=cpu_threads,
                num_workers=1,
            )
        raise exc


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

    global _RUNTIME_FORCE_DEVICE

    audio_path = Path(file_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_model()

    # Why beam search?
    # - beam_size=5 + best_of=5 improves decoding stability and accuracy,
    #   especially on short banking utterances (names, amounts, payment terms).
    # - temperature=0.0 encourages deterministic decoding, which is desirable
    #   for production voice pipelines.
    beam_size = _parse_int_env("VOICE_AGENT_WHISPER_BEAM_SIZE", 3)
    best_of = _parse_int_env("VOICE_AGENT_WHISPER_BEST_OF", 1)
    temperature = _parse_float_env("VOICE_AGENT_WHISPER_TEMPERATURE", 0.0)

    condition_on_previous_text = _env(
        "VOICE_AGENT_WHISPER_CONDITION_ON_PREVIOUS_TEXT", "true"
    ).strip().lower() not in {"0", "false", "no", "off"}

    # Default to French for stability in a France banking call context.
    lang_raw = _env("VOICE_AGENT_WHISPER_LANGUAGE", "fr").strip()
    language = None if lang_raw.lower() in {"", "auto", "none"} else lang_raw
    task = _env("VOICE_AGENT_WHISPER_TASK", "transcribe").strip() or "transcribe"

    # Performance toggles (env-configurable).
    vad_filter = _parse_bool_env("VOICE_AGENT_WHISPER_VAD_FILTER", True)
    without_timestamps = _parse_bool_env("VOICE_AGENT_WHISPER_WITHOUT_TIMESTAMPS", True)
    use_numpy_wav = _parse_bool_env("VOICE_AGENT_STT_USE_NUMPY_WAV", True)

    audio_input = str(audio_path)
    if use_numpy_wav and audio_path.suffix.lower() == ".wav":
        in_mem = _load_wav_16k_mono_float32(audio_path)
        if in_mem is not None:
            audio_input = in_mem

    def _transcribe_once(active_model):
        segments, _info = active_model.transcribe(
            audio_input,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            condition_on_previous_text=condition_on_previous_text,
            vad_filter=vad_filter,
            language=language,
            task=task,
            without_timestamps=without_timestamps,
        )
        # Important: `segments` is a lazy generator. Runtime CUDA/DLL errors can
        # occur during iteration, so force evaluation inside the try/except.
        return list(segments)

    try:
        segments_list = _transcribe_once(model)
    except Exception as exc:
        # Production robustness: if CUDA is selected but runtime DLLs are missing
        # (common on Windows), fall back to CPU int8 and retry once.
        if (
            _RUNTIME_FORCE_DEVICE or ""
        ).strip().lower() != "cpu" and _looks_like_cuda_runtime_error(exc):
            log_device = _env(
                "VOICE_AGENT_STT_LOG_DEVICE", "true"
            ).strip().lower() not in {
                "0",
                "false",
                "no",
                "off",
            }
            if log_device:
                print(
                    "[stt] CUDA runtime error detected (often missing DLLs). Retrying on CPU (int8)."
                )
            _RUNTIME_FORCE_DEVICE = "cpu"
            _get_model.cache_clear()
            model = _get_model()
            try:
                segments_list = _transcribe_once(model)
            except Exception as exc2:
                raise RuntimeError(
                    "Transcription failed after CUDA fallback. Ensure the audio file is valid and ffmpeg is installed/available."
                ) from exc2
        else:
            raise RuntimeError(
                "Transcription failed. Ensure the audio file is valid and ffmpeg is installed/available."
            ) from exc

    text_parts: list[str] = []
    for segment in segments_list:
        segment_text = (segment.text or "").strip()
        if segment_text:
            text_parts.append(segment_text)

    raw_text = " ".join(text_parts).strip()
    return _postprocess_french_banking_text(raw_text)


def warmup_stt() -> None:
    """Warm up the STT model.

    Purpose:
    - Reduce first-turn latency by forcing model initialization early.
    - Useful to run in a background thread while TTS greeting plays.

    This does not run a transcription; it only loads the model into memory.
    """

    _get_model()
