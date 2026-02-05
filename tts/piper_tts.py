"""Text-to-Speech (TTS) using Piper (local, CPU-compatible).

This module exposes a minimal TTS boundary:

    synthesize_speech(text: str, output_path: str) -> None

Neural TTS (academic overview)
------------------------------
Modern neural TTS pipelines typically have two conceptual stages:

1) Text / linguistic front-end
   - Normalizes text (numbers, dates, abbreviations)
   - Converts text into phonemes or phoneme-like units ("grapheme-to-phoneme")
   - Adds prosodic features (stress, pauses) implicitly or explicitly

2) Acoustic model + vocoder (waveform generation)
   - Acoustic model predicts a compact speech representation from phonemes
     (e.g., mel-spectrogram or other intermediate features)
   - Vocoder converts that representation into a time-domain waveform

Piper provides an efficient, local TTS runtime (ONNX-based) that bundles the
necessary components behind a simple API.

Implementation notes
--------------------
- Local-only: no external APIs.
- CPU-compatible: ONNX Runtime CPU execution is the default.
- Singleton-style voice loading: model load is expensive, so we cache it.

Configuration
-------------
Set environment variables to point to a Piper voice:
- VOICE_AGENT_PIPER_MODEL: path to `.onnx` model
- VOICE_AGENT_PIPER_CONFIG: optional path to `.onnx.json` config

Example (PowerShell):
    $env:VOICE_AGENT_PIPER_MODEL="C:\\voices\\en_US-lessac-medium.onnx"
    $env:VOICE_AGENT_PIPER_CONFIG="C:\\voices\\en_US-lessac-medium.onnx.json"
"""

from __future__ import annotations

import os
import subprocess
import wave
from functools import lru_cache
from pathlib import Path


def _env(name: str) -> str:
    """Read env vars at runtime.

    Why: if `.env` is loaded after module import, reading os.environ at import
    time would miss the values. This helper keeps behavior robust.
    """

    return os.environ.get(name, "")


_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(value: str) -> str:
    """Resolve a potentially-relative path safely.

    If the env var contains a relative path like `voices/voice.onnx`, resolve it
    relative to the project root so `uv run` works even when launched from a
    different working directory.
    """

    if not value:
        return ""
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((_PROJECT_ROOT / path).resolve())


def _infer_config_path(model_path: str) -> str:
    """Infer a default config path from a model path.

    Many Piper voices ship as:
      - `voice.onnx`
      - `voice.onnx.json`
    """

    if not model_path:
        return ""
    candidate = f"{model_path}.json" if not model_path.endswith(".json") else model_path
    # Prefer the common `.onnx.json` next to the model.
    onnx_json = f"{model_path}.json" if model_path.endswith(".onnx") else candidate
    return onnx_json


@lru_cache(maxsize=1)
def _get_voice():
    """Load and cache the Piper voice model (singleton-style).

    Academic note:
    Neural TTS model initialization involves loading ONNX weights and creating
    inference sessions/kernels. This is expensive, so we do it once.
    """

    model_path = _resolve_path(_env("VOICE_AGENT_PIPER_MODEL"))
    if not model_path:
        raise RuntimeError(
            "Missing Piper voice model path. Set VOICE_AGENT_PIPER_MODEL to a .onnx voice file."
        )

    config_path = _resolve_path(_env("VOICE_AGENT_PIPER_CONFIG")) or _infer_config_path(
        model_path
    )

    try:
        # Preferred import path for the `piper-tts` Python package.
        from piper.voice import PiperVoice  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Piper is not installed. Add `piper-tts` to your environment (uv) and retry."
        ) from exc

    model_file = Path(model_path)
    if not model_file.exists():
        raise FileNotFoundError(f"Piper model not found: {model_file}")

    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(
            f"Piper config not found: {config_file}. Set VOICE_AGENT_PIPER_CONFIG explicitly."
        )

    # Piper runs on CPU by default via ONNX Runtime.
    # If you later add CUDA, you can expose a flag here.
    return PiperVoice.load(str(model_file), config_path=str(config_file))


def synthesize_speech(text: str, output_path: str) -> None:
    """Synthesize `text` to a WAV file at `output_path`.

    Parameters
    ----------
    text:
        Input text to speak.
    output_path:
        Output WAV path to create/overwrite.

    Output format
    -------------
    - WAV container
    - 16-bit PCM
    - Mono
    - Sample rate defined by the Piper voice configuration

    Voice UX note:
    For voice assistants, keep outputs short and well-punctuated. Long run-on
    sentences can reduce intelligibility and make turn-taking harder.
    """

    clean_text = (text or "").strip()
    if not clean_text:
        raise ValueError("text must be non-empty")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Windows-friendly path: if a standalone Piper binary is configured, use it.
    # This avoids Python-package phonemizer issues (e.g., missing `espeakbridge`).
    if _env("VOICE_AGENT_PIPER_BIN"):
        _synthesize_with_piper_binary(clean_text, str(out_path))
        return

    voice = _get_voice()

    # Start synthesis *before* opening the WAV file.
    # Reason: if Piper fails before producing any chunk (common when the
    # phonemizer backend is missing), opening a wave writer and then returning
    # early causes wave.close() to raise "channels not specified".
    try:
        audio_stream = iter(voice.synthesize(clean_text))
        first_chunk = next(audio_stream, None)
    except ImportError as exc:  # pragma: no cover
        if _env("VOICE_AGENT_PIPER_BIN"):
            _synthesize_with_piper_binary(clean_text, str(out_path))
            return

        raise RuntimeError(
            "Piper phonemizer backend is missing (often `espeakbridge`). "
            "On Windows this can occur even on Python 3.12. "
            "Fix options: (1) use a standalone Piper release binary and set VOICE_AGENT_PIPER_BIN to its path, "
            "or (2) build/install Piper with phonemization support for Windows."
        ) from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Piper synthesis failed") from exc

    if first_chunk is None:
        raise RuntimeError(
            "Piper synthesis produced no audio chunks. "
            "Check that the voice model/config are correct and that the phonemizer backend is installed."
        )

    # Important API note:
    # `PiperVoice.synthesize(text)` RETURNS AudioChunk objects.
    # We must write `audio_int16_bytes` to a WAV container.
    with wave.open(str(out_path), "wb") as wav_file:
        # Set WAV params from the first produced chunk.
        wav_file.setframerate(int(getattr(first_chunk, "sample_rate", 22_050)))
        wav_file.setsampwidth(int(getattr(first_chunk, "sample_width", 2)))
        wav_file.setnchannels(int(getattr(first_chunk, "sample_channels", 1)))

        total_audio_bytes = 0
        # Write first chunk, then the rest of the stream.
        for chunk in (first_chunk,):
            audio_bytes = getattr(chunk, "audio_int16_bytes", None) or getattr(
                chunk, "audio", b""
            )
            if audio_bytes:
                wav_file.writeframes(audio_bytes)
                total_audio_bytes += len(audio_bytes)

        for chunk in audio_stream:
            audio_bytes = getattr(chunk, "audio_int16_bytes", None) or getattr(
                chunk, "audio", b""
            )
            if audio_bytes:
                wav_file.writeframes(audio_bytes)
                total_audio_bytes += len(audio_bytes)

        if total_audio_bytes == 0:
            raise RuntimeError(
                "Piper synthesis produced no audio samples (0-byte audio). "
                "Check that the voice model/config are correct and that the phonemizer backend is installed."
            )


def _synthesize_with_piper_binary(text: str, output_path: str) -> None:
    """Fallback: run Piper as an external executable.

    This is useful on Windows when the Python `piper-tts` package lacks
    the phonemizer backend (e.g., missing `espeakbridge`).

    Requires:
    - VOICE_AGENT_PIPER_BIN: path to `piper.exe` from a standalone release
    - VOICE_AGENT_PIPER_MODEL / VOICE_AGENT_PIPER_CONFIG
    """

    piper_bin = _resolve_path(_env("VOICE_AGENT_PIPER_BIN"))
    if not piper_bin:
        raise RuntimeError("VOICE_AGENT_PIPER_BIN is not set")

    model_path = _resolve_path(_env("VOICE_AGENT_PIPER_MODEL"))
    if not model_path:
        raise RuntimeError(
            "Missing VOICE_AGENT_PIPER_MODEL. Point it to your voice .onnx file (absolute path recommended)."
        )

    config_path = _resolve_path(_env("VOICE_AGENT_PIPER_CONFIG")) or _infer_config_path(
        model_path
    )

    # If the inferred config doesn't exist, give a clearer hint: many users
    # have voice files with Windows '(1)' suffixes or slightly different names.
    if config_path and not Path(config_path).exists():
        raise FileNotFoundError(
            f"Piper config not found: {config_path}. "
            "Either rename your config to '<model>.json' (e.g., 'voice.onnx.json') "
            "or set VOICE_AGENT_PIPER_CONFIG explicitly to the existing .json file."
        )

    # Best-effort espeak data discovery.
    # Piper binary can require an explicit `--espeak_data` when invoked from
    # outside its own folder.
    espeak_data_dir = _resolve_path(_env("VOICE_AGENT_PIPER_ESPEAK_DATA"))
    if not espeak_data_dir:
        maybe = Path(piper_bin).resolve().parent / "espeak-ng-data"
        if maybe.exists():
            espeak_data_dir = str(maybe)

    cmd = [
        piper_bin,
        "-m",
        model_path,
        "-f",
        output_path,
    ]

    if config_path:
        cmd.extend(["-c", config_path])

    if espeak_data_dir:
        cmd.extend(["--espeak_data", espeak_data_dir])

    try:
        subprocess.run(
            cmd,
            input=(text + "\n").encode("utf-8"),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Piper binary not found: {piper_bin}") from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"Piper binary failed: {stderr.strip()}") from exc
