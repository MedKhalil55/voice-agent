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


# ---------------------------------------------------------------------------
# Streaming TTS playback (no intermediate WAV files)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _get_piper_sample_rate() -> int:
    """Read and cache the audio sample rate from the Piper voice config JSON."""
    import json as _json

    config_path = _resolve_path(_env("VOICE_AGENT_PIPER_CONFIG")) or _infer_config_path(
        _resolve_path(_env("VOICE_AGENT_PIPER_MODEL"))
    )
    if config_path and Path(config_path).exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            config = _json.load(fh)
        return int(config.get("audio", {}).get("sample_rate", 22050))
    return 22050


def warmup_tts() -> None:
    """Pre-validate Piper paths and cache the voice config at startup.

    For the binary path this validates file existence and caches the sample
    rate.  For the Python piper-tts path it also loads the ONNX model.
    """
    _get_piper_sample_rate()
    if not _env("VOICE_AGENT_PIPER_BIN"):
        _get_voice()


def speak_streaming(text: str) -> None:
    """Synthesize *text* and play it immediately — no WAV files.

    Audio chunks are streamed to ``sounddevice.OutputStream`` as Piper
    generates them, so playback starts as soon as the first chunk is ready.
    """
    clean_text = (text or "").strip()
    if not clean_text:
        return

    if _env("VOICE_AGENT_PIPER_BIN"):
        _speak_streaming_binary(clean_text)
    else:
        _speak_streaming_python(clean_text)


def _speak_streaming_binary(text: str) -> None:
    """Stream TTS via the standalone Piper binary (``--output-raw``).

    Piper writes raw 16-bit signed-LE mono PCM to *stdout*.  We read it in
    small chunks and feed each one directly to a ``sounddevice.OutputStream``
    so playback begins while Piper is still synthesising.
    """
    import time

    import numpy as np
    import sounddevice as sd

    piper_bin = _resolve_path(_env("VOICE_AGENT_PIPER_BIN"))
    if not piper_bin:
        raise RuntimeError("VOICE_AGENT_PIPER_BIN is not set")
    model_path = _resolve_path(_env("VOICE_AGENT_PIPER_MODEL"))
    if not model_path:
        raise RuntimeError("VOICE_AGENT_PIPER_MODEL is not set")

    config_path = _resolve_path(_env("VOICE_AGENT_PIPER_CONFIG")) or _infer_config_path(
        model_path
    )
    sample_rate = _get_piper_sample_rate()

    espeak_data_dir = _resolve_path(_env("VOICE_AGENT_PIPER_ESPEAK_DATA"))
    if not espeak_data_dir:
        maybe = Path(piper_bin).resolve().parent / "espeak-ng-data"
        if maybe.exists():
            espeak_data_dir = str(maybe)

    cmd = [piper_bin, "-m", model_path, "--output-raw"]
    if config_path:
        cmd.extend(["-c", config_path])
    if espeak_data_dir:
        cmd.extend(["--espeak_data", espeak_data_dir])

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    proc.stdin.write((text + "\n").encode("utf-8"))
    proc.stdin.close()

    # 1024 samples × 2 bytes/sample = 2048 bytes per read.
    CHUNK_BYTES = 2048

    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16")
    stream.start()
    wrote_any = False
    try:
        while True:
            data = proc.stdout.read(CHUNK_BYTES)
            if not data:
                break
            # Guarantee an even byte-count for int16 framing.
            if len(data) % 2 != 0:
                data = data[:-1]
            if not data:
                continue
            audio = np.frombuffer(data, dtype=np.int16)
            stream.write(audio.reshape(-1, 1))
            wrote_any = True
        # Let the output ring-buffer drain before closing the stream.
        if wrote_any:
            time.sleep(0.2)
    finally:
        stream.stop()
        stream.close()

    rc = proc.wait()
    if rc != 0 and not wrote_any:
        stderr_text = (proc.stderr.read() or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"Piper binary failed (exit {rc}): {stderr_text.strip()}")


def _speak_streaming_python(text: str) -> None:
    """Stream TTS via the Python ``piper-tts`` package (fallback)."""
    import time

    import numpy as np
    import sounddevice as sd

    voice = _get_voice()
    sample_rate = _get_piper_sample_rate()

    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16")
    stream.start()
    wrote_any = False
    try:
        for chunk in voice.synthesize(text):
            audio_bytes = getattr(chunk, "audio_int16_bytes", None) or getattr(
                chunk, "audio", b""
            )
            if audio_bytes:
                audio = np.frombuffer(audio_bytes, dtype=np.int16)
                stream.write(audio.reshape(-1, 1))
                wrote_any = True
        if wrote_any:
            time.sleep(0.2)
    finally:
        stream.stop()
        stream.close()
