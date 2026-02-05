"""Audio playback (local) using sounddevice.

Public API
----------
    play_wav(file_path: str) -> None

Academic overview
-----------------
WAV is a simple container that often stores raw PCM samples.
Playback involves:
1) Decoding the WAV container (sample rate, sample width, channels)
2) Converting bytes to numeric samples (e.g., int16)
3) Sending samples to the audio device at the correct sample rate

This module keeps things minimal and local.
"""

from __future__ import annotations

import wave
from pathlib import Path


def play_wav(file_path: str) -> None:
    """Play a WAV file through the default output device."""

    wav_path = Path(file_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    try:
        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies. Install `sounddevice` and `numpy` for playback."
        ) from exc

    with wave.open(str(wav_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width != 2:
        raise ValueError(
            f"Only 16-bit PCM WAV is supported right now (got sample width {sample_width})."
        )

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)

    sd.play(audio, samplerate=sample_rate)
    sd.wait()
