"""Microphone audio recording (local) using sounddevice.

Public API
----------
    record_audio(output_path: str, duration: int) -> None

Audio capture (academic overview)
--------------------------------
Digital audio capture is a pipeline:

1) Microphone + ADC (Analog-to-Digital Conversion)
   - The microphone produces an analog voltage proportional to air pressure.
   - The ADC samples this voltage at a fixed rate (sample rate), producing
     discrete-time samples.

2) Sampling rate (here: 16 kHz)
   - A 16,000 Hz sampling rate means 16,000 samples per second.
   - By the Nyquist theorem, the highest representable frequency is 8 kHz,
     which is sufficient for speech intelligibility.

3) Quantization (here: 16-bit PCM)
   - We store audio as 16-bit signed integers (PCM), a common WAV format.
   - sounddevice records float samples (typically in [-1.0, 1.0]) which we
     scale/clip into int16 for WAV storage.

Implementation notes
--------------------
- Local-only: uses PortAudio via the `sounddevice` Python library.
- We write WAV via Python's standard `wave` module (no extra frameworks).
- This is a simple synchronous recorder suitable for a first prototype.
"""

from __future__ import annotations

from pathlib import Path


SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


def record_audio(output_path: str, duration: int) -> None:
    """Record microphone audio and write it to a WAV file.

    Parameters
    ----------
    output_path:
        WAV file path to create/overwrite.
    duration:
        Recording duration in seconds (integer).

    Output format
    -------------
    - WAV container
    - Mono channel
    - 16 kHz sample rate
    - 16-bit PCM
    """

    if duration <= 0:
        raise ValueError("duration must be a positive integer (seconds)")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies. Install `sounddevice` and `numpy` to use microphone recording."
        ) from exc

    frames = int(SAMPLE_RATE_HZ * duration)

    # Record returns a NumPy array with shape (frames, channels).
    # dtype=float32 gives normalized samples in roughly [-1.0, 1.0].
    try:
        recording = sd.rec(
            frames,
            samplerate=SAMPLE_RATE_HZ,
            channels=CHANNELS,
            dtype="float32",
            blocking=True,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Microphone recording failed. Ensure an input device is available and PortAudio is working."
        ) from exc

    # Convert float32 -> int16 PCM with clipping to avoid wrap-around distortion.
    pcm = np.clip(recording, -1.0, 1.0)
    pcm_int16 = (pcm * 32767.0).astype(np.int16)

    # Write WAV using standard library.
    import wave

    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE_HZ)

        # Ensure contiguous bytes in little-endian int16.
        wav_file.writeframes(pcm_int16.tobytes())
