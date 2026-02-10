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

import os
import threading
from pathlib import Path


SAMPLE_RATE_HZ = 16_000
CHANNELS = 1
SAMPLE_WIDTH_BYTES = 2  # 16-bit PCM


def _env(name: str) -> str:
    return os.environ.get(name, "")


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

    stop_on_silence_s = _env("VOICE_AGENT_RECORD_STOP_ON_SILENCE_SECONDS").strip()

    # If configured, record until the user stops speaking (silence) or until the
    # max duration is reached. This makes turns feel more natural than a fixed
    # 5-second window.
    if stop_on_silence_s:
        try:
            silence_seconds = max(float(stop_on_silence_s), 0.0)
        except ValueError:
            silence_seconds = 0.0

        _record_until_silence(
            output_path=str(out_path),
            max_duration_s=float(duration),
            silence_duration_s=silence_seconds,
        )
        return

    frames = int(SAMPLE_RATE_HZ * duration)

    # Optional: pick a specific input device.
    # Useful on Windows when the default device is not your microphone.
    # Accepts either an integer index or a device name string.
    input_device: int | str | None
    raw_device = _env("VOICE_AGENT_AUDIO_INPUT_DEVICE").strip()
    if raw_device:
        try:
            input_device = int(raw_device)
        except ValueError:
            input_device = raw_device
    else:
        input_device = None

    # Record returns a NumPy array with shape (frames, channels).
    # dtype=float32 gives normalized samples in roughly [-1.0, 1.0].
    try:
        recording = sd.rec(
            frames,
            samplerate=SAMPLE_RATE_HZ,
            channels=CHANNELS,
            dtype="float32",
            device=input_device,
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


def _record_until_silence(
    output_path: str,
    max_duration_s: float,
    silence_duration_s: float,
) -> None:
    """Record audio until silence is detected or max duration is reached.

    Silence detection is energy-based (RMS threshold). It is a lightweight
    alternative to a full VAD model and works well for a prototype.
    """

    if max_duration_s <= 0:
        raise ValueError("max_duration_s must be positive")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import numpy as np  # type: ignore
        import sounddevice as sd  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependencies. Install `sounddevice` and `numpy` to use microphone recording."
        ) from exc

    # Optional device selection (same as fixed recorder).
    input_device: int | str | None
    raw_device = _env("VOICE_AGENT_AUDIO_INPUT_DEVICE").strip()
    if raw_device:
        try:
            input_device = int(raw_device)
        except ValueError:
            input_device = raw_device
    else:
        input_device = None

    # Tuning knobs (env overrides).
    # RMS threshold: ~0.01 corresponds to about -40 dBFS for normalized float32.
    # If your mic is quiet, reduce this (e.g., 0.005). If you get false triggers,
    # increase it (e.g., 0.02).
    try:
        silence_rms = float(_env("VOICE_AGENT_SILENCE_RMS_THRESHOLD") or "0.01")
    except ValueError:
        silence_rms = 0.01

    try:
        min_record_s = float(_env("VOICE_AGENT_MIN_RECORD_SECONDS") or "0.6")
    except ValueError:
        min_record_s = 0.6

    blocksize = 1024
    max_samples = int(SAMPLE_RATE_HZ * max_duration_s)
    silence_samples_needed = int(SAMPLE_RATE_HZ * max(0.0, silence_duration_s))
    min_samples_needed = int(SAMPLE_RATE_HZ * max(0.0, min_record_s))

    chunks: list[np.ndarray] = []
    total_samples = 0
    silent_samples = 0

    stop_reason = ""

    stop_event = threading.Event()

    def callback(indata, frames, time_info, status):  # type: ignore[no-untyped-def]
        nonlocal total_samples, silent_samples, stop_reason

        if status:
            # Non-fatal; keep recording.
            pass

        # Copy so PortAudio buffer can be reused.
        chunk = indata.copy()
        chunks.append(chunk)
        total_samples += int(frames)

        # RMS energy of this block.
        rms = float(np.sqrt(np.mean(np.square(chunk)))) if frames else 0.0

        if rms < silence_rms:
            silent_samples += int(frames)
        else:
            silent_samples = 0

        reached_max = total_samples >= max_samples
        reached_silence = (
            silence_samples_needed > 0
            and total_samples >= min_samples_needed
            and silent_samples >= silence_samples_needed
        )

        if reached_max or reached_silence:
            if reached_silence:
                stop_reason = "silence"
            elif reached_max:
                stop_reason = "max_duration"
            stop_event.set()
            raise sd.CallbackStop()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE_HZ,
            channels=CHANNELS,
            dtype="float32",
            blocksize=blocksize,
            device=input_device,
            callback=callback,
        ):
            while not stop_event.is_set():
                sd.sleep(50)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Microphone recording failed during streaming. Check your input device and permissions."
        ) from exc

    if not chunks:
        raise RuntimeError("No audio captured")

    if not stop_reason:
        stop_reason = "unknown"

    recorded_s = total_samples / float(SAMPLE_RATE_HZ)
    print(f"[recorder] stopped ({stop_reason}) after {recorded_s:.1f}s")

    recording = np.concatenate(chunks, axis=0)
    pcm = np.clip(recording, -1.0, 1.0)
    pcm_int16 = (pcm * 32767.0).astype(np.int16)

    import wave

    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(CHANNELS)
        wav_file.setsampwidth(SAMPLE_WIDTH_BYTES)
        wav_file.setframerate(SAMPLE_RATE_HZ)
        wav_file.writeframes(pcm_int16.tobytes())
