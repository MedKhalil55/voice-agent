"""Streaming microphone transcription using faster-whisper (single final decode).

Important reality check
----------------------
Whisper / faster-whisper is not a token-by-token streaming ASR model. The model
expects an audio window and returns text segments after decoding.

This module implements *production-friendly pseudo-streaming* with **no partial
decoding**:
- Capture microphone audio in small chunks (default: 0.5s)
- Maintain a rolling float32 buffer in memory (no temp WAVs)
- Use lightweight VAD (RMS energy) to detect speech start/end
- When speech ends, run **one** final decode and emit a final transcript

Why this design?
---------------
Many telephone assistants do *turn-based* ASR: detect end-of-utterance, then do
one decode. This avoids heavy GPU work while the user is speaking and typically
reduces compute usage and stabilizes latency.

Public API
----------
    class StreamingWhisper
        - __init__()
        - start_stream()
        - process_chunk()
        - stop_stream()

Dependencies
------------
- numpy
- sounddevice
- faster-whisper

No temporary files are written.
"""

from __future__ import annotations

import os
import queue
import threading
from dataclasses import dataclass
from typing import Callable, Optional


def _parse_bool_env(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _parse_hotwords_env(name: str) -> list[str] | None:
    """Parse comma-separated hotwords from env.

    Returns None if not provided.
    """

    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    parts = [p for p in parts if p]
    return parts or None


@dataclass(frozen=True)
class VadConfig:
    """Lightweight VAD configuration.

    This is not a neural VAD. It is energy-based and intentionally cheap.
    We still pass `vad_filter=True` to faster-whisper when decoding to apply its
    internal Silero-VAD filter as an additional robustness layer.
    """

    rms_start: float = 0.012
    rms_end: float = 0.010
    silence_seconds_to_end: float = 0.8
    min_utterance_seconds: float = 0.3


class _RollingRingBuffer:
    """A rolling ring buffer backed by a 1D float32 NumPy array."""

    def __init__(self, sample_rate_hz: int, seconds: float):
        import numpy as np  # type: ignore

        capacity = int(max(1.0, seconds) * sample_rate_hz)
        self._buffer = np.zeros((capacity,), dtype=np.float32)
        self._capacity = int(capacity)
        self._write_index = 0
        self._size = 0

    def append(self, chunk: "object") -> None:
        import numpy as np  # type: ignore

        data = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if data.size <= 0:
            return

        if data.size >= self._capacity:
            # Keep only the last part if someone passes a huge chunk.
            data = data[-self._capacity :]

        end_index = self._write_index + data.size
        if end_index <= self._capacity:
            self._buffer[self._write_index : end_index] = data
        else:
            first = self._capacity - self._write_index
            self._buffer[self._write_index :] = data[:first]
            self._buffer[: end_index % self._capacity] = data[first:]

        self._write_index = end_index % self._capacity
        self._size = min(self._capacity, self._size + data.size)

    def last_seconds(self, sample_rate_hz: int, seconds: float):
        import numpy as np  # type: ignore

        n = int(max(0.0, seconds) * sample_rate_hz)
        n = min(n, self._size)
        if n <= 0:
            return np.zeros((0,), dtype=np.float32)

        # Fast path: buffer not full yet (no wrap-around has happened).
        if self._size < self._capacity:
            end = self._write_index
            start = max(0, end - n)
            return self._buffer[start:end].copy()

        start_index = (self._write_index - n) % self._capacity
        # Buffer is full: either a contiguous slice or a wrapped slice.
        if start_index < self._write_index:
            return self._buffer[start_index : self._write_index].copy()

        return np.concatenate(
            (self._buffer[start_index:], self._buffer[: self._write_index]), axis=0
        ).copy()


class StreamingWhisper:
    """Streaming microphone transcription using faster-whisper.

    Typical usage
    -------------
    >>> stt = StreamingWhisper(model_size="medium", language="fr")
    >>> stt.start_stream()
    >>> ... do other work in main thread ...
    >>> stt.stop_stream()

    Notes
    -----
    - Prints partial results to stdout by default.
    - You can pass callbacks to integrate into your agent loop.
    """

    def __init__(
        self,
        *,
        sample_rate_hz: int = 16_000,
        chunk_seconds: float = 0.5,
        rolling_buffer_seconds: float = 30.0,
        partial_window_seconds: float = 6.0,
        partial_interval_seconds: float = 0.7,
        vad: VadConfig | None = None,
        language: str | None = "fr",
        task: str = "transcribe",
        model_size: str = "medium",
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
        input_device: int | str | None = None,
    ) -> None:
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be positive")

        self.sample_rate_hz = int(sample_rate_hz)
        self.chunk_seconds = float(chunk_seconds)
        self.chunk_frames = int(round(self.sample_rate_hz * self.chunk_seconds))

        if self.chunk_frames <= 0:
            raise ValueError("chunk_seconds too small for the given sample_rate_hz")

        self.rolling_buffer_seconds = float(rolling_buffer_seconds)
        self.partial_window_seconds = float(partial_window_seconds)

        self.vad = vad or VadConfig()
        self.language = language
        self.task = (task or "transcribe").strip() or "transcribe"
        self.model_size = (model_size or "medium").strip() or "medium"

        self.on_partial = on_partial or (lambda text: print(f"[partial] {text}"))
        self.on_final = on_final or (lambda text: print(f"[final] {text}"))

        self.input_device = input_device

        # Threading primitives.
        self._stop_event = threading.Event()
        self._paused_event = threading.Event()
        self._queue: queue.Queue["object"] = queue.Queue(maxsize=50)
        self._worker: threading.Thread | None = None

        # Audio stream handle (created on start).
        self._stream = None

        # Rolling buffer used for monitoring/debugging and for short-window partials.
        self._rolling = _RollingRingBuffer(
            sample_rate_hz=self.sample_rate_hz, seconds=self.rolling_buffer_seconds
        )

        # State for current utterance.
        self._in_speech = False
        self._speech_chunks: list["object"] = []
        self._speech_samples = 0
        self._silence_samples = 0

        # Load model once (cached) to keep latency low.
        self._model = self._load_model()

    def _load_model(self):
        """Load and return the faster-whisper model (GPU enabled when available)."""

        # Reuse the project's model loader and CUDA fallback behavior.
        from . import whisper_stt as stt_mod

        # Force the desired model size for this streaming instance.
        os.environ.setdefault("VOICE_AGENT_WHISPER_MODEL", self.model_size)

        return stt_mod._get_model()  # pylint: disable=protected-access

    def start_stream(self) -> None:
        """Start microphone capture + background transcription."""

        if self._worker and self._worker.is_alive():
            return

        try:
            import sounddevice as sd  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Missing dependency: sounddevice. Install it to use streaming microphone capture."
            ) from exc

        self._stop_event.clear()
        # By default, start in "resumed" state.
        self._paused_event.clear()

        def callback(indata, frames, time_info, status):  # type: ignore[no-untyped-def]
            if status:
                # Non-fatal glitches happen on Windows; avoid printing too much.
                pass

            # If paused (e.g., while TTS is speaking), drop microphone audio.
            # Dropping is intentional: it prevents feeding assistant audio back
            # into STT and keeps the callback non-blocking.
            if self._paused_event.is_set():
                return

            # Copy immediately; PortAudio reuses the buffer.
            chunk = indata.copy()
            try:
                self._queue.put_nowait(chunk)
            except queue.Full:
                # If the worker is too slow, drop audio rather than block the callback.
                # This keeps the app responsive; the user will see a gap instead of a freeze.
                return

        self._stream = sd.InputStream(
            samplerate=self.sample_rate_hz,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_frames,
            device=self.input_device,
            callback=callback,
        )
        self._stream.start()

        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def pause(self) -> None:
        """Pause STT processing (drops microphone audio).

        Intended use: call before playing TTS to avoid echo/transcription of the
        assistant voice.
        """

        self._paused_event.set()

        # Clear any queued chunks and reset utterance state so resume starts clean.
        try:
            while True:
                self._queue.get_nowait()
        except queue.Empty:
            pass

        self._reset_utterance_state()

    def resume(self) -> None:
        """Resume STT processing after a pause."""

        self._paused_event.clear()

    def stop_stream(self) -> None:
        """Stop transcription and close the microphone stream."""

        self._stop_event.set()
        self._paused_event.clear()

        # Best-effort: close stream.
        try:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
        except Exception:
            pass
        self._stream = None

        if self._worker is not None:
            self._worker.join(timeout=2.0)
        self._worker = None

    def process_chunk(self, chunk_float32):
        """Process a single chunk of float32 mono audio.

        This method is safe to call from a background worker thread.
        It must NOT be called from the sounddevice callback.
        """

        import numpy as np  # type: ignore

        audio = np.asarray(chunk_float32, dtype=np.float32)
        if audio.ndim == 2:
            audio = audio[:, 0]
        audio = audio.reshape(-1)
        if audio.size <= 0:
            return

        # Maintain rolling buffer (requirement).
        self._rolling.append(audio)

        # Energy-based VAD to detect start/end in real time.
        rms = float(np.sqrt(np.mean(np.square(audio))))

        if not self._in_speech:
            if rms >= self.vad.rms_start:
                self._in_speech = True
                self._speech_chunks = [audio]
                self._speech_samples = int(audio.size)
                self._silence_samples = 0
            return

        # In speech.
        self._speech_chunks.append(audio)
        self._speech_samples += int(audio.size)

        if rms < self.vad.rms_end:
            self._silence_samples += int(audio.size)
        else:
            self._silence_samples = 0

        utterance_seconds = self._speech_samples / float(self.sample_rate_hz)

        # End-of-utterance detection (hangover silence).
        silence_seconds = self._silence_samples / float(self.sample_rate_hz)
        if (
            utterance_seconds >= self.vad.min_utterance_seconds
            and silence_seconds >= self.vad.silence_seconds_to_end
        ):
            self._emit_final()
            self._reset_utterance_state()

    def _reset_utterance_state(self) -> None:
        self._in_speech = False
        self._speech_chunks = []
        self._speech_samples = 0
        self._silence_samples = 0

    def _decode(self, audio_float32):
        """Decode float32 mono audio in memory using faster-whisper."""

        from . import whisper_stt as stt_mod

        beam_size = int(os.environ.get("VOICE_AGENT_WHISPER_BEAM_SIZE", "1") or "1")
        best_of = int(os.environ.get("VOICE_AGENT_WHISPER_BEST_OF", "1") or "1")
        temperature = float(
            os.environ.get("VOICE_AGENT_WHISPER_TEMPERATURE", "0") or "0"
        )

        # Optional decoding bias knobs for better accuracy on domain phrases.
        initial_prompt = os.environ.get(
            "VOICE_AGENT_WHISPER_INITIAL_PROMPT", ""
        ).strip()
        hotwords = _parse_hotwords_env("VOICE_AGENT_WHISPER_HOTWORDS")

        # In this project we already do RMS-VAD to segment utterances.
        # Using Whisper's internal VAD filter can sometimes trim low-energy
        # endings (e.g., "au revoir"). Default: disabled, but configurable.
        vad_filter = _parse_bool_env("VOICE_AGENT_WHISPER_VAD_FILTER", False)

        transcribe_kwargs: dict = {
            "beam_size": max(1, beam_size),
            "best_of": max(1, best_of),
            "temperature": max(0.0, temperature),
            "vad_filter": vad_filter,
            "language": self.language,
            "task": self.task,
            "without_timestamps": True,
            "condition_on_previous_text": True,
        }

        if initial_prompt:
            transcribe_kwargs["initial_prompt"] = initial_prompt
        if hotwords:
            transcribe_kwargs["hotwords"] = hotwords

        # Decode the utterance once at end-of-speech.
        try:
            segments, _info = self._model.transcribe(audio_float32, **transcribe_kwargs)
        except TypeError:
            # Backward compatibility: older faster-whisper versions may not
            # support some kwargs (notably `hotwords`). Retry without them.
            transcribe_kwargs.pop("hotwords", None)
            segments, _info = self._model.transcribe(audio_float32, **transcribe_kwargs)

        # Force evaluation (CUDA errors can occur during iteration).
        parts: list[str] = []
        for seg in list(segments):
            text = (seg.text or "").strip()
            if text:
                parts.append(text)

        raw = " ".join(parts).strip()
        if not raw:
            return ""

        # Reuse the existing banking-friendly post-processing.
        return stt_mod._postprocess_french_banking_text(raw)  # pylint: disable=protected-access

    def _emit_final(self) -> None:
        import numpy as np  # type: ignore

        if not self._speech_chunks:
            return

        try:
            audio = np.concatenate(self._speech_chunks, axis=0).astype(np.float32)
            text = self._decode(audio)
        except Exception:
            return

        cleaned = (text or "").strip()
        if cleaned:
            self.on_final(cleaned)

    def _run_worker(self) -> None:
        while not self._stop_event.is_set():
            try:
                chunk = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self.process_chunk(chunk)
            except Exception:
                # Keep the worker alive; audio devices can be noisy.
                continue

        # Flush: if we were mid-speech, attempt a final decode.
        if self._in_speech and self._speech_chunks:
            self._emit_final()
            self._reset_utterance_state()


def _parse_input_device_from_env() -> int | str | None:
    raw = os.environ.get("VOICE_AGENT_AUDIO_INPUT_DEVICE", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def run_streaming_demo() -> None:
    """Convenience helper used by streaming_demo.py."""

    input_device = _parse_input_device_from_env()

    try:
        chunk_seconds = float(
            os.environ.get("VOICE_AGENT_STREAM_CHUNK_SECONDS", "0.25")
        )
    except ValueError:
        chunk_seconds = 0.25

    try:
        end_silence = float(
            os.environ.get("VOICE_AGENT_STREAM_END_SILENCE_SECONDS", "0.6")
        )
    except ValueError:
        end_silence = 0.6

    stt = StreamingWhisper(
        model_size=os.environ.get("VOICE_AGENT_WHISPER_MODEL", "small") or "small",
        language=os.environ.get("VOICE_AGENT_WHISPER_LANGUAGE", "fr") or "fr",
        chunk_seconds=max(0.05, chunk_seconds),
        vad=VadConfig(silence_seconds_to_end=max(0.05, end_silence)),
        input_device=input_device,
    )

    print("[stream] Starting microphone streaming. Press Ctrl+C to stop.")
    stt.start_stream()

    try:
        while True:
            # Keep main thread alive without blocking the callback.
            threading.Event().wait(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        print("[stream] Stopping...")
        stt.stop_stream()
        print("[stream] Stopped.")
