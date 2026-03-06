"""Run streaming transcription from the microphone.

Usage
-----
    uv run streaming_demo.py

Notes
-----
- Uses faster-whisper (medium) with CUDA if available.
- Captures 0.5s chunks, keeps all audio in memory (float32), prints partials.
- Configure microphone device via VOICE_AGENT_AUDIO_INPUT_DEVICE.
"""

from __future__ import annotations

from stt.streaming_whisper import run_streaming_demo


if __name__ == "__main__":
    run_streaming_demo()
