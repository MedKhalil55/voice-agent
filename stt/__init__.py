"""STT package.

Currently exposes a single public helper built on faster-whisper.
"""

from .whisper_stt import transcribe_audio, warmup_stt

__all__ = ["transcribe_audio", "warmup_stt"]
