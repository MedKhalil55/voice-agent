"""STT package.

Currently exposes a single public helper built on faster-whisper.
"""

from .whisper_stt import transcribe_audio

__all__ = ["transcribe_audio"]
