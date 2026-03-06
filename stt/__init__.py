"""STT package.

Exports:
- `transcribe_audio`: offline file transcription
- `warmup_stt`: preload Whisper model
- `StreamingWhisper`: in-memory pseudo-streaming transcription
"""

from .streaming_whisper import StreamingWhisper
from .whisper_stt import transcribe_audio, warmup_stt

__all__ = ["StreamingWhisper", "transcribe_audio", "warmup_stt"]
