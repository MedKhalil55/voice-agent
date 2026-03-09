"""TTS package.

Public API:
- synthesize_speech: synthesize text to a WAV file using Piper.
- speak_streaming: synthesize and play text with streaming playback (no WAV).
- warmup_tts: pre-validate Piper paths and cache config at startup.
"""

from .piper_tts import speak_streaming, synthesize_speech, warmup_tts

__all__ = ["synthesize_speech", "speak_streaming", "warmup_tts"]
