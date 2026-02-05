"""TTS package.

Public API:
- synthesize_speech: synthesize text to a WAV file using Piper.
"""

from .piper_tts import synthesize_speech

__all__ = ["synthesize_speech"]
