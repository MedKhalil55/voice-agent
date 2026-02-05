"""Audio package.

Public API:
- record_audio: record from microphone to a WAV file.
 - play_wav: play a WAV file through speakers.
"""

from .recorder import record_audio
from .player import play_wav

__all__ = ["record_audio", "play_wav"]
