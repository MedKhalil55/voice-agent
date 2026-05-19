from __future__ import annotations

import asyncio
import os
from typing import Awaitable, Callable

import numpy as np

from dotenv import load_dotenv

from main import OUTBOUND_GREETING, VoiceAgent, clean_for_tts

load_dotenv(override=True)


class WebSocketVoiceAgent(VoiceAgent):
    """VoiceAgent variant that receives audio from WebSocket.

    This keeps the existing negotiation / verification / LLM logic intact,
    and only replaces audio I/O:
    - Input: `process_audio_chunk(pcm_bytes)` accepts raw PCM 16kHz 16-bit mono.
    - Output: `speak()` generates WAV bytes and sends them via callbacks.

    Notes
    -----
    - Does NOT start the microphone (`StreamingWhisper.start_stream`).
    - Uses the same VAD + end-of-utterance logic as the local microphone mode.
    - Mocks the inherited `_stt` boundary so pause/resume calls are safe.
    """

    def __init__(
        self,
        customer_id: int,
        send_audio_callback: Callable[[bytes], Awaitable[None]],
        send_transcript_callback: Callable[[dict], Awaitable[None]],
    ) -> None:
        # Store callbacks BEFORE super().__init__() because VoiceAgent methods may speak.
        self._send_audio = send_audio_callback
        self._send_transcript = send_transcript_callback

        # Capture the loop used by the websocket connection.
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = asyncio.get_event_loop()

        # Override customer_id in env before init (VoiceAgent reads it).
        os.environ["VOICE_AGENT_CUSTOMER_ID"] = str(customer_id)

        super().__init__()
        self._customer_id = int(customer_id)

        # --- Replace microphone STT with StreamingWhisper (no microphone) ---
        # We reuse the project's exact VAD segmentation behavior, but we do NOT
        # call start_stream(); we only feed chunks via process_chunk().
        from stt.streaming_whisper import StreamingWhisper, VadConfig

        self._ws_stt = StreamingWhisper(
            model_size=os.environ.get("VOICE_AGENT_WHISPER_MODEL", "small") or "small",
            language=os.environ.get("VOICE_AGENT_WHISPER_LANGUAGE", "fr") or "fr",
            chunk_seconds=0.25,
            vad=VadConfig(
                silence_seconds_to_end=0.8,
                min_utterance_seconds=0.3,
                rms_start=0.012,
                rms_end=0.010,
            ),
            on_final=self._on_ws_final,
            on_partial=lambda _text: None,
        )

        self._ws_audio_paused = False

        # Replace inherited streaming-mic STT boundary with a no-op so all
        # VoiceAgent pause/resume calls keep working without a microphone.
        class _WsSTTProxy:  # noqa: N801
            def __init__(self) -> None:
                self._last_decode_seconds = 0.0

            def pause(self) -> None:
                self_outer._ws_audio_paused = True

            def resume(self) -> None:
                self_outer._ws_audio_paused = False

            def stop_stream(self) -> None:
                return None

        self_outer = self
        self._stt = _WsSTTProxy()

    def _on_ws_final(self, text: str) -> None:
        """Callback from StreamingWhisper when an utterance is complete."""

        from main import _log

        cleaned = (text or "").strip()
        if not cleaned:
            return

        _log(f"[WS-STT] Final: {cleaned!r}")

        try:
            self._stt._last_decode_seconds = float(
                getattr(self._ws_stt, "_last_decode_seconds", 0.0)
            )
        except Exception:
            pass

        # Send transcript event to dashboard.
        asyncio.run_coroutine_threadsafe(
            self._send_transcript({"type": "client_speech", "text": cleaned}),
            self._loop,
        )

        # Feed into the inherited VoiceAgent flow.
        from threading import Thread

        Thread(target=self.handle_final, args=(cleaned,), daemon=True).start()

    def start_ws(self) -> None:
        """Start VoiceAgent WITHOUT microphone.

        Loads client info and sends a greeting via websocket TTS.
        """

        from threading import Thread

        # Retrieve client name before greeting (same logic as VoiceAgent.start).
        client_name = ""
        try:
            from db.tools import get_client_info

            info = get_client_info(self._customer_id)
            if info.get("found"):
                self._client_info = info
                client_name = str(info.get("customer_name", "")).strip()
        except Exception:
            client_name = ""

        civilite = "Monsieur"

        if client_name:
            greeting = (
                "Bonjour, je suis l'assistant de recouvrement "
                "de votre établissement bancaire. "
                "Je vous contacte au sujet de votre compte. "
                f"Ai-je bien {civilite} {client_name} en ligne ?"
            )
            self._awaiting_name_confirmation = True
        else:
            greeting = OUTBOUND_GREETING
            self._awaiting_dob = True

        # Send greeting as TTS audio via WebSocket.
        self.speak(greeting)

        # Preload client profile in background.
        Thread(target=self._preload_client_info, daemon=True).start()

    def speak(self, text: str, log_output: bool = True) -> None:
        """Override: generate WAV bytes and send via WebSocket.

        The inherited logic calls `speak()` from background threads; therefore
        we schedule the async callbacks onto the captured event loop.
        """

        from main import _log

        raw_text = (text or "").strip()
        if raw_text and log_output:
            _log(f"[WS-TTS] {raw_text!r}")

        cleaned = clean_for_tts(text)
        if not cleaned:
            return

        # Send transcript event to dashboard.
        asyncio.run_coroutine_threadsafe(
            self._send_transcript({"type": "agent_speech", "text": raw_text}),
            self._loop,
        )

        audio_bytes = self._tts_to_bytes(cleaned)
        if audio_bytes:
            asyncio.run_coroutine_threadsafe(self._send_audio(audio_bytes), self._loop)

    def _tts_to_bytes(self, text: str) -> bytes | None:
        """Run Piper TTS and return WAV bytes instead of playing locally."""

        from main import _log

        import tempfile

        tmp_path = ""
        try:
            from tts.piper_tts import synthesize_speech

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            synthesize_speech(text, tmp_path)
            with open(tmp_path, "rb") as fh:
                return fh.read()
        except Exception as exc:
            _log(f"[WS-TTS] Error: {exc}")
            return None
        finally:
            try:
                if tmp_path:
                    os.unlink(tmp_path)
            except Exception:
                pass

    async def process_audio_chunk(self, pcm_bytes: bytes) -> None:
        """Feed PCM16 16kHz mono chunks into StreamingWhisper (VAD+buffering)."""

        if not pcm_bytes:
            return

        if getattr(self, "_ws_audio_paused", False):
            return

        # PCM int16 -> float32 in [-1, 1]
        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
        if audio_int16.size <= 0:
            return
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._ws_stt.process_chunk, audio_float32)

    def shutdown(self) -> None:
        try:
            self._ws_stt.stop_stream()
        except Exception:
            pass
        super().shutdown()

    # Override STT pause/resume (no-op since no microphone)
    def _pause_stt(self) -> None:
        return None

    def _resume_stt(self) -> None:
        return None
