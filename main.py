"""Local AI voice assistant (streaming STT).

This is the refactored entrypoint that uses `StreamingWhisper` instead of the
old synchronous pipeline (record → write WAV → transcribe).

High-level flow (production-style)
---------------------------------
1) Start a single `StreamingWhisper` instance at startup.
2) `on_final` events: trigger one LLM+TTS response at a time.
4) While TTS is speaking, pause STT to avoid echo / feedback loops.

Design constraints
------------------
- No temporary WAV files for user audio.
- Main thread must remain responsive (callbacks must stay lightweight).
- Prevent double-processing with a lock.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from time import strftime

from dotenv import load_dotenv

from llm import generate_ai_response, warmup_llm
from stt import StreamingWhisper, warmup_stt
from stt.streaming_whisper import VadConfig
from tts import speak_streaming, warmup_tts


OUTBOUND_GREETING = (
    "Bonjour, je suis l’assistant bancaire automatique et je vous appelle au sujet de votre compte. "
    "Comment puis-je vous aider aujourd’hui ?"
)


def _log(message: str) -> None:
    print(f"[{strftime('%H:%M:%S')}] {message}")


def clean_for_tts(text: str) -> str:
    """Post-process text to sound natural when spoken.

    - Strips common Markdown formatting (bold/italic, headings, bullets)
    - Removes numbered list prefixes (e.g., "1. ")
    - Removes special symbols used for formatting (e.g., "#", "**")
    - Turns multiple newlines into a natural pause (". ")
    """

    value = (text or "").strip()
    if not value:
        return ""

    # Normalize line endings first.
    value = value.replace("\r\n", "\n").replace("\r", "\n")

    # Remove fenced code block markers and inline code backticks.
    value = value.replace("```", "")
    value = value.replace("`", "")

    # Remove common emphasis markers.
    value = re.sub(r"(\*\*|__)(.+?)(\1)", r"\2", value)
    value = re.sub(r"(\*|_)(.+?)(\1)", r"\2", value)

    # Strip Markdown structural prefixes line-by-line.
    value = re.sub(r"^\s*#+\s+", "", value, flags=re.MULTILINE)  # headings
    value = re.sub(r"^\s*>\s+", "", value, flags=re.MULTILINE)  # blockquotes
    value = re.sub(r"^\s*[-*+]\s+", "", value, flags=re.MULTILINE)  # bullets
    value = re.sub(r"^\s*\d+[\.)]\s+", "", value, flags=re.MULTILINE)  # numbered

    # Remove leftover formatting symbols that tend to be read aloud badly.
    value = value.replace("#", " ")
    value = value.replace("*", " ")
    value = value.replace("_", " ")

    # Replace multiple newlines with a pause, then remaining newlines with spaces.
    value = re.sub(r"\n\s*\n+", ". ", value)
    value = value.replace("\n", " ")

    # Collapse whitespace.
    value = re.sub(r"\s{2,}", " ", value).strip()
    return value


def _write_conversation_summary(out_dir: Path, summary: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Append one JSON object per line (JSONL). This keeps a full history of runs
    # without overwriting prior sessions.
    out_path = out_dir / "conversations.jsonl"
    with out_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary, ensure_ascii=False))
        file.write("\n")
    return out_path


def _parse_input_device_from_env() -> int | str | None:
    """Parse VOICE_AGENT_AUDIO_INPUT_DEVICE (index or exact device name)."""

    raw = os.environ.get("VOICE_AGENT_AUDIO_INPUT_DEVICE", "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return raw


def _parse_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


class VoiceAgent:
    """Voice agent that continuously listens and responds.

    Concurrency model
    -----------------
    - Microphone capture runs in PortAudio's callback thread.
    - STT decoding runs in `StreamingWhisper`'s worker thread.
    - LLM+TTS runs in a dedicated response thread spawned per final transcript.

    We keep callbacks lightweight and use a `Lock` to prevent overlapping
    responses and double-processing.
    """

    def __init__(self) -> None:
        # On Windows/PowerShell, environment variables may already be set in the
        # session. We want `.env` to take precedence.
        load_dotenv(override=True)

        self._shutdown_event = Event()
        self._processing_lock = Lock()

        # Keep artifacts local and easy to inspect.
        self._out_dir = Path("artifacts")
        self._out_dir.mkdir(parents=True, exist_ok=True)

        self._session_summary: dict = {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "streaming_mic",
            "turns": [],
            "ended_reason": None,
        }

        self._bye_keywords = {
            "au revoir",
            "aurevoir",
            # Common ASR confusions for "au revoir" in call audio.
            "on va voir",
            "on va voir maintenant",
            "bonne journée",
            "a bientôt",
            "a bientot",
            "à plus tard",
            "a plus tard",
            "merci, au revoir",
            "terminer",
            "quitter",
            "arrêter",
            "arreter",
        }

        self._last_final_text = ""
        input_device = _parse_input_device_from_env()

        # Streaming knobs (tuned for phone-call UX).
        chunk_seconds = _parse_float_env("VOICE_AGENT_STREAM_CHUNK_SECONDS", 0.25)
        end_silence_seconds = _parse_float_env(
            "VOICE_AGENT_STREAM_END_SILENCE_SECONDS", 0.6
        )
        self._stt = StreamingWhisper(
            model_size=os.environ.get("VOICE_AGENT_WHISPER_MODEL", "small") or "small",
            language=os.environ.get("VOICE_AGENT_WHISPER_LANGUAGE", "fr") or "fr",
            chunk_seconds=max(0.05, chunk_seconds),
            vad=VadConfig(silence_seconds_to_end=max(0.05, end_silence_seconds)),
            on_partial=self.handle_partial,
            on_final=self.handle_final,
            input_device=input_device,
        )

    def start(self) -> None:
        """Start STT streaming and greet the user."""

        # Warm up heavy models (optional). Runs in background.
        warmup_enabled = os.environ.get(
            "VOICE_AGENT_WARMUP", "true"
        ).strip().lower() not in {
            "0",
            "false",
            "no",
            "off",
        }
        if warmup_enabled:

            def _warmup_stt_safe() -> None:
                try:
                    warmup_stt()
                except Exception as exc:
                    _log(f"Warmup STT skipped: {exc}")

            def _warmup_llm_safe() -> None:
                try:
                    warmup_llm()
                except Exception as exc:
                    _log(f"Warmup LLM skipped: {exc}")

            def _warmup_tts_safe() -> None:
                try:
                    warmup_tts()
                except Exception as exc:
                    _log(f"Warmup TTS skipped: {exc}")

            Thread(target=_warmup_stt_safe, daemon=True).start()
            Thread(target=_warmup_llm_safe, daemon=True).start()
            Thread(target=_warmup_tts_safe, daemon=True).start()

        # Start microphone streaming once.
        self._stt.start_stream()

        # Pause during greeting playback to avoid STT hearing the assistant.
        self._stt.pause()
        try:
            self.speak(OUTBOUND_GREETING)
            self._session_summary["turns"].append(
                {
                    "user_text": None,
                    "assistant_text": OUTBOUND_GREETING,
                    "event": "greeting",
                }
            )
        finally:
            self._stt.resume()

        _log("Assistant prêt. Parlez, puis faites une courte pause.")

    def handle_partial(self, text: str) -> None:
        """Partial transcripts: log for debugging."""

        cleaned = (text or "").strip()
        if cleaned:
            _log(f"Partial: {cleaned}")

    def handle_final(self, text: str) -> None:
        """Final transcript: trigger LLM response (one at a time)."""

        cleaned = (text or "").strip()
        if not cleaned:
            return

        normalized = " ".join(cleaned.lower().split())
        if not normalized:
            return

        # Avoid occasional duplicate finals caused by partial-window re-decodes.
        if normalized == self._last_final_text:
            return
        self._last_final_text = normalized

        # Exit intent (optional) for operator convenience.
        if normalized in self._bye_keywords or any(
            k in normalized for k in self._bye_keywords
        ):
            _log("Detected exit keyword. Shutting down...")
            Thread(target=self.shutdown, daemon=True).start()
            return

        # Prevent overlapping responses.
        if not self._processing_lock.acquire(blocking=False):
            _log("Ignoring final transcript (assistant is busy).")
            return

        Thread(target=self._respond_worker, args=(cleaned,), daemon=True).start()

    def _respond_worker(self, user_text: str) -> None:
        """Background worker: LLM → TTS → playback."""
        import time as _time

        try:
            _log(f"User: {user_text!r}")
            self._session_summary["turns"].append({"user_text": user_text})

            t0 = _time.monotonic()
            assistant_text = generate_ai_response(user_text)
            llm_elapsed = _time.monotonic() - t0
            _log(f"LLM latency: {llm_elapsed:.2f} sec")
            _log(f"Assistant: {assistant_text!r}")

            # Pause STT while speaking to avoid echo.
            self._stt.pause()
            try:
                self.speak(assistant_text)
            finally:
                self._stt.resume()

            stt_latency = getattr(self._stt, "_last_decode_seconds", 0.0)
            total = stt_latency + llm_elapsed
            _log(f"TOTAL pipeline latency: {total:.2f} sec")

            # Persist conversation turn.
            self._session_summary["turns"][-1]["assistant_text"] = assistant_text

        except Exception as exc:
            _log(f"Error in response worker: {exc}")
        finally:
            try:
                self._processing_lock.release()
            except RuntimeError:
                pass

    def generate_response(self, user_text: str) -> str:
        """LLM boundary (kept as a method for easy future tool/RAG integration)."""

        return generate_ai_response(user_text)

    def speak(self, text: str) -> None:
        """TTS boundary: streaming playback with no intermediate WAV files."""

        speak_streaming(clean_for_tts(text))

    def shutdown(self) -> None:
        """Stop streaming and write a session summary."""

        if self._shutdown_event.is_set():
            return

        self._shutdown_event.set()
        try:
            self._stt.stop_stream()
        except Exception:
            pass

        self._session_summary["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        if self._session_summary.get("ended_reason") is None:
            self._session_summary["ended_reason"] = "shutdown"

        try:
            summary_path = _write_conversation_summary(
                self._out_dir, self._session_summary
            )
            _log(f"Conversation summary saved: {summary_path}")
        except Exception as exc:
            _log(f"Failed to write conversation summary: {exc}")


def main() -> None:
    agent = VoiceAgent()
    agent.start()

    try:
        # Keep main thread alive; all work happens on background threads.
        while True:
            if agent._shutdown_event.wait(0.25):
                break
    except KeyboardInterrupt:
        _log("Interrupted by user")
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()
