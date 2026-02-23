"""Local AI voice assistant orchestrator.

Flow
----
1) Record audio (microphone)
2) Transcribe audio (faster-whisper)
3) Generate response (LangChain + Ollama)
4) Synthesize speech (Piper)
5) Play audio (speaker)

All components run locally.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from time import sleep, strftime

from dotenv import load_dotenv

from audio import play_wav, record_audio
from llm import generate_ai_response
from stt import transcribe_audio
from tts import synthesize_speech


DEFAULT_RECORD_DURATION_S = 5

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


def _utc_timestamp_for_filename() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _write_conversation_summary(out_dir: Path, summary: dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Append one JSON object per line (JSONL). This keeps a full history of runs
    # without overwriting prior sessions.
    out_path = out_dir / "conversations.jsonl"
    with out_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(summary, ensure_ascii=False))
        file.write("\n")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local voice agent")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a deterministic demo conversation (no microphone/STT).",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=DEFAULT_RECORD_DURATION_S,
        help="Microphone recording duration per turn (seconds). If stop-on-silence is enabled, this is the MAX duration.",
    )
    return parser.parse_args()


def main() -> None:
    # Important on Windows/PowerShell: environment variables may already be set
    # in the shell/session. We want the project's `.env` to take precedence so
    # switching voices (e.g., EN -> FR) actually applies.
    load_dotenv(override=True)

    args = _parse_args()

    # Debug visibility: confirm which Piper voice paths are active.
    piper_model = os.environ.get("VOICE_AGENT_PIPER_MODEL", "").strip()
    piper_config = os.environ.get("VOICE_AGENT_PIPER_CONFIG", "").strip()
    if piper_model:
        _log(f"TTS voice model: {piper_model}")
    if piper_config:
        _log(f"TTS voice config: {piper_config}")

    # Keep artifacts local and easy to inspect.
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    recorded_wav = out_dir / "user.wav"
    tts_wav = out_dir / "assistant.wav"

    duration_s = max(int(args.duration), 1)

    # If silence-stop recording is enabled, `duration_s` becomes a MAX duration.
    # With a short max duration (e.g., 5s) you may never reach "3 seconds of
    # silence" if the user speaks for a couple seconds first. To avoid the
    # recording always ending at 5s, we automatically bump the max duration.
    stop_on_silence_raw = os.environ.get(
        "VOICE_AGENT_RECORD_STOP_ON_SILENCE_SECONDS", ""
    ).strip()
    silence_seconds = 0.0
    if stop_on_silence_raw:
        try:
            silence_seconds = max(float(stop_on_silence_raw), 0.0)
        except ValueError:
            silence_seconds = 0.0

    if silence_seconds > 0:
        # In stop-on-silence mode, `--duration` is interpreted as a MAX duration.
        # If the user didn't override `--duration`, we use a more generous default
        # to avoid cutting off longer utterances.
        max_raw = os.environ.get("VOICE_AGENT_RECORD_MAX_SECONDS", "").strip()
        max_env_s = 0
        if max_raw:
            try:
                max_env_s = int(float(max_raw))
            except ValueError:
                max_env_s = 0

        if max_env_s > 0:
            duration_s = max(max_env_s, 1)
        elif int(args.duration) == DEFAULT_RECORD_DURATION_S:
            duration_s = 30

        min_max_duration = int(silence_seconds) + 7  # buffer for speaking time
        if duration_s < min_max_duration:
            duration_s = min_max_duration

        _log(
            f"Recording mode: stop-on-silence ({silence_seconds:.1f}s). Max duration per turn: {duration_s}s"
        )
    else:
        _log(f"Recording mode: fixed window. Duration per turn: {duration_s}s")

    bye_keywords = {
        "au revoir",
        "aurevoir",
        "bonne journée",
        "à bientôt",
        "a bientot",
        "à plus tard",
        "a plus tard",
        "merci, au revoir",
        "terminer",
        "quitter",
        "arrêter",
        "arreter",
    }

    _log("Assistant prêt. Dites « au revoir » pour terminer.")

    session_summary: dict = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "demo" if args.demo else "mic",
        "record_duration_seconds": duration_s,
        "turns": [],
        "ended_reason": None,
    }

    # Outbound-call behavior: the assistant starts the conversation.
    _log("Greeting: synthesizing speech...")
    synthesize_speech(clean_for_tts(OUTBOUND_GREETING), str(tts_wav))
    _log("Greeting: playing audio...")
    play_wav(str(tts_wav))
    session_summary["turns"].append(
        {
            "user_text": None,
            "assistant_text": OUTBOUND_GREETING,
            "exit": False,
            "event": "greeting",
        }
    )
    # Small buffer so playback tail doesn't immediately leak into the next recording.
    if not args.demo:
        sleep(0.25)

    demo_script = [
        "Bonjour.",
        "J'ai reçu un rappel de paiement et je ne peux pas régler la totalité ce mois-ci.",
        "Pouvez-vous me proposer un plan de paiement en plusieurs fois ?",
        "Pouvez-vous me rappeler la semaine prochaine ?",
        "au revoir",
    ]

    try:
        while True:
            try:
                if args.demo:
                    if not demo_script:
                        session_summary["ended_reason"] = "demo_complete"
                        _log("Demo script finished.")
                        break

                    user_text = demo_script.pop(0)
                    _log(f"(DEMO) User text: {user_text!r}")
                else:
                    _log("Step 1/5: recording microphone audio...")
                    record_audio(str(recorded_wav), duration=duration_s)
                    _log(f"Recorded: {recorded_wav}")

                    _log("Step 2/5: transcribing audio...")
                    user_text = transcribe_audio(str(recorded_wav))
                    _log(f"User text: {user_text!r}")

                normalized = " ".join((user_text or "").lower().split())
                if normalized in bye_keywords or any(
                    k in normalized for k in bye_keywords
                ):
                    farewell = (
                        "Au revoir et merci de votre appel. "
                        "Si vous avez besoin d'aide, n'hésitez pas à nous recontacter."
                    )
                    _log("Detected exit keyword. Closing conversation...")
                    _log("Step 4/5: synthesizing speech...")
                    synthesize_speech(clean_for_tts(farewell), str(tts_wav))
                    _log("Step 5/5: playing audio...")
                    play_wav(str(tts_wav))

                    session_summary["turns"].append(
                        {
                            "user_text": user_text,
                            "assistant_text": farewell,
                            "exit": True,
                        }
                    )
                    session_summary["ended_reason"] = "user_said_bye"
                    break

                if not normalized:
                    _log("No speech detected. Try again.")
                    continue

                _log("Step 3/5: generating AI response...")
                response_text = generate_ai_response(user_text)
                _log(f"Assistant text: {response_text!r}")

                _log("Step 4/5: synthesizing speech...")
                synthesize_speech(clean_for_tts(response_text), str(tts_wav))
                _log(f"Synthesized: {tts_wav}")

                _log("Step 5/5: playing audio...")
                play_wav(str(tts_wav))

                session_summary["turns"].append(
                    {
                        "user_text": user_text,
                        "assistant_text": response_text,
                        "exit": False,
                    }
                )

                # Continuous loop: immediately go to the next recording.
                if not args.demo:
                    sleep(0.25)

            except Exception as exc:
                _log(f"Error: {exc}")
                _log(
                    "Hints: Ollama must be running; Piper needs VOICE_AGENT_PIPER_MODEL/VOICE_AGENT_PIPER_CONFIG (and/or VOICE_AGENT_PIPER_BIN); "
                    "ffmpeg may be needed for STT decoding; check mic/speaker devices."
                )
                if args.demo:
                    session_summary["ended_reason"] = "error_in_demo"
                    break

                # In mic mode, keep running unless the user says an exit intent.
                sleep(0.5)
                continue

    except KeyboardInterrupt:
        session_summary["ended_reason"] = "keyboard_interrupt"
        _log("Interrupted by user")
    finally:
        session_summary["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        if session_summary.get("ended_reason") is None:
            session_summary["ended_reason"] = "completed"
        summary_path = _write_conversation_summary(out_dir, session_summary)
        _log(f"Conversation summary saved: {summary_path}")


if __name__ == "__main__":
    main()
