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
from datetime import datetime, timezone
from pathlib import Path
from time import strftime

from dotenv import load_dotenv

from audio import play_wav, record_audio
from llm import generate_ai_response
from stt import transcribe_audio
from tts import synthesize_speech


def _log(message: str) -> None:
    print(f"[{strftime('%H:%M:%S')}] {message}")


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
        default=5,
        help="Microphone recording duration per turn (seconds).",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()

    args = _parse_args()

    # Keep artifacts local and easy to inspect.
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    recorded_wav = out_dir / "user.wav"
    tts_wav = out_dir / "assistant.wav"

    duration_s = max(int(args.duration), 1)

    bye_keywords = {
        "bye",
        "goodbye",
        "good bye",
        "exit",
        "quit",
        "stop",
        "see you",
        "see you later",
    }

    _log("Voice agent ready. Speak normally; say 'bye' to exit.")

    session_summary: dict = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "demo" if args.demo else "mic",
        "record_duration_seconds": duration_s,
        "turns": [],
        "ended_reason": None,
    }

    demo_script = [
        "Hello.",
        "I received a payment reminder and I can't pay the full amount this month.",
        "Can you propose an installment plan?",
        "Please call me back next week.",
        "bye",
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
                        "Goodbye! If you need banking help later, just come back."
                    )
                    _log("Detected exit keyword. Closing conversation...")
                    _log("Step 4/5: synthesizing speech...")
                    synthesize_speech(farewell, str(tts_wav))
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
                synthesize_speech(response_text, str(tts_wav))
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

                # Optional console control: type 'bye' to exit without speaking.
                if not args.demo:
                    typed = (
                        input("Press Enter to talk again (or type 'bye' to quit): ")
                        .strip()
                        .lower()
                    )
                    if typed in bye_keywords:
                        session_summary["ended_reason"] = "user_typed_bye"
                        _log("Exiting...")
                        break

            except Exception as exc:
                _log(f"Error: {exc}")
                _log(
                    "Hints: Ollama must be running; Piper needs VOICE_AGENT_PIPER_MODEL/VOICE_AGENT_PIPER_CONFIG (and/or VOICE_AGENT_PIPER_BIN); "
                    "ffmpeg may be needed for STT decoding; check mic/speaker devices."
                )
                if args.demo:
                    session_summary["ended_reason"] = "error_in_demo"
                    break

                typed = (
                    input("Type 'retry' to continue or 'bye' to quit: ").strip().lower()
                )
                if typed in bye_keywords:
                    session_summary["ended_reason"] = "user_quit_after_error"
                    break

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
