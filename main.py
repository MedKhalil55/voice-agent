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

from pathlib import Path
from time import strftime

from dotenv import load_dotenv

from audio import play_wav, record_audio
from llm import generate_ai_response
from stt import transcribe_audio
from tts import synthesize_speech


def _log(message: str) -> None:
    print(f"[{strftime('%H:%M:%S')}] {message}")


def main() -> None:
    load_dotenv()

    # Keep artifacts local and easy to inspect.
    out_dir = Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)

    recorded_wav = out_dir / "user.wav"
    tts_wav = out_dir / "assistant.wav"

    # Default duration is short for quick iteration.
    duration_s = 5

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

    try:
        while True:
            try:
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

                # Optional console control: type 'bye' to exit without speaking.
                typed = (
                    input("Press Enter to talk again (or type 'bye' to quit): ")
                    .strip()
                    .lower()
                )
                if typed in bye_keywords:
                    _log("Exiting...")
                    break

            except Exception as exc:
                _log(f"Error: {exc}")
                _log(
                    "Hints: Ollama must be running; Piper needs VOICE_AGENT_PIPER_MODEL/VOICE_AGENT_PIPER_CONFIG (and/or VOICE_AGENT_PIPER_BIN); "
                    "ffmpeg may be needed for STT decoding; check mic/speaker devices."
                )
                typed = (
                    input("Type 'retry' to continue or 'bye' to quit: ").strip().lower()
                )
                if typed in bye_keywords:
                    break

    except KeyboardInterrupt:
        _log("Interrupted by user")


if __name__ == "__main__":
    main()
