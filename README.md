
# Voice-Agent (local voice assistant)

Local-only voice assistant pipeline:

1) Record microphone audio (WAV)
2) Transcribe with Whisper (faster-whisper)
3) Generate a response with Ollama (LangChain)
4) Synthesize speech with Piper (recommended: standalone `piper.exe` on Windows)
5) Play the audio response

## Prerequisites

- Python (project is set up for Python 3.12)
- `uv` installed
- Ollama installed and running (default: `http://localhost:11434`)
- A Piper voice model (`.onnx` + `.onnx.json`)

## Configuration (.env)

Create a `.env` file in the project root (same folder as `main.py`). Example:

```
VOICE_AGENT_PIPER_BIN=C:\Users\MSI\Documents\piper\piper.exe
VOICE_AGENT_PIPER_MODEL=voices/en_US-lessac-medium.onnx
VOICE_AGENT_PIPER_CONFIG=voices/en_US-lessac-medium.onnx.json

# Optional
VOICE_AGENT_OLLAMA_MODEL=llama3.2
VOICE_AGENT_OLLAMA_BASE_URL=http://localhost:11434
VOICE_AGENT_WHISPER_MODEL=small
VOICE_AGENT_WHISPER_COMPUTE_TYPE=int8
VOICE_AGENT_WHISPER_BEAM_SIZE=1
```

Notes:
- Relative paths like `voices/...` are resolved relative to the project root.
- On Windows, using `VOICE_AGENT_PIPER_BIN` (standalone Piper) avoids common Python-package phonemizer issues.

## Run

From the project root:

```
uv run main.py
```

Outputs are written to the `artifacts/` folder (for example: `artifacts/user.wav` and `artifacts/assistant.wav`).

## Quick TTS smoke test

This explicitly loads `.env` and synthesizes a WAV:

```
uv run python -c "from dotenv import load_dotenv; load_dotenv(); from tts import synthesize_speech; synthesize_speech('Hello!', 'artifacts/tts_test.wav')"
```

