# Voice-Agent — AI Agentic Voice Assistant

> **Branch note:** the `main` branch contains the original local voice-assistant pipeline. The **`mcpBranch`** branch is the final, most complete version of this project — it extends the pipeline into a full **agentic AI system** with **LangGraph** orchestration, a **RAG** knowledge-retrieval layer, and **MCP (Model Context Protocol)** tool integration. Check out `mcpBranch` to see the full implementation.

A local-first, end-to-end voice AI agent: it listens, reasons over retrieved knowledge, can call tools through MCP, and speaks back — all orchestrated as an agentic workflow rather than a single prompt-response loop.

## Pipeline overview (mcpBranch)

1. **Capture** — record microphone audio (WAV / streaming chunks)
2. **Transcribe** — speech-to-text with Whisper (`faster-whisper`)
3. **Retrieve** — relevant context is pulled from a **RAG** pipeline backed by **ChromaDB**
4. **Reason & orchestrate** — a **LangGraph** agent graph drives multi-step reasoning over an LLM (via Ollama), deciding when to retrieve, when to call a tool, and when to respond
5. **Act** — the agent can invoke external tools/capabilities through **MCP (Model Context Protocol)**, rather than being limited to text generation alone
6. **Synthesize** — the response is converted back to speech with Piper TTS
7. **Play** — the audio response is played back to the user

This turns the original single-shot voice pipeline (`main` branch: Whisper → LangChain → Piper) into a genuine **agentic voice assistant**, capable of retrieval-augmented, tool-using, multi-step reasoning in a live conversation.

## Why this project

This started as a local voice-assistant experiment and evolved into a testbed for agentic AI patterns — LangGraph-based orchestration, RAG retrieval, and MCP tool calling — applied to a real-time, low-latency voice interface. It's the foundation for the AI voice agent architecture used in my Talys Consulting internship project.

## Prerequisites

- Python (project is set up for Python 3.12)
- `uv` installed
- Ollama installed and running (default: `http://localhost:11434`)
- A Piper voice model (`.onnx` + `.onnx.json`)
- ChromaDB (for the RAG layer — see `mcpBranch`)
- An MCP-compatible tool/server configuration (see `mcpBranch`)

## Configuration (.env)

Create a `.env` file in the project root (same folder as `main.py`). Example:

```
VOICE_AGENT_PIPER_BIN=C:\Users\MSI\Documents\piper\piper.exe
VOICE_AGENT_PIPER_MODEL=voices/en_US-lessac-medium.onnx
VOICE_AGENT_PIPER_CONFIG=voices/en_US-lessac-medium.onnx.json

# Optional
VOICE_AGENT_OLLAMA_MODEL=llama3.2
VOICE_AGENT_OLLAMA_BASE_URL=http://localhost:11434
VOICE_AGENT_WHISPER_MODEL=small.en
VOICE_AGENT_WHISPER_LANGUAGE=en
VOICE_AGENT_WHISPER_COMPUTE_TYPE=int8
VOICE_AGENT_WHISPER_BEAM_SIZE=1

# Audio recording
# Stop recording when the speaker is silent for N seconds.
VOICE_AGENT_RECORD_STOP_ON_SILENCE_SECONDS=3

# Optional
# VOICE_AGENT_RECORD_MAX_SECONDS=30
# VOICE_AGENT_SILENCE_RMS_THRESHOLD=0.01
# VOICE_AGENT_MIN_RECORD_SECONDS=0.6
# VOICE_AGENT_AUDIO_INPUT_DEVICE=1
# VOICE_AGENT_AUDIO_INPUT_DEVICE=Microphone (Realtek(R) Audio)
```

> On `mcpBranch`, additional environment variables are required for the ChromaDB RAG store and MCP server connection — see that branch's README/config for the full list.

Notes:
- Relative paths like `voices/...` are resolved relative to the project root.
- On Windows, using `VOICE_AGENT_PIPER_BIN` (standalone Piper) avoids common Python-package phonemizer issues.

## Run

From the project root:

```
uv run main.py
```

For the full agentic pipeline (LangGraph + RAG + MCP), switch to `mcpBranch` and follow its setup instructions.

## Streaming transcription (real-time partials)

This project also includes a pseudo-streaming mode that:

- Captures the microphone in 0.5s float32 chunks
- Keeps audio in memory (no temporary WAV files)
- Uses lightweight VAD for start/end detection
- Prints partial transcripts while you speak, then a final transcript

Run:

```
uv run streaming_demo.py
```

Tip: if the wrong microphone is used, set `VOICE_AGENT_AUDIO_INPUT_DEVICE` in `.env`.

Outputs are written to the `artifacts/` folder (for example: `artifacts/user.wav` and `artifacts/assistant.wav`).

## Troubleshooting STT (wrong language / hearing the assistant)

- If you speak English but transcription comes out in Arabic/Russian/etc, set:
	- `VOICE_AGENT_WHISPER_LANGUAGE=en`
	- and optionally use an English-only model like `VOICE_AGENT_WHISPER_MODEL=small.en`
- If the transcript sounds like it is hearing the **speaker output** (the assistant) instead of your voice, your input device may be wrong (e.g., Windows "Stereo Mix").
	- Listen to `artifacts/user.wav` to confirm what was actually recorded.
	- Set `VOICE_AGENT_AUDIO_INPUT_DEVICE` to your real microphone.

## Dynamic recording (stop on silence)

By default, recording can stop automatically when you stop speaking (silence) instead of always waiting a fixed duration.

- Enable: set `VOICE_AGENT_RECORD_STOP_ON_SILENCE_SECONDS=3`
- If it cuts off while you're still talking: increase `VOICE_AGENT_RECORD_MAX_SECONDS` (e.g., `60`)
- If it stops too early: lower `VOICE_AGENT_SILENCE_RMS_THRESHOLD` (e.g., `0.005`)
- If it never stops: increase `VOICE_AGENT_SILENCE_RMS_THRESHOLD` (e.g., `0.02`)

## Latency tuning (phone-call UX)

If your "temps de réponse" feels too slow, the total latency is usually the sum of:

- Stop-on-silence delay (often up to 1–3s)
- STT decode time (Whisper)
- Agent reasoning time (LangGraph + LLM, plus any RAG retrieval / MCP tool calls)
- TTS synthesis time (Piper)

Recommended knobs (keep quality good, reduce latency):

1) Recording

- Lower silence stop time (faster turn-taking):
	- `VOICE_AGENT_RECORD_STOP_ON_SILENCE_SECONDS=1.2`

- Trim the trailing silence kept for stop detection (reduces STT time):
	- `VOICE_AGENT_TRIM_TRAILING_SILENCE=true`
	- `VOICE_AGENT_TRAILING_SILENCE_PADDING_SECONDS=0.25`

2) STT (Whisper)

- Keep `VOICE_AGENT_WHISPER_MODEL=medium` for quality.
- For faster decoding with still-good quality:
	- `VOICE_AGENT_WHISPER_BEAM_SIZE=3`
	- `VOICE_AGENT_WHISPER_BEST_OF=1`

- Skip STT on near-silent audio (avoids wasting seconds on empty turns):
	- `VOICE_AGENT_SKIP_STT_ON_SILENCE=true`
	- `VOICE_AGENT_SKIP_STT_RMS_THRESHOLD=0.003`

- Optional faster-whisper performance toggles:
	- `VOICE_AGENT_WHISPER_WITHOUT_TIMESTAMPS=true`
	- `VOICE_AGENT_WHISPER_VAD_FILTER=true`
	- `VOICE_AGENT_STT_USE_NUMPY_WAV=true`

3) LLM / Agent (Ollama + LangGraph)

- Cap output tokens to force shorter answers (faster + less audio playback):
	- `VOICE_AGENT_OLLAMA_NUM_PREDICT=120` (try 96 if you want even shorter)
- Reduce context window if you don't need long history (can help speed/memory):
	- `VOICE_AGENT_OLLAMA_NUM_CTX=2048`
- Keep the model resident longer:
	- `VOICE_AGENT_OLLAMA_KEEP_ALIVE=10m`

4) Warmup (reduces first-turn latency)

- By default the app warms up STT/LLM in a background thread while the greeting is playing.
- Disable if needed:
	- `VOICE_AGENT_WARMUP=false`

## Quick TTS smoke test

This explicitly loads `.env` and synthesizes a WAV:

```
uv run python -c "from dotenv import load_dotenv; load_dotenv(); from tts import synthesize_speech; synthesize_speech('Hello!', 'artifacts/tts_test.wav')"
```

## Roadmap / branches

- `main` — core local pipeline: Whisper → LLM (LangChain) → Piper
- `mcpBranch` **(final / recommended)** — full agentic version: Whisper → RAG (ChromaDB) → LangGraph agent orchestration → MCP tool calling → Piper
