"""FastAPI layer for the existing local voice assistant.

Design goals
------------
- Do NOT start microphone streaming: never call `VoiceAgent.start()`.
- Keep architecture intact: reuse existing LLM functions in `llm/agent.py`.
- Production-safe init: single global `VoiceAgent` instance + FastAPI lifespan warmups.
- Thread-safe: serialize access to shared in-process conversation memory.

Run:
    uvicorn api:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from threading import Lock
from typing import Iterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from llm.agent import generate_ai_response, stream_ai_response_sentences, warmup_llm
from main import VoiceAgent
from stt import warmup_stt
from tts import warmup_tts


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User message")


class ChatResponse(BaseModel):
    response: str


class HealthResponse(BaseModel):
    status: str
    service: str


# ---------------------------------------------------------------------------
# Singleton agent + locks
# ---------------------------------------------------------------------------

_agent_init_lock = Lock()
_chat_lock = Lock()
_agent: VoiceAgent | None = None


def _get_or_create_agent() -> VoiceAgent:
    """Return the single global VoiceAgent instance.

    Important: creating a VoiceAgent must NOT start mic streaming.
    We therefore never call `VoiceAgent.start()` in this file.
    """

    global _agent
    if _agent is None:
        with _agent_init_lock:
            if _agent is None:
                _agent = VoiceAgent()
    return _agent


# ---------------------------------------------------------------------------
# App lifespan (safe startup initialization)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Ensure config is loaded for warmups and model construction.
    load_dotenv(override=True)

    # Create singleton once (does not start microphone streaming).
    _get_or_create_agent()

    # Warm up heavy components (best-effort; should never prevent startup).
    try:
        warmup_stt()
    except Exception:
        pass

    try:
        warmup_llm()
    except Exception:
        pass

    try:
        warmup_tts()
    except Exception:
        pass

    yield


app = FastAPI(title="Voice Agent API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service="voice-agent")


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    # Ensure singleton exists even if called before lifespan completes.
    _get_or_create_agent()

    # The LLM module keeps conversation memory in-process.
    # Serialize access so concurrent requests don't interleave and corrupt state.
    with _chat_lock:
        try:
            response_text = generate_ai_response(text)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"LLM error: {exc}") from exc

    return ChatResponse(response=response_text)


@app.post("/chat/stream")
def chat_stream(payload: ChatRequest) -> StreamingResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    _get_or_create_agent()

    def _iter_sentences() -> Iterator[bytes]:
        # Server-Sent Events (SSE): stream each sentence as a separate event.
        # Note: we intentionally avoid locking the full generation loop for latency.
        try:
            generator = stream_ai_response_sentences(text)
            for sentence in generator:
                s = (sentence or "").strip()
                if not s:
                    continue
                # Keep each SSE event single-line.
                s = " ".join(s.splitlines()).strip()
                yield f"data: {s}\n\n".encode("utf-8")
        except Exception as exc:
            yield f"data: [error] {exc}\n\n".encode("utf-8")

    return StreamingResponse(
        _iter_sentences(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Useful when running behind some proxies (e.g., nginx) to disable buffering.
            "X-Accel-Buffering": "no",
        },
    )


# Helpful for local debugging without uvicorn, but optional.
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        reload=True,
    )
