"""LLM agent (local) using LangChain + Ollama.

This module implements a small, modular boundary:

    generate_ai_response(user_text: str) -> str

Design goals
-----------
- Local-only: uses an Ollama server running on the same machine (default).
- Minimal surface area: a single function and a cached model initializer.
- Prompt engineering documented: comments explain WHY prompts are structured
  as they are, and what failure modes they mitigate.

Background (academic-style)
--------------------------
Ollama serves LLMs locally over HTTP (typically http://localhost:11434).
LangChain provides abstractions for:
- Chat model invocation
- Message role separation (system/human/assistant)
- Optional tooling/memory (not used yet to keep this simple)

We intentionally avoid agent tool-use frameworks at this stage.
"""

from __future__ import annotations

import os
from functools import lru_cache


def _env(name: str, default: str) -> str:
    """Read environment variables at runtime.

    Why: `.env` may be loaded after module import depending on the entrypoint.
    Reading env vars at runtime avoids "frozen" defaults.
    """

    return os.environ.get(name, default)


def _build_system_prompt() -> str:
    """System prompt for a banking voice assistant.

    Prompt engineering notes
    ------------------------
    1) Role clarity: The system message defines the assistant persona and domain.
       This reduces "role drift" and keeps outputs consistent.

    2) Safety + compliance: In banking contexts, the assistant should avoid
       requesting or exposing sensitive data. We explicitly instruct it to:
       - not ask for full card numbers, CVV, or passwords
       - encourage secure channels for authentication
       - provide general guidance and next steps

    3) Voice-first style: Voice assistants must be concise, confirm intent, and
       avoid long paragraphs. We request short sentences and clarifying questions.

    4) Determinism: We also instruct the model to be structured, which improves
       reliability in downstream voice UX (TTS) and reduces hallucinated steps.
    """

    return (
        "You are a helpful banking voice assistant operating locally on the user's computer.\n"
        "Your job: answer questions about everyday banking topics (accounts, cards, transfers, fees, budgeting) "
        "and guide the user through safe next steps.\n\n"
        "Communication style (voice-first):\n"
        "- Be concise (1–5 short sentences).\n"
        "- Ask 1 clarifying question if the request is ambiguous.\n"
        "- Prefer step-by-step guidance when appropriate.\n\n"
        "Banking safety rules (must follow):\n"
        "- Never ask for or repeat passwords, PINs, CVV, full card numbers, or one-time codes.\n"
        "- If authentication is needed, direct the user to their bank app/website or official support.\n"
        "- If the user requests a high-risk action (e.g., cancel card, dispute charge), provide safe guidance "
        "and recommend verifying via official channels.\n\n"
        "Grounding:\n"
        "- If you are unsure, say so briefly and suggest what information is needed.\n"
    )


@lru_cache(maxsize=1)
def _get_chat_model():
    """Create and cache the ChatOllama model (singleton-style).

    Academic note:
    Creating the model object can involve configuration and connection setup.
    Caching avoids repeated initialization overhead for each user utterance.

    Implementation note:
    ChatOllama has moved packages across LangChain versions.
    We first try the modern `langchain_ollama` package, then fall back to older
    `langchain_community` locations when available.
    """

    chat_ollama = None
    import_error: Exception | None = None

    try:
        from langchain_ollama import ChatOllama as _ChatOllama  # type: ignore

        chat_ollama = _ChatOllama
    except Exception as exc:  # pragma: no cover
        import_error = exc

    if chat_ollama is None:
        try:
            from langchain_community.chat_models import ChatOllama as _ChatOllama  # type: ignore

            chat_ollama = _ChatOllama
            import_error = None
        except Exception as exc:  # pragma: no cover
            import_error = exc

    if chat_ollama is None:  # pragma: no cover
        raise RuntimeError(
            "LangChain ChatOllama is not available. Install `langchain` and `langchain-ollama` (recommended)."
        ) from import_error

    # Temperature: lower values -> more consistent answers.
    # For voice assistants, consistency matters more than creative variation.
    model_name = _env("VOICE_AGENT_OLLAMA_MODEL", "llama3.2")
    base_url = _env("VOICE_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    return chat_ollama(
        model=model_name,
        base_url=base_url,
        temperature=0.2,
    )


def generate_ai_response(user_text: str) -> str:
    """Generate an AI response to the user's text (local LLM via Ollama).

    This is intentionally minimal: no tools, no memory, no external APIs.

    Parameters
    ----------
    user_text:
        The user's transcribed utterance.

    Returns
    -------
    str
        Assistant response text.

    Prompt engineering notes
    ------------------------
    We separate messages by role:
    - System message: persistent rules and persona.
    - Human message: the user's request.

    This separation is important because chat models are trained to treat the
    system role as higher priority. It reduces prompt injection risk and keeps
    banking safety constraints applied across turns.
    """

    text = (user_text or "").strip()
    if not text:
        return "I didn't catch that. What would you like help with?"

    try:
        from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "LangChain core messages are not available. Install `langchain` (and `langchain-core` if needed)."
        ) from exc

    chat = _get_chat_model()

    messages = [
        SystemMessage(content=_build_system_prompt()),
        HumanMessage(content=text),
    ]

    # `.invoke()` is the simplest LangChain execution method for a single turn.
    # It returns an AIMessage-like object with `.content`.
    response = chat.invoke(messages)

    content = getattr(response, "content", "")
    return (content or "").strip() or "Sorry — I couldn't generate a response."
