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
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Deque, List, Tuple


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


class ConversationState(str, Enum):
    """Explicit, lightweight state machine for a banking call."""

    INTRO = "INTRO"
    DISCOVERY = "DISCOVERY"
    NEGOTIATION = "NEGOTIATION"
    CLOSING = "CLOSING"


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def _parse_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _state_guidance(state: ConversationState) -> str:
    """State-specific prompt guidance.

    This is intentionally short and voice-oriented.
    """

    if state == ConversationState.INTRO:
        return (
            "Call state: INTRO.\n"
            "Goal: establish why you're calling and what the user needs help with.\n"
            "Next action: ask one focused question to route the request (e.g., account type/topic).\n"
            "Tone: professional, calm, brief.\n"
        )

    if state == ConversationState.DISCOVERY:
        return (
            "Call state: DISCOVERY.\n"
            "Goal: collect only the minimum non-sensitive details needed to help.\n"
            "Next action: ask at most one clarifying question (amount range, due date, product type), then give a short plan.\n"
            "Tone: supportive, practical.\n"
        )

    if state == ConversationState.NEGOTIATION:
        return (
            "Call state: NEGOTIATION.\n"
            "Goal: propose realistic options (installments, hardship support, payment timing) and confirm constraints.\n"
            "Next action: propose one option + ask for confirmation or a budget number (without requesting sensitive credentials).\n"
            "Tone: collaborative and solution-focused.\n"
        )

    return (
        "Call state: CLOSING.\n"
        "Goal: summarize what was decided and the next safe step via official channels.\n"
        "Next action: confirm if they need anything else, then provide a brief closing line.\n"
        "Tone: warm and concise.\n"
    )


def _infer_next_state(
    current: ConversationState,
    user_text: str,
    assistant_text: str | None = None,
) -> ConversationState:
    """Heuristic state transitions.

    We keep this rule-based for predictability and to avoid adding tool/agent complexity.
    """

    u = _normalize(user_text)
    a = _normalize(assistant_text or "")

    closing_signals = (
        "thanks",
        "thank you",
        "that helps",
        "that's all",
        "that is all",
        "no that's all",
        "no thats all",
        "goodbye",
        "bye",
        "see you",
        "see you later",
    )

    negotiation_signals = (
        "installment",
        "installments",
        "payment plan",
        "plan",
        "can't pay",
        "cannot pay",
        "struggling",
        "hardship",
        "late fee",
        "overdue",
        "past due",
        "minimum payment",
        "due date",
        "defer",
        "deferral",
        "reduce payment",
    )

    if any(s in u for s in closing_signals):
        return ConversationState.CLOSING

    if current == ConversationState.INTRO:
        # After the first user response, we generally move into discovery.
        if u:
            return ConversationState.DISCOVERY

    if current in (ConversationState.DISCOVERY, ConversationState.INTRO):
        if any(s in u for s in negotiation_signals):
            return ConversationState.NEGOTIATION

    if current == ConversationState.NEGOTIATION:
        # If the assistant has summarized an option and the user is agreeable, close.
        agree = ("yes", "ok", "okay", "sounds good", "that works", "agree")
        if any(s in u for s in agree) and (
            "next step" in a or "we can" in a or "option" in a
        ):
            return ConversationState.CLOSING

    return current


@dataclass
class ConversationSession:
    """In-process state + bounded memory for one call.

    Memory resets when the program exits (call ends). A manual reset function
    is also provided for future reuse.
    """

    max_turns: int = 5
    state: ConversationState = ConversationState.INTRO
    # Store (user, assistant) turns. We keep turns (not raw messages) so we can
    # trim in pairs.
    turns: Deque[Tuple[str, str]] = field(default_factory=deque)

    def reset(self) -> None:
        self.state = ConversationState.INTRO
        self.turns.clear()

    def append_turn(self, user_text: str, assistant_text: str) -> None:
        self.turns.append((user_text, assistant_text))
        while len(self.turns) > max(self.max_turns, 1):
            self.turns.popleft()
        self.state = _infer_next_state(self.state, user_text, assistant_text)

    def build_messages(self, user_text: str):
        """Build chat messages including bounded history + state guidance."""

        try:
            from langchain_core.messages import (  # type: ignore
                AIMessage,
                HumanMessage,
                SystemMessage,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "LangChain core messages are not available. Install `langchain` (and `langchain-core` if needed)."
            ) from exc

        # State can also transition based on the new user text.
        state_for_prompt = _infer_next_state(self.state, user_text)

        system = _build_system_prompt() + "\n" + _state_guidance(state_for_prompt)

        messages: List[object] = [SystemMessage(content=system)]

        # Include last N turns as alternating user/assistant messages.
        for u, a in self.turns:
            messages.append(HumanMessage(content=u))
            messages.append(AIMessage(content=a))

        messages.append(HumanMessage(content=user_text))
        return messages


_SESSION: ConversationSession | None = None


def _get_session() -> ConversationSession:
    global _SESSION
    if _SESSION is None:
        # Requirement: keep the last N turns (typical 4–6). Default to 5.
        max_turns = _parse_int_env("VOICE_AGENT_MEMORY_TURNS", 5)
        max_turns = max(1, min(max_turns, 12))
        _SESSION = ConversationSession(max_turns=max_turns)
    return _SESSION


def reset_conversation() -> None:
    """Reset conversation state/memory for a new call."""

    global _SESSION
    if _SESSION is not None:
        _SESSION.reset()
    _SESSION = None


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

    chat = _get_chat_model()

    session = _get_session()
    messages = session.build_messages(text)

    # `.invoke()` is the simplest LangChain execution method for a single turn.
    # It returns an AIMessage-like object with `.content`.
    response = chat.invoke(messages)

    content = getattr(response, "content", "")

    final_text = (content or "").strip() or "Sorry — I couldn't generate a response."
    session.append_turn(text, final_text)
    return final_text
