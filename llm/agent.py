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

import json as _json
import os
import re
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Deque, List, Tuple

_SENTENCE_END_RE = re.compile(r"[.!?]")
_SENTENCE_SPLIT_RE = re.compile(r"[.!?](?:\s|$)")


def _trim_to_last_sentence(text: str) -> str:
    """Trim *text* to the last complete sentence.

    When num_predict cuts the LLM output mid-sentence, this drops the
    trailing fragment so the spoken response always sounds natural.
    If the text already ends with sentence-ending punctuation, it is
    returned unchanged.  If no sentence boundary is found at all the
    full text is returned as-is (better than returning nothing).
    """

    text = text.strip()
    if not text:
        return text

    # Already ends cleanly.
    if text[-1] in ".!?":
        return text

    # Find the last sentence-ending punctuation.
    match = None
    for match in _SENTENCE_END_RE.finditer(text):
        pass  # advance to the last match

    if match is not None:
        return text[: match.end()].strip()

    # No sentence boundary at all — return as-is.
    return text


def _env(name: str, default: str) -> str:
    """Read environment variables at runtime.

    Why: `.env` may be loaded after module import depending on the entrypoint.
    Reading env vars at runtime avoids "frozen" defaults.
    """

    return os.environ.get(name, default)


def _build_system_prompt() -> str:
    return (
        "Vous êtes un agent vocal de recouvrement d'une banque, en français (vouvoiement). Ton: formel, empathique, ferme, jamais agressif.\n"
        "Objectif: comprendre la situation, proposer une prochaine étape concrète et sûre (échelonnement, report, paiement partiel) et demander au plus UNE info de clarification non sensible.\n"
        "Style voix: 1 à 2 phrases courtes, une seule idée principale, un seul paragraphe, pas de listes/Markdown.\n"
        "Sécurité: ne jamais demander ni répéter mot de passe, PIN, CVV, numéro de carte, OTP, identifiants; orienter vers canaux officiels pour authentification.\n"
    )


def _parse_optional_int_env(name: str) -> int | None:
    raw = _env(name, "").strip()
    if not raw:
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def _parse_optional_float_env(name: str) -> float | None:
    raw = _env(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


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
    """State-specific prompt guidance."""

    if state == ConversationState.INTRO:
        return "INTRO: expliquer l'objet de l'appel, poser une question ciblée.\n"
    if state == ConversationState.DISCOVERY:
        return (
            "DISCOVERY: recueillir infos minimales non sensibles, proposer un plan.\n"
        )
    if state == ConversationState.NEGOTIATION:
        return "NEGOTIATION: proposer options réalistes, demander confirmation. Pas d'identifiants.\n"
    return "CLOSING: résumer la solution, indiquer la prochaine étape officielle.\n"


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
        "merci",
        "merci beaucoup",
        "c'est tout",
        "ça suffit",
        "bonne journée",
        "au revoir",
        "à bientôt",
    )

    negotiation_signals = (
        "échéancier",
        "paiement en plusieurs fois",
        "paiement fractionné",
        "je ne peux pas payer",
        "je ne peux pas régler",
        "difficulté",
        "retard",
        "impayé",
        "échéance",
        "report",
        "délai",
        "réduire le montant",
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

    def build_raw_messages(self, user_text: str) -> list[dict]:
        """Build messages as plain dicts for direct Ollama API calls."""
        state_for_prompt = _infer_next_state(self.state, user_text)
        system = _build_system_prompt() + _state_guidance(state_for_prompt)
        msgs: list[dict] = [{"role": "system", "content": system}]
        for u, a in self.turns:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_text})
        return msgs


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
    # Model: small local model for low TTFT on consumer GPUs.
    model_name = _env("VOICE_AGENT_OLLAMA_MODEL", "qwen2.5:3b")
    base_url = _env("VOICE_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    keep_alive = _env("VOICE_AGENT_OLLAMA_KEEP_ALIVE", "10m").strip() or None

    # Latency/UX controls (optional):
    # - num_predict: caps output tokens -> faster + shorter spoken responses.
    # - num_ctx: context window; smaller can be faster and uses less memory.
    # - num_gpu/num_thread: advanced knobs; leave unset unless you know why.
    # num_predict: cap output tokens -> shorter speech + lower latency.
    num_predict = _parse_optional_int_env("VOICE_AGENT_OLLAMA_NUM_PREDICT")
    # num_ctx: keep context window moderate -> faster prompt evaluation.
    num_ctx = _parse_optional_int_env("VOICE_AGENT_OLLAMA_NUM_CTX")
    num_gpu = _parse_optional_int_env("VOICE_AGENT_OLLAMA_NUM_GPU")
    num_thread = _parse_optional_int_env("VOICE_AGENT_OLLAMA_NUM_THREAD")
    # temperature: low randomness for consistent, professional answers.
    temperature = _parse_optional_float_env("VOICE_AGENT_OLLAMA_TEMPERATURE")

    # top_p/top_k: stable nucleus + limited candidates for coherent short replies.
    top_p = _parse_optional_float_env("VOICE_AGENT_OLLAMA_TOP_P")
    top_k = _parse_optional_int_env("VOICE_AGENT_OLLAMA_TOP_K")

    # repeat_*: reduce looping/rambling, important for TTS UX.
    repeat_penalty = _parse_optional_float_env("VOICE_AGENT_OLLAMA_REPEAT_PENALTY")
    repeat_last_n = _parse_optional_int_env("VOICE_AGENT_OLLAMA_REPEAT_LAST_N")

    # stop: avoid multi-paragraph responses (keep one paragraph for voice).
    # Use env override as comma-separated values if needed.
    stop_raw = _env("VOICE_AGENT_OLLAMA_STOP", "").strip()
    stop = (
        [s for s in (p.strip() for p in stop_raw.split(",")) if s] if stop_raw else None
    )

    # seed: set only if you need repeatable responses for demos/tests.
    seed = _parse_optional_int_env("VOICE_AGENT_OLLAMA_SEED")

    # mirostat: leave disabled by default for predictable latency.
    mirostat = _parse_optional_int_env("VOICE_AGENT_OLLAMA_MIROSTAT")

    # Good default for phone-like UX if not overridden.
    if num_predict is None:
        num_predict = 48
    if temperature is None:
        temperature = 0.2
    if top_p is None:
        top_p = 0.9
    if top_k is None:
        top_k = 40
    if repeat_penalty is None:
        repeat_penalty = 1.12
    if repeat_last_n is None:
        repeat_last_n = 64
    if mirostat is None:
        mirostat = 0

    # Default stop tokens for voice UX: stop on double newline (paragraph break).
    if stop is None:
        stop = ["\n\n", "<|im_end|>", "<|endoftext|>"]

    return chat_ollama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        num_predict=num_predict,
        num_ctx=num_ctx,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        repeat_last_n=repeat_last_n,
        seed=seed,
        stop=stop,
        mirostat=mirostat,
        num_gpu=num_gpu,
        num_thread=num_thread,
        keep_alive=keep_alive,
    )


def warmup_llm() -> None:
    """Warm up the local LLM (Ollama).

    We keep it light by default: initialize the LangChain client.
    Optionally, you can force a tiny generation request via env var.
    """

    chat = _get_chat_model()

    do_request = _env("VOICE_AGENT_LLM_WARMUP_REQUEST", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not do_request:
        return

    # Tiny request to ensure the Ollama model is resident.
    try:
        from langchain_core.messages import HumanMessage  # type: ignore

        chat.invoke([HumanMessage(content="OK")])
    except Exception:
        # Warmup must never crash the app.
        return


# ---------------------------------------------------------------------------
# Direct Ollama HTTP streaming (bypasses LangChain for lower TTFT)
# ---------------------------------------------------------------------------

_OLLAMA_HTTP: object | None = None


def _get_ollama_http():
    """Persistent httpx.Client for direct Ollama streaming."""
    global _OLLAMA_HTTP
    if _OLLAMA_HTTP is None:
        import httpx  # transitive dep of langchain-ollama

        _OLLAMA_HTTP = httpx.Client(
            timeout=httpx.Timeout(60.0, connect=5.0),
            headers={"Connection": "keep-alive"},
            http1=True,
        )
    return _OLLAMA_HTTP


def call_llm_raw(
    messages: list[dict],
    num_predict: int = 256,
    temperature: float = 0.0,
) -> str:
    """Stateless Ollama call for orchestration layers.

    - Does NOT use ConversationSession (no memory, no state).
    - Does NOT inject this module's voice system prompt.
    - Does NOT trim or post-process output.
    - Uses non-streaming /api/chat (stream=False) so callers can parse full JSON.
    """

    model = _env("VOICE_AGENT_OLLAMA_MODEL", "qwen2.5:3b")
    base_url = _env("VOICE_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    keep_alive = _env("VOICE_AGENT_OLLAMA_KEEP_ALIVE", "10m").strip() or None

    if not isinstance(messages, list) or not messages:
        return ""

    options: dict = {
        "num_predict": int(num_predict),
        "temperature": float(temperature),
    }

    payload: dict = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options,
    }
    if keep_alive:
        payload["keep_alive"] = keep_alive

    url = f"{base_url.rstrip('/')}/api/chat"
    client = _get_ollama_http()

    try:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        content = ((data or {}).get("message") or {}).get("content", "")
        return (content or "").strip()
    except Exception:
        return ""


def _stream_ollama_tokens(messages_raw: list[dict]):
    """Stream tokens from Ollama /api/chat, bypassing LangChain overhead."""
    # Model: default to a small local model for low TTFT.
    model = _env("VOICE_AGENT_OLLAMA_MODEL", "qwen2.5:3b")
    base_url = _env("VOICE_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
    keep_alive = _env("VOICE_AGENT_OLLAMA_KEEP_ALIVE", "10m").strip() or None

    options: dict = {}
    for key, env_name, parser in (
        ("num_predict", "VOICE_AGENT_OLLAMA_NUM_PREDICT", _parse_optional_int_env),
        ("num_ctx", "VOICE_AGENT_OLLAMA_NUM_CTX", _parse_optional_int_env),
        ("top_p", "VOICE_AGENT_OLLAMA_TOP_P", _parse_optional_float_env),
        ("top_k", "VOICE_AGENT_OLLAMA_TOP_K", _parse_optional_int_env),
        (
            "repeat_penalty",
            "VOICE_AGENT_OLLAMA_REPEAT_PENALTY",
            _parse_optional_float_env,
        ),
        ("repeat_last_n", "VOICE_AGENT_OLLAMA_REPEAT_LAST_N", _parse_optional_int_env),
        ("seed", "VOICE_AGENT_OLLAMA_SEED", _parse_optional_int_env),
        ("mirostat", "VOICE_AGENT_OLLAMA_MIROSTAT", _parse_optional_int_env),
        ("num_gpu", "VOICE_AGENT_OLLAMA_NUM_GPU", _parse_optional_int_env),
        ("num_thread", "VOICE_AGENT_OLLAMA_NUM_THREAD", _parse_optional_int_env),
        ("temperature", "VOICE_AGENT_OLLAMA_TEMPERATURE", _parse_optional_float_env),
    ):
        val = parser(env_name)
        if val is not None:
            options[key] = val

    # num_predict: low cap reduces rambling and improves TTS latency.
    options.setdefault("num_predict", 80)
    # temperature: low randomness for stable/professional voice responses.
    options.setdefault("temperature", 0.2)
    # top_p/top_k: keep decoding focused and coherent.
    options.setdefault("top_p", 0.9)
    options.setdefault("top_k", 40)
    # repeat_penalty/repeat_last_n: avoid repeated phrases (annoying in audio).
    options.setdefault("repeat_penalty", 1.12)
    options.setdefault("repeat_last_n", 64)
    # mirostat: disable for predictable latency.
    options.setdefault("mirostat", 0)

    # stop tokens: keep one paragraph (voice-friendly). Can be overridden.
    stop_raw = _env("VOICE_AGENT_OLLAMA_STOP", "").strip()
    if stop_raw:
        options["stop"] = [s for s in (p.strip() for p in stop_raw.split(",")) if s]
    else:
        options.setdefault("stop", ["\n\n"])

    payload: dict = {
        "model": model,
        "messages": messages_raw,
        "stream": True,
        "options": options,
    }
    if keep_alive:
        payload["keep_alive"] = keep_alive

    url = f"{base_url.rstrip('/')}/api/chat"
    client = _get_ollama_http()
    import time

    _t0 = time.perf_counter()
    first_token = True

    with client.stream("POST", url, json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            # inside the for loop, at the top:
            if first_token:
                print(f"[DIAG] Time to first token: {time.perf_counter() - _t0:.3f}s")
                first_token = False

            if not line:
                continue
            data = _json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                if first_token:
                    print(f"[DIAG] TTFT: {time.perf_counter() - _t0:.3f}s")
                    first_token = False
                yield token
            if data.get("done", False):
                # Log total generation time
                print(
                    f"[DIAG] Total gen: {time.perf_counter() - _t0:.3f}s | "
                    f"prompt_eval: {data.get('prompt_eval_duration', 0) / 1e9:.3f}s | "
                    f"eval: {data.get('eval_duration', 0) / 1e9:.3f}s"
                )
                break


def generate_ai_response(user_text: str) -> str:
    """Generate a single LLM response and update conversation state.

    This is the main public API for the LLM module.
    """

    text = (user_text or "").strip()
    if not text:
        return "Je n’ai pas bien compris. En quoi puis-je vous aider ?"

    chat = _get_chat_model()

    session = _get_session()
    messages = session.build_messages(text)

    try:
        response = chat.invoke(messages)
        content = getattr(response, "content", "")
        assistant_text = _trim_to_last_sentence((content or "").strip())
        if not assistant_text:
            assistant_text = "Je suis désolé, je ne parviens pas à formuler une réponse pour le moment."
    except Exception:
        assistant_text = (
            "Je suis désolé, je ne parviens pas à formuler une réponse pour le moment."
        )

    session.append_turn(text, assistant_text)
    return assistant_text


def stream_ai_response_sentences(user_text: str):
    """Stream LLM response, yielding complete sentences as they form.

    Uses direct Ollama HTTP streaming (bypassing LangChain) for minimal
    time-to-first-token latency.
    """

    text = (user_text or "").strip()
    if not text:
        yield "Je n'ai pas bien compris. En quoi puis-je vous aider ?"
        return

    session = _get_session()
    messages = session.build_raw_messages(text)

    buffer = ""
    full_response = ""
    _MAX_BUFFER = 200  # Force-yield at a word boundary if no punctuation found

    try:
        for token in _stream_ollama_tokens(messages):
            buffer += token
            full_response += token

            # Extract complete sentences from the buffer.
            while True:
                match = _SENTENCE_SPLIT_RE.search(buffer)
                if match:
                    end = match.start() + 1  # include punctuation, not trailing space
                    sentence = buffer[:end].strip()
                    buffer = buffer[end:].lstrip()
                    if sentence:
                        yield sentence
                    continue
                if len(buffer) > _MAX_BUFFER:
                    # Force-yield without cutting mid-word. Keep the remainder.
                    cut = buffer.rfind(" ")
                    if cut > 0:
                        forced = buffer[:cut].strip()
                        buffer = buffer[cut:].lstrip()
                    else:
                        forced = buffer.strip()
                        buffer = ""
                    if forced:
                        yield forced
                break

        # Yield remaining text only if it contains a complete sentence.
        remaining = buffer.strip()
        if remaining:
            last_end = -1
            for match in _SENTENCE_END_RE.finditer(remaining):
                last_end = match.end()
            if last_end > 0:
                tail_sentence = remaining[:last_end].strip()
                if tail_sentence:
                    yield tail_sentence

        # Update conversation session.
        assistant_text = _trim_to_last_sentence(full_response.strip())
        if not assistant_text:
            assistant_text = (
                full_response.strip()
                or "Je suis désolé, je ne parviens pas à formuler une réponse pour le moment."
            )
        session.append_turn(text, assistant_text)

    except Exception:
        remaining = buffer.strip()
        if remaining:
            last_end = -1
            for match in _SENTENCE_END_RE.finditer(remaining):
                last_end = match.end()
            if last_end > 0:
                tail_sentence = remaining[:last_end].strip()
                if tail_sentence:
                    yield tail_sentence
        final = (
            full_response.strip()
            or "Je suis désolé, je ne parviens pas à formuler une réponse pour le moment."
        )
        if not full_response.strip():
            yield final
        session.append_turn(text, final)


def stream_raw_sentences(user_content: str, system_content: str = ""):
    text = (user_content or "").strip()
    if not text:
        return
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": text})

    buffer = ""
    _MAX_BUFFER = 200  # Force-yield at a word boundary if no punctuation found

    try:
        for token in _stream_ollama_tokens(messages):
            buffer += token

            # Extract complete sentences from the buffer.
            while True:
                match = _SENTENCE_SPLIT_RE.search(buffer)
                if match:
                    end = match.start() + 1  # include punctuation, not trailing space
                    sentence = buffer[:end].strip()
                    buffer = buffer[end:].lstrip()
                    if sentence:
                        yield sentence
                    continue

                if len(buffer) > _MAX_BUFFER:
                    # Force-yield without cutting mid-word. Keep the remainder.
                    cut = buffer.rfind(" ")
                    if cut > 0:
                        forced = buffer[:cut].strip()
                        buffer = buffer[cut:].lstrip()
                    else:
                        forced = buffer.strip()
                        buffer = ""
                    if forced:
                        yield forced
                break

        # Flush remaining text. Prefer a complete sentence; otherwise yield
        # the remaining fragment so TTS doesn't get stuck with no output.
        remaining = buffer.strip()
        if remaining:
            last_end = -1
            for match in _SENTENCE_END_RE.finditer(remaining):
                last_end = match.end()
            if last_end > 0:
                tail_sentence = remaining[:last_end].strip()
                if tail_sentence:
                    yield tail_sentence
            else:
                yield remaining

    except Exception:
        # Best-effort: flush a complete sentence if possible.
        remaining = buffer.strip()
        if remaining:
            last_end = -1
            for match in _SENTENCE_END_RE.finditer(remaining):
                last_end = match.end()
            if last_end > 0:
                tail_sentence = remaining[:last_end].strip()
                if tail_sentence:
                    yield tail_sentence
            else:
                yield remaining
