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
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from time import strftime

from dotenv import load_dotenv

from llm import stream_raw_sentences, warmup_llm
from llm.langgraph_agent import (
    classify_client_profile,
    run_voice_agent_prepare,
    verify_identity,
)
from stt import StreamingWhisper, warmup_stt
from stt.streaming_whisper import VadConfig
from tts import speak_streaming, warmup_tts


OUTBOUND_GREETING = "Bonjour, je suis l'assistant de recouvrement de votre établissement bancaire. Je vous contacte aujourd'hui concernant des échéances impayées sur votre compte. Afin de traiter votre dossier, j'ai besoin de vérifier votre identité. "


def _log(message: str) -> None:
    print(f"[{strftime('%H:%M:%S')}] {message}")


def format_phone_tunisian(phone: str | int | None) -> str:
    """Formate un numero tunisien pour la lecture TTS.

    '21623766755' -> '216 23 766 755'
    On cible le format 216 + 2 chiffres + 3 chiffres + 3 chiffres.
    """

    if phone is None:
        return ""
    digits = re.sub(r"\D", "", str(phone))
    if digits.startswith("216") and len(digits) == 11:
        cc = digits[0:3]  # 216
        p1 = digits[3:5]  # 23
        p2 = digits[5:8]  # 766
        p3 = digits[8:11]  # 755
        return f"{cc} {p1} {p2} {p3}"
    # fallback : groupes de 2
    return " ".join(digits[i : i + 2] for i in range(0, len(digits), 2))


def _tts_phone_digits_grouped(phone: str | int | None) -> str:
    """Render Tunisian phone number as grouped digits for robust TTS.

    Example:
    21673445566 -> "2 1 6, 7 3, 4 4 5, 5 6 6"
    """

    if phone is None:
        return ""

    digits = re.sub(r"\D", "", str(phone))
    groups: list[str]
    if digits.startswith("216") and len(digits) == 11:
        groups = [digits[0:3], digits[3:5], digits[5:8], digits[8:11]]
    elif len(digits) == 8:
        groups = [digits[0:2], digits[2:5], digits[5:8]]
    else:
        groups = [digits[i : i + 2] for i in range(0, len(digits), 2)]

    return ", ".join(" ".join(ch for ch in group) for group in groups if group)


def normalize_tunisian_phones_in_text(text: str) -> str:
    """Normalize Tunisian phone numbers found in free text before TTS."""

    pattern = re.compile(r"(?<!\d)(?:\+?216[\s\-\.]?)((?:\d[\s\-\.]?){7}\d)(?!\d)")

    def _repl(match: re.Match[str]) -> str:
        local_digits = re.sub(r"\D", "", match.group(1))
        return _tts_phone_digits_grouped(f"216{local_digits}")

    return pattern.sub(_repl, text)


def normalize_tnd_amounts_in_text(text: str) -> str:
    """Normalize Tunisian currency amounts for natural French TTS.

    Example:
    "444.44 DT" -> "444 dinars virgule 44"
    "1 DT" -> "1 dinar"
    "148 DT" -> "148 dinars"
    """

    # Accept up to 4 decimals and optional whitespace after the separator.
    # Why: sentence splitters sometimes emit "1562." then "49DT" or even
    # "1562. 49 DT"; we still want to normalize it into a single spoken amount.
    pattern = re.compile(
        r"(?<!\d)(\d+(?:[\.,]\s*\d{1,4})?)\s*(?:dt|dinar(?:s)?(?:\s+tunisien(?:s)?)?)\b",
        flags=re.IGNORECASE,
    )

    def _repl(match: re.Match[str]) -> str:
        raw_amount = re.sub(r"\s+", "", (match.group(1) or "")).replace(",", ".")
        try:
            value = round(float(raw_amount), 2)
        except ValueError:
            return match.group(0)

        whole = int(value)
        cents = int(round((value - whole) * 100))
        if cents == 100:
            whole += 1
            cents = 0

        unit = "dinar" if whole == 1 else "dinars"
        if cents > 0:
            return f"{whole} {unit} virgule {cents:02d}"
        return f"{whole} {unit}"

    return pattern.sub(_repl, text)


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

    # Normalize Tunisian phone numbers so TTS reads them as grouped numbers.
    value = normalize_tunisian_phones_in_text(value)
    # Normalize currency so TTS says "dinar(s) virgule xx" instead of "DT".
    value = normalize_tnd_amounts_in_text(value)

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


def _mcp(tool_name: str, arguments: dict) -> dict:
    """Single entry point for all MCP tool calls from main.py."""
    from llm.mcp_client import ACMMCPClient
    mcp = ACMMCPClient.get_instance()
    return mcp.call_tool(tool_name, arguments)


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
        # Change this value to test with different customers
        self._customer_id: int = int(os.environ.get("VOICE_AGENT_CUSTOMER_ID", 1002))
        self._session_id: str = str(uuid.uuid4())
        self._turn_number: int = 0
        self._verified: bool = False
        self._verification_attempts: int = 0
        self._awaiting_dob: bool = False
        self._awaiting_name_confirmation: bool = False
        self._name_confirmation_attempts: int = 0
        self._negotiation_active: bool = False
        self._negotiation_profile: dict | None = None
        self._client_info: dict | None = None
        self._negotiation_step: str = "present_debt"
        self._proposed_installments: int = 0
        self._proposed_amount: float = 0.0
        self._proposed_date: str = ""
        self._negotiation_refusals: int = 0
        self._client_reason: str = "unknown"
        self._client_reason_raw: str = ""

        self._claim_active: bool = False
        self._claim_step: str = ""  # "await_subject" | "await_body"
        self._claim_subject: str = ""
        self._claim_body: str = ""

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

        # Récupérer le nom du client AVANT le greeting
        client_name = ""
        try:
            info = _mcp("get_client_info", {"customer_id": self._customer_id})
            if info.get("found"):
                self._client_info = info
                client_name = str(info.get("customer_name", "")).strip()
        except Exception:
            pass

        # Mark call as in-progress.
        try:
            _mcp("set_call_status", {
                "customer_id": self._customer_id,
                "status": "IN_CALL",
                "session_id": self._session_id,
                "notes": "Appel en cours",
            })
        except Exception as exc:
            _log(f"[CALL_STATUS] set IN_CALL failed: {exc}")

        # Greeting personnalisé avec le nom
        civilite = "Monsieur"  # ou logique selon le nom

        if client_name:
            greeting = (
                f"Bonjour, je suis l'assistant de recouvrement "
                f"de votre établissement bancaire. "
                f"Je vous contacte au sujet de votre compte. "
                f"Ai-je bien {civilite} {client_name} en ligne ?"
            )
        else:
            greeting = OUTBOUND_GREETING

        # Pause during greeting playback to avoid STT hearing the assistant.
        self._stt.pause()
        try:
            self.speak(greeting)
            self._session_summary["turns"].append(
                {
                    "user_text": None,
                    "assistant_text": greeting,
                    "event": "greeting",
                }
            )
        finally:
            self._stt.resume()

        # Attendre confirmation identité
        if client_name:
            self._awaiting_name_confirmation = True
        else:
            self._awaiting_dob = True

        # Pre-load client profile in background for faster negotiation start.
        Thread(target=self._preload_client_info, daemon=True).start()

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
        if len(normalized) < 3 or all(c in ".,!?…- " for c in normalized):
            return

        NOISE_TOKENS = {
            "mouh",
            "mh",
            "hm",
            "hmm",
            "euh",
            "ah",
            "oh",
            "hein",
            "bah",
            "ben",
            "pff",
            "voila",
            "ouais ouais",
            "mall",
            "mal",
            "mmm",
            "allo",
            "allô",
            "mhm",
            "ouais",
            "heu",
            "euh voila",
            "nan",
            "bof",
        }
        if normalized in NOISE_TOKENS:
            _log(f"[STT] Filtered ASR noise: {normalized!r}")
            return
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

    def _preload_client_info(self) -> None:
        try:
            info = _mcp("get_client_info", {"customer_id": self._customer_id})
            if info.get("found"):
                self._client_info = info
                self._negotiation_profile = classify_client_profile(info)
                _log(
                    f"[NEGO] Profile: {self._negotiation_profile['profile']}, "
                    f"max_installments={self._negotiation_profile['max_installments']}"
                )
        except Exception as exc:
            _log(f"[NEGO] Preload failed: {exc}")

    def _respond_worker(self, user_text: str) -> None:
        """Background worker: streamed LLM → TTS → playback.

        Streams LLM tokens, detects sentence boundaries, and speaks each
        sentence immediately so TTS playback overlaps with LLM generation.
        """
        import time as _time

        try:
            _log(f"User: {user_text!r}")
            self._session_summary["turns"].append({"user_text": user_text})
            self._turn_number += 1
            current_turn = self._turn_number

            if not self._verified:
                # Nouvelle étape : confirmation nom avant DOB
                if self._awaiting_name_confirmation:
                    self._handle_name_confirmation(user_text, current_turn)
                    return

                if self._awaiting_dob:
                    verification = verify_identity(
                        transcript=user_text,
                        customer_id=self._customer_id,
                        attempts=self._verification_attempts,
                    )
                    self._verification_attempts = int(
                        verification.get("attempts", self._verification_attempts)
                    )

                    if verification.get("verified"):
                        self._verified = True
                        self._awaiting_dob = False
                        self._negotiation_active = True
                        self._negotiation_step = "present_debt"
                        verification_msg = "Merci, votre identité a bien été vérifiée."
                        self._stt.pause()
                        try:
                            self.speak(verification_msg)
                            nego_intro = self._start_negotiation_turn()
                        finally:
                            self._stt.resume()
                        self._log_call_event(
                            transcript=user_text,
                            intent="identity_verification",
                            outcome="verified",
                            agent_decision=(
                                f"{verification_msg} {(nego_intro or '').strip()}"
                            ).strip(),
                            turn_number=current_turn,
                        )
                        return

                    if verification.get("should_hangup"):
                        hangup_msg = (
                            "Je suis désolé, je ne peux pas vérifier votre identité. "
                            "Cette communication va prendre fin. Au revoir."
                        )
                        self._stt.pause()
                        try:
                            self.speak(hangup_msg)
                        finally:
                            self._stt.resume()
                        self._log_call_event(
                            transcript=user_text,
                            intent="identity_verification",
                            outcome="hangup",
                            agent_decision=hangup_msg,
                            turn_number=current_turn,
                        )
                        Thread(target=self.shutdown, daemon=True).start()
                        return

                    if self._verification_attempts <= 1:
                        retry_message = "Je suis désolé, cette date ne correspond pas à nos enregistrements. Pouvez-vous réessayer ?"
                    else:
                        retry_message = "Ce n'est toujours pas correct. Il vous reste une dernière tentative."

                    self._stt.pause()
                    try:
                        self.speak(retry_message)
                    finally:
                        self._stt.resume()

                    self._log_call_event(
                        transcript=user_text,
                        intent="identity_verification",
                        outcome="retry",
                        agent_decision=retry_message,
                        turn_number=current_turn,
                    )

                    self._awaiting_dob = True
                    return

                prompt_msg = "Pour vérifier votre identité, pouvez-vous me donner votre date de naissance ?"
                self._stt.pause()
                try:
                    self.speak(prompt_msg)
                finally:
                    self._stt.resume()
                self._log_call_event(
                    transcript=user_text,
                    intent="identity_verification",
                    outcome="prompt_dob",
                    agent_decision=prompt_msg,
                    turn_number=current_turn,
                )
                self._awaiting_dob = True
                return

            # Claim detection — triggered even during active negotiation
            if self._claim_active:
                self._handle_claim_turn(user_text, current_turn)
                return

            # Claim trigger detection — check if client wants to file a claim
            if self._verified and not self._claim_active:
                claim_keywords = [
                    "réclamation",
                    "réclamer",
                    "signaler un problème",
                    "j'ai déjà payé",
                    "erreur sur",
                    "contester",
                    "plainte",
                    "problème avec mon compte",
                    "paiement non enregistré",
                    "décalage",
                    "restructuration",
                ]
                user_lower = user_text.lower()
                if any(kw in user_lower for kw in claim_keywords):
                    self._claim_active = True
                    self._claim_step = "await_subject"
                    self._stt.pause()
                    try:
                        self.speak(
                            "Je comprends que vous souhaitez déposer une réclamation. "
                            "Quel est le sujet ? Par exemple : erreur sur montant, "
                            "paiement non enregistré, demande de décalage, "
                            "restructuration, comportement inapproprié, ou autre ?"
                        )
                    finally:
                        self._stt.resume()
                    self._log_call_event(
                        transcript=user_text,
                        intent="claim_trigger",
                        outcome="claim_started",
                        agent_decision="Claim flow initiated",
                        turn_number=current_turn,
                    )
                    return

            if self._verified and self._negotiation_active:
                self._handle_negotiation_turn(user_text, current_turn)
                return

            t0 = _time.monotonic()
            first_audio_time = None
            full_response_parts = []

            # Prepare context via LangGraph (decision + RAG + tools only).
            state = run_voice_agent_prepare(user_text, customer_id=self._customer_id)
            print(
                f"[RAG DEBUG] rag_context: {repr(state.get('rag_context', '')[:300])}"
            )

            route = state.get("route", "general")
            rag = (state.get("rag_context") or "").strip()
            tool_res = state.get("tool_results") or []
            secondary = state.get("secondary")
            transcript_norm = " ".join(state["transcript"].lower().split()).rstrip(
                ".!?"
            )

            # Detect ASR correction pattern: "je suis désolé, j'ai dit X" -> extract X as real question
            import re as _re

            correction_match = _re.search(
                r"(?:je suis désolé[,.]?\s*)?j['\s]ai dit\s+(.+)",
                state["transcript"].lower(),
            )
            if correction_match:
                corrected = correction_match.group(1).strip()
                # Re-route as a general banking question with the corrected term
                route = "general"
                user_msg_override = corrected
            else:
                user_msg_override = None

            if route == "ack":
                self.speak("Très bien, je reste à votre disposition.")
                return

            ACK_TOKENS = {
                "ok",
                "okay",
                "d'accord",
                "dacord",
                "je vois",
                "compris",
                "entendu",
                "très bien",
                "parfait",
            }

            if route == "ack" or transcript_norm in ACK_TOKENS:
                system_msg = (
                    "Tu es un agent de recouvrement bancaire tunisien au téléphone. "
                    "Le client dit juste merci ou ok. Réponds en UNE seule phrase courte et naturelle. "
                    "Exemples: 'De rien, bonne journée.' ou 'Avec plaisir.' ou 'Je vous en prie.' "
                    "Ne pose pas de question. Ne propose pas d'aide supplémentaire. Maximum 5 mots."
                    "Parle comme un conseiller bancaire humain au téléphone. "
                    "Utilise un ton naturel, simple, et direct. "
                    "Évite les définitions académiques."
                )
                user_msg = state["transcript"]
                if user_msg_override is not None:
                    user_msg = user_msg_override

            elif route == "casual":
                system_msg = (
                    "Parle comme un conseiller bancaire humain au téléphone. "
                    "Utilise un ton naturel, simple, et direct. "
                    "Évite les définitions académiques."
                )
                user_msg = state["transcript"]
                if user_msg_override is not None:
                    user_msg = user_msg_override

            elif tool_res:
                if not any(r.get("ok") for r in tool_res):
                    # All tools failed
                    for sentence in stream_raw_sentences(
                        "erreur outil",
                        system_content="Dis: 'Je n arrive pas à récupérer vos données pour le moment.'",
                    ):
                        self.speak(sentence)
                    return
                greeting = "Bien sûr ! " if secondary == "casual" else ""
                system_msg = (
                    "Tu es un agent de recouvrement bancaire tunisien au téléphone. "
                    "Tu as accès au dossier complet du client. "
                    "Réponds UNIQUEMENT à la question posée par le client, en utilisant "
                    "les données exactes de son dossier. "
                    "Ne donne qu'une seule information si le client demande une seule information. "
                    "N'ajoute jamais des détails non demandés (pas de numéro de compte, pas de téléphone, pas d'email, pas de solde) sauf si la question le demande explicitement. "
                    "Ne parle des impayés QUE si le client pose une question sur ses impayés, "
                    "son solde, ou son compte. "
                    "Si le client demande son numéro de téléphone, donne-lui son numéro. "
                    "Si le client demande son email, donne-lui son email. "
                    "Le numéro de téléphone est déjà formaté avec des espaces, "
                    "lis-le chiffre par chiffre en respectant les groupes : "
                    "'216 23 766 755' se lit 'deux cent seize, vingt-trois, sept cent soixante-six, sept cent cinquante-cinq'. "
                    "Sois naturel, direct, et concis. Maximum 2 phrases."
                )
                client = tool_res[0] if tool_res else {}
                user_msg = (
                    f"{'Bonjour ' + client.get('customer_name', '') + '. ' if greeting else ''}"
                    f"Question du client: {state['transcript']}\n\n"
                    f"Dossier complet du client:\n"
                    f"- Nom: {client.get('customer_name', '')}\n"
                    f"- Téléphone: {format_phone_tunisian(client.get('telephone_1', ''))}\n"
                    f"- Email: {client.get('email', '')}\n"
                    f"- Numéro de compte: {client.get('account_number', '')}\n"
                    f"- Statut client: {client.get('customer_status', '')}\n"
                    f"- Montant impayé: {client.get('unpaid_amount', 0)} DT\n"
                    f"- Jours de retard: {client.get('late_days', 0)} jours\n"
                    f"- Mensualités impayées: {client.get('number_of_unpaid_installment', 0)}\n"
                    f"- Mensualité normale: {client.get('normal_payment', 0)} DT\n"
                    f"- Montant total du prêt: {client.get('apply_amount_total', 0)} DT\n"
                    f"- Durée du prêt: {client.get('term_period', 0)} mois\n"
                    f"- Statut workflow: {client.get('statut_workflow', '')}\n\n"
                    f"Réponds uniquement à la question demandée. N'ajoute aucune donnée non demandée."
                )
                if user_msg_override is not None:
                    user_msg = user_msg_override

            elif route == "rag":
                if rag:
                    system_msg = "Tu es un conseiller bancaire tunisien qui parle à un client au téléphone. En utilisant le contexte juridique fourni, explique la réponse en langage simple et naturel comme tu parlerais à quelqu'un qui ne connaît pas le droit. Cite l'article uniquement si c'est utile pour rassurer le client. Maximum 2 phrases courtes. Ne lis pas le texte juridique mot pour mot."
                    user_msg = (
                        f"Contexte juridique (utilise-le obligatoirement): {rag[:500]}\n\n"
                        f"Question du client: {state['transcript']}\n"
                        f"Réponse courte:"
                    )
                    if user_msg_override is not None:
                        user_msg = user_msg_override
                else:
                    system_msg = (
                        "Tu es un conseiller bancaire tunisien. "
                        "Parle comme un humain au téléphone. "
                        "Réponds avec ta connaissance générale bancaire, de façon claire et simple. "
                        "1 à 2 phrases maximum. "
                        "Ne dis jamais 'je n’ai pas cette information'."
                    )
                    user_msg = state["transcript"]
                    if user_msg_override is not None:
                        user_msg = user_msg_override

            else:
                # general
                system_msg = (
                    "Parle comme un conseiller bancaire humain au téléphone. "
                    "Utilise un ton naturel, simple, et direct. "
                    "Évite les définitions académiques."
                )
                user_msg = state["transcript"]
                if user_msg_override is not None:
                    user_msg = user_msg_override

            # Universal language lock: avoid multilingual drift in long generations.
            system_msg += (
                " Réponds exclusivement en français. "
                "N'utilise jamais l'espagnol, le portugais, ni l'anglais. "
                "Si un montant est mentionné, garde la devise telle qu'elle est fournie dans les données. "
                "Si aucune devise n'est fournie, n'en invente pas. "
                "Les montants sont en dinars tunisiens (DT), pas en euros. "
                "Ne dis jamais 'euros', dis toujours 'dinars' ou 'DT'."
            )

            self._stt.pause()
            try:
                for sentence in self._stream_sentences_with_decimal_fix(
                    user_msg, system_msg
                ):
                    full_response_parts.append(sentence)

                    if first_audio_time is None:
                        first_audio_time = _time.monotonic() - t0
                        _log(f"LLM first-sentence latency: {first_audio_time:.2f} sec")

                    self.speak(sentence, log_output=False)
            finally:
                self._stt.resume()

            assistant_text = " ".join(full_response_parts)
            _log(f"Assistant: {assistant_text!r}")

            try:
                _mcp("log_call", {
                    "customer_id": state.get("customer_id") or self._customer_id,
                    "transcript": user_text,
                    "intent": route,
                    "outcome": "completed",
                    "agent_decision": assistant_text[:500],
                    "session_id": self._session_id,
                    "turn_number": current_turn,
                })
            except Exception as log_exc:
                _log(f"[DB] log_call failed: {log_exc}")

            stt_latency = getattr(self._stt, "_last_decode_seconds", 0.0)
            perceived = stt_latency + (first_audio_time or 0.0)
            _log(f"TOTAL perceived latency: {perceived:.2f} sec")

            # Persist conversation turn.
            self._session_summary["turns"][-1]["assistant_text"] = assistant_text

        except Exception as exc:
            _log(f"Error in response worker: {exc}")
        finally:
            try:
                self._processing_lock.release()
            except RuntimeError:
                pass

    def _start_negotiation_turn(self) -> None:
        # Step 1: ask the reason before proposing the plan.
        ask_reason_msg = (
            "Avant tout, puis-je vous demander la raison de ce retard de paiement ?"
        )
        already_paused = bool(
            getattr(
                getattr(self._stt, "_paused_event", None), "is_set", lambda: False
            )()
        )
        if not already_paused:
            self._stt.pause()
        try:
            self.speak(ask_reason_msg)
        finally:
            if not already_paused:
                self._stt.resume()
        self._negotiation_step = "await_reason"
        return ask_reason_msg

    def _start_negotiation_turn_with_plan(self) -> None:
        """Start the negotiation by presenting the debt and proposing a plan.

        This is the previous body of `_start_negotiation_turn()`.

        This method pauses STT while speaking, but only if it is not already
        paused (to avoid nested pause/resume bugs).
        """

        already_paused = bool(
            getattr(
                getattr(self._stt, "_paused_event", None), "is_set", lambda: False
            )()
        )
        if not already_paused:
            self._stt.pause()
        try:
            profile = self._negotiation_profile
            client = self._client_info

            if not client:
                try:
                    info = _mcp("get_client_info", {"customer_id": self._customer_id})
                    if info.get("found"):
                        self._client_info = info
                        client = info
                except Exception as exc:
                    _log(f"[NEGO] Client fetch failed: {exc}")

            if client and not profile:
                profile = classify_client_profile(client)
                self._negotiation_profile = profile

            if not profile or not client:
                fallback_msg = "Je vais maintenant vous présenter votre situation."
                self.speak(fallback_msg)
                self._negotiation_step = "await_confirmation"
                return fallback_msg

            unpaid = client.get("unpaid_amount", 0)
            late = client.get("late_days", 0)
            missed = client.get("number_of_unpaid_installment", 0)
            profile_type = profile["profile"]
            max_inst = profile["max_installments"]
            suggested = profile["suggested_amount"]
            first_date = profile["first_payment_date"]

            if profile_type == "FIDELE":
                system_msg = (
                    "Tu es un conseiller bancaire bienveillant qui appelle un bon client. "
                    "Ce client a un petit retard de paiement. Sois compréhensif et chaleureux. "
                    "Présente sa situation avec empathie. Propose un échéancier souple. "
                    f"Maximum {max_inst} mensualités. Objectif: obtenir un engagement amiable. "
                    "Réponds exclusivement en français. Maximum 3 phrases."
                )
            elif profile_type == "DIFFICILE":
                system_msg = (
                    "Tu es un agent de recouvrement bancaire professionnel. "
                    "Ce client a plusieurs mensualités impayées. Sois ferme mais respectueux. "
                    "Présente les faits clairement. Insiste sur la nécessité de régulariser rapidement. "
                    f"Propose un plan sur maximum {max_inst} mensualités. "
                    "Réponds exclusivement en français. Maximum 3 phrases."
                )
            else:
                system_msg = (
                    "Tu es un agent de recouvrement bancaire senior. "
                    "Ce client est en situation critique avec un retard grave. "
                    "Sois strict et professionnel. Mentionne les conséquences légales possibles "
                    "(inscription au fichier des mauvais payeurs, poursuites judiciaires). "
                    f"Propose un règlement immédiat ou un plan sur {max_inst} mois maximum. "
                    "Réponds exclusivement en français. Maximum 3 phrases."
                )

            user_msg = (
                f"Situation du client {client.get('customer_name', '')}:\n"
                f"- Montant impayé total: {unpaid} DT\n"
                f"- Jours de retard: {late} jours\n"
                f"- Mensualités manquantes: {missed}\n"
                f"- Mensualité normale: {client.get('normal_payment', 0)} DT\n\n"
                f"Présente-lui sa situation et propose un plan de {max_inst} mensualités "
                f"de {suggested} DT chacune, première échéance le {first_date}.\n"
                f"Demande-lui s'il accepte ce plan ou s'il préfère autre chose."
            )

            generated_sentences = []
            for sentence in self._stream_sentences_with_decimal_fix(
                user_msg, system_msg
            ):
                generated_sentences.append(sentence)
                self.speak(sentence, log_output=False)

            generated_text = " ".join(generated_sentences).strip()
            if generated_text:
                _log(f"Assistant: {generated_text!r}")

            # CRITICAL: persister le plan proposé immédiatement après l'avoir
            # présenté au client. Sans ça, self._proposed_amount reste à 0.0
            # jusqu'à la première contre-proposition/acceptation, et toute
            # réclamation déposée AVANT ce moment-là ne peut pas afficher de
            # récap correct dans _save_claim() (condition proposed_amount > 0).
            self._proposed_installments = max_inst
            self._proposed_amount = suggested
            self._proposed_date = first_date

            self._negotiation_step = "await_confirmation"
            return generated_text
        finally:
            if not already_paused:
                self._stt.resume()

    def _handle_negotiation_turn(self, user_text: str, turn_number: int) -> None:
        from llm.agent import call_llm_raw
        from llm.langgraph_agent import (
            classify_client_profile,
            run_negotiation_graph,
        )

        # Guard: transition phrases after a claim — keep negotiation on track
        # (avoid routing to RAG via intent="question" when client just wants to resume).
        RESUME_NEGO_PATTERNS = [
            "revenons",
            "retour",
            "reprendre",
            "continuer",
            "suite",
            "négociation",
            "negociation",
            "on continue",
            "on reprend",
        ]
        user_lower = (user_text or "").lower()
        if (
            self._negotiation_step == "await_confirmation"
            and self._proposed_amount > 0
            and any(p in user_lower for p in RESUME_NEGO_PATTERNS)
        ):
            recap_msg = (
                "Bien sûr. Pour rappel, notre proposition est : "
                f"{self._proposed_installments} mensualité(s) "
                f"de {self._proposed_amount} DT, "
                f"première échéance le {self._proposed_date}. "
                "Acceptez-vous ce plan ?"
            )
            self._stt.pause()
            try:
                self.speak(recap_msg)
            finally:
                self._stt.resume()
            self._log_call_event(
                transcript=user_text,
                intent="negotiation_resume",
                outcome="await_confirmation",
                agent_decision=recap_msg,
                turn_number=turn_number,
            )
            return

        # Claim detection BEFORE any intent logic
        claim_keywords = [
            "réclamation",
            "réclamer",
            "signaler un problème",
            "j'ai déjà payé",
            "erreur sur",
            "contester",
            "plainte",
            "problème avec mon compte",
            "paiement non enregistré",
            "décalage",
            "restructuration",
        ]
        if any(kw in user_lower for kw in claim_keywords):
            self._claim_active = True
            self._claim_step = "await_subject"
            self._stt.pause()
            try:
                self.speak(
                    "Je comprends que vous souhaitez déposer une réclamation. "
                    "Quel est le sujet ? Par exemple : erreur sur montant, "
                    "paiement non enregistré, demande de décalage, "
                    "restructuration, comportement inapproprié, ou autre ?"
                )
            finally:
                self._stt.resume()
            self._log_call_event(
                transcript=user_text,
                intent="claim_trigger",
                outcome="claim_started_during_negotiation",
                agent_decision="Claim flow initiated during negotiation",
                turn_number=turn_number,
            )
            return

        # STEP 2 — Handle reason capture before any other negotiation logic.
        if self._negotiation_step == "await_reason":
            reason_prompt = (
                "Classe la raison donnée par le client pour le retard de paiement. "
                "Choisis exactement une valeur parmi: financial_difficulty / forgot / dispute / other. "
                "financial_difficulty: difficulté financière, chômage, baisse de revenus, maladie, imprévu. "
                "forgot: oubli, négligence, pas vu, problème de date. "
                "dispute: contestation, déjà payé, erreur, je ne dois pas, litige. "
                "other: toute autre raison ou incertain. "
                'Réponds uniquement en JSON: {"reason": "financial_difficulty"|"forgot"|"dispute"|"other"}.'
            )
            reason_raw = call_llm_raw(
                [
                    {"role": "system", "content": reason_prompt},
                    {"role": "user", "content": user_text},
                ],
                num_predict=64,
                temperature=0.0,
            )
            parsed_reason = str((reason_raw or "").strip())
            reason_data: dict = {}
            try:
                match = re.search(r"\{.*?\}", parsed_reason, flags=re.DOTALL)
                if match:
                    loaded = json.loads(match.group(0)) if match else {}
                    reason_data = loaded if isinstance(loaded, dict) else {}
            except Exception:
                reason_data = {}

            reason_value = str((reason_data or {}).get("reason") or "other").strip()
            if reason_value not in {
                "financial_difficulty",
                "forgot",
                "dispute",
                "other",
            }:
                reason_value = "other"
            self._client_reason_raw = user_text
            self._client_reason = reason_value

            plan_text = self._start_negotiation_turn_with_plan()

            self._log_call_event(
                transcript=user_text,
                intent="negotiation_reason",
                outcome=f"reason_{self._client_reason}_plan_proposed",
                agent_decision=(plan_text or ""),
                turn_number=turn_number,
            )

            self._negotiation_step = "await_confirmation"
            return

        if not self._client_info or not self._negotiation_profile:
            self._preload_client_info()

        if self._client_info and not self._negotiation_profile:
            self._negotiation_profile = classify_client_profile(self._client_info)

        result = run_negotiation_graph(
            user_text=user_text,
            negotiation_step=self._negotiation_step,
            proposed_installments=self._proposed_installments,
            proposed_amount=self._proposed_amount,
            proposed_date=self._proposed_date,
            negotiation_refusals=self._negotiation_refusals,
            profile=self._negotiation_profile or {},
            client_info=self._client_info or {},
            client_reason=self._client_reason,
        )

        self._negotiation_step = result.get("next_step", self._negotiation_step)
        self._proposed_installments = result.get(
            "new_installments", self._proposed_installments
        )
        self._proposed_amount = result.get("new_amount", self._proposed_amount)
        self._proposed_date = result.get("new_date", self._proposed_date)
        self._negotiation_refusals = result.get(
            "new_refusals", self._negotiation_refusals
        )

        action = result.get("action", "speak")
        response_text = str(result.get("response_text", "") or "").strip()
        intent = str(result.get("intent", "other") or "other").strip()

        if action == "save":
            self._save_payment_promise(
                transcript=user_text,
                turn_number=turn_number,
                intent="negotiation_accept",
            )
            return

        if action == "hangup":
            if response_text:
                self._stt.pause()
                try:
                    self.speak(response_text)
                finally:
                    self._stt.resume()
            self._log_call_event(
                transcript=user_text,
                intent="negotiation_refuse",
                outcome="hangup",
                agent_decision=response_text,
                turn_number=turn_number,
            )
            self._negotiation_active = False
            Thread(target=self.shutdown, daemon=True).start()
            return

        if action == "rag":
            state = run_voice_agent_prepare(user_text, customer_id=self._customer_id)
            rag = (state.get("rag_context") or "").strip()
            tool_res = state.get("tool_results") or []

            # Always anchor the answer to the currently proposed negotiation plan.
            system_msg = (
                "Tu es un agent de recouvrement bancaire tunisien au téléphone. "
                "Réponds à la question de manière simple et courte, "
                "en reliant la réponse au contexte du recouvrement de dette. "
                "Ne cite pas d'articles juridiques. "
                "Si la question concerne une procédure légale, explique "
                "ce que ça signifie concrètement pour le client. "
                "Si la question porte sur une mise en demeure, définis-la comme une lettre formelle "
                "envoyée par le créancier (ou son avocat) au débiteur, demandant de payer dans un délai précis, "
                "avant d'éventuelles poursuites judiciaires; ce n'est pas un acte d'un juge. "
                "Réponds exclusivement en français. Maximum 2 phrases."
            )

            if tool_res and any(item.get("ok") for item in tool_res):
                tool_client = tool_res[0]
                answer_user_msg = (
                    f"Question: {user_text}\n"
                    f"Données: montant impayé={tool_client.get('unpaid_amount', 0)} DT, "
                    f"retard={tool_client.get('late_days', 0)} jours"
                )
            elif rag:
                answer_user_msg = (
                    f"Question: {user_text}\n"
                    "(contexte juridique disponible mais non prioritaire)"
                )
            else:
                answer_user_msg = user_text

            self._stt.pause()
            question_answers: list[str] = []
            try:
                for sentence in self._stream_sentences_with_decimal_fix(
                    answer_user_msg, system_msg
                ):
                    question_answers.append(sentence)
                    self.speak(sentence, log_output=False)
                followup_msg = "Revenons à notre proposition, acceptez-vous ce plan?"
                question_answers.append(followup_msg)
                self.speak(followup_msg)
            finally:
                self._stt.resume()

            if question_answers:
                _log(f"Assistant: {' '.join(question_answers).strip()!r}")

            self._log_call_event(
                transcript=user_text,
                intent="negotiation_question",
                outcome="answered_and_resumed",
                agent_decision=" ".join(question_answers).strip(),
                turn_number=turn_number,
            )

            self._negotiation_step = "await_confirmation"
            return

        # Default: speak whatever the graph decided
        if response_text:
            self._stt.pause()
            try:
                self.speak(response_text)
            finally:
                self._stt.resume()

        self._log_call_event(
            transcript=user_text,
            intent=f"negotiation_{intent}",
            outcome=self._negotiation_step,
            agent_decision=response_text,
            turn_number=turn_number,
        )

        if self._negotiation_step == "done":
            self._negotiation_active = False
        return

    def _handle_name_confirmation(self, user_text: str, turn_number: int) -> None:
        from llm.agent import call_llm_raw
        import unicodedata as _ud

        def _norm(text: str) -> str:
            normalized = _ud.normalize("NFKD", (text or "").lower())
            normalized = "".join(ch for ch in normalized if not _ud.combining(ch))
            normalized = normalized.replace("'", " ")
            normalized = re.sub(r"[^a-z0-9\s\-]", " ", normalized)
            normalized = re.sub(r"\s+", " ", normalized).strip()
            return normalized

        user_norm = _norm(user_text)

        # Deterministic fast-path BEFORE calling the LLM.
        # Why: with only num_predict=32, the local LLM sometimes fails to
        # return valid/complete JSON for slightly longer answers like
        # "oui c'est moi" or "oui je suis Khalil", silently falling back to
        # "other" even though the intent is obvious. Catch the clear cases
        # here first; only ambiguous text goes to the LLM.
        NO_MARKERS = (
            "non",
            "mauvais numero",
            "pas moi",
            "vous faites erreur",
            "ce n est pas moi",
        )
        YES_MARKERS = (
            "oui",
            "exact",
            "affirmatif",
            "c est bien moi",
            "c est moi",
            "bien sur",
            "tout a fait",
        )

        confirm = None
        if any(m in user_norm for m in NO_MARKERS):
            confirm = "no"
        elif any(m in user_norm for m in YES_MARKERS) or re.search(
            r"\bje\s+suis\b", user_norm
        ):
            confirm = "yes"

        if confirm is None:
            # LLM détecte si le client confirme son identité
            confirm_prompt = (
                "Le client répond à la question 'Ai-je bien X en ligne ?'. "
                "Détecte sa réponse. "
                "yes: oui, c'est moi, exact, bien sûr, affirmatif, oui c'est bien moi. "
                "no: non, vous faites erreur, mauvais numéro, ce n'est pas moi. "
                "other: réponse incompréhensible ou hors sujet. "
                'Réponds uniquement en JSON: {"confirm": "yes"|"no"|"other"}'
            )

            raw = call_llm_raw(
                [
                    {"role": "system", "content": confirm_prompt},
                    {"role": "user", "content": user_text},
                ],
                num_predict=32,
                temperature=0.0,
            )

            import json as _json

            confirm = "other"
            try:
                match = re.search(r"\{.*?\}", raw or "", flags=re.DOTALL)
                if match:
                    parsed = _json.loads(match.group(0))
                    confirm = str(parsed.get("confirm", "other")).strip()
            except Exception:
                pass

        client_name = str((self._client_info or {}).get("customer_name", "")).strip()

        if confirm == "yes":
            # Client confirmé → demander DOB
            self._awaiting_name_confirmation = False
            self._awaiting_dob = True
            dob_msg = (
                f"Merci {client_name}. "
                "Pour vérifier votre identité, "
                "pouvez-vous me communiquer "
                "votre date de naissance ?"
            )
            self._stt.pause()
            try:
                self.speak(dob_msg)
            finally:
                self._stt.resume()
            self._log_call_event(
                transcript=user_text,
                intent="name_confirmation",
                outcome="confirmed",
                agent_decision=dob_msg,
                turn_number=turn_number,
            )

        elif confirm == "no":
            # Mauvais numéro → raccroche poli
            wrong_msg = (
                "Je suis désolé pour le dérangement. "
                "Il semble que nous ayons le mauvais numéro. "
                "Bonne journée."
            )
            self._stt.pause()
            try:
                self.speak(wrong_msg)
            finally:
                self._stt.resume()
            self._log_call_event(
                transcript=user_text,
                intent="name_confirmation",
                outcome="wrong_number",
                agent_decision=wrong_msg,
                turn_number=turn_number,
            )
            Thread(target=self.shutdown, daemon=True).start()

        else:
            # Réponse incomprise → réessayer max 2 fois
            self._name_confirmation_attempts += 1
            if self._name_confirmation_attempts >= 2:
                # Passer directement à DOB
                self._awaiting_name_confirmation = False
                self._awaiting_dob = True
                fallback_msg = (
                    "Pour vérifier votre identité, "
                    "pouvez-vous me communiquer "
                    "votre date de naissance ?"
                )
                self._stt.pause()
                try:
                    self.speak(fallback_msg)
                finally:
                    self._stt.resume()
            else:
                retry_msg = (
                    f"Je suis désolé, je n'ai pas bien compris. "
                    f"Ai-je bien {client_name} en ligne ?"
                )
                self._stt.pause()
                try:
                    self.speak(retry_msg)
                finally:
                    self._stt.resume()
            self._log_call_event(
                transcript=user_text,
                intent="name_confirmation",
                outcome="unclear",
                agent_decision=retry_msg
                if self._name_confirmation_attempts < 2
                else fallback_msg,
                turn_number=turn_number,
            )

    def _save_payment_promise(
        self,
        transcript: str = "",
        turn_number: int | None = None,
        intent: str = "negotiation_save",
    ) -> None:
        try:
            result = _mcp("create_payment_promise", {
                "customer_id": self._customer_id,
                "amount": self._proposed_amount,
                "installments": self._proposed_installments,
                "promised_date": self._proposed_date,
                "reason": self._client_reason,
                "reason_raw": self._client_reason_raw,
            })
            if result.get("success") or result.get("ok"):
                # Block calls until promised_date
                try:
                    _mcp("set_call_status", {
                        "customer_id": self._customer_id,
                        "status": "PROMISED",
                        "next_call_date": self._proposed_date,
                        "session_id": self._session_id,
                        "notes": f"Promesse: {self._proposed_installments}x{self._proposed_amount}DT",
                    })
                except Exception as exc:
                    _log(f"[CALL_STATUS] set PROMISED failed: {exc}")

                self._negotiation_active = False
                self._negotiation_step = "done"
                confirmation = (
                    "Parfait ! Votre engagement de paiement a été enregistré. "
                    f"Vous paierez {self._proposed_installments} mensualité(s) "
                    f"de {self._proposed_amount} DT, "
                    f"première échéance le {self._proposed_date}. "
                    "Merci pour votre coopération. "
                    "Y a-t-il autre chose que je peux faire pour vous ?"
                )
            else:
                confirmation = (
                    "J'ai bien noté votre engagement. Notre équipe va confirmer "
                    "les détails de votre plan de paiement. Merci."
                )
                self._negotiation_active = False
                self._negotiation_step = "done"
            self._stt.pause()
            try:
                self.speak(confirmation)
            finally:
                self._stt.resume()
            self._log_call_event(
                transcript=transcript,
                intent=intent,
                outcome=(
                    "promise_saved"
                    if (result.get("success") or result.get("ok"))
                    else "promise_record_pending"
                ),
                agent_decision=confirmation,
                turn_number=turn_number,
            )
        except Exception as exc:
            _log(f"[NEGO] save_payment_promise failed: {exc}")
            fallback_msg = "Votre engagement a été noté. Notre équipe vous contactera pour confirmer."
            self._stt.pause()
            try:
                self.speak(fallback_msg)
            finally:
                self._stt.resume()
            self._negotiation_active = False
            self._log_call_event(
                transcript=transcript,
                intent=intent,
                outcome="save_failed_fallback",
                agent_decision=fallback_msg,
                turn_number=turn_number,
            )

    def _handle_claim_turn(self, user_text: str, turn_number: int) -> None:
        from llm.agent import call_llm_raw

        if self._claim_step == "await_subject":
            subject_prompt = (
                "Classe la réclamation du client parmi ces sujets exactement. "
                "Réponds UNIQUEMENT avec le sujet exact, rien d'autre, en JSON. "
                "Sujets valides: "
                "'Erreur sur montant impayé', "
                "'Paiement effectué non enregistré', "
                "'Demande de décalage échéance', "
                "'Demande de restructuration', "
                "'Comportement inapproprié', "
                "'Autre réclamation'. "
                'Réponds uniquement en JSON: {"subject": "..."}'
            )
            import json as _json

            raw = call_llm_raw(
                [
                    {"role": "system", "content": subject_prompt},
                    {"role": "user", "content": user_text},
                ],
                num_predict=64,
                temperature=0.0,
            )
            # Parse subject from LLM response
            subject = "Autre réclamation"  # safe default
            try:
                match = re.search(r"\{.*?\}", (raw or ""), flags=re.DOTALL)
                if match:
                    parsed = _json.loads(match.group(0))
                    candidate = str(parsed.get("subject", "")).strip()
                    valid_subjects = {
                        "Erreur sur montant impayé",
                        "Paiement effectué non enregistré",
                        "Demande de décalage échéance",
                        "Demande de restructuration",
                        "Comportement inapproprié",
                        "Autre réclamation",
                    }
                    if candidate in valid_subjects:
                        subject = candidate
            except Exception:
                pass

            self._claim_subject = subject
            self._claim_step = "await_body"

            self._stt.pause()
            try:
                self.speak(
                    f"J'ai bien noté : {subject}. "
                    "Pouvez-vous décrire brièvement votre problème en quelques mots ?"
                )
            finally:
                self._stt.resume()

            self._log_call_event(
                transcript=user_text,
                intent="claim_subject",
                outcome=f"subject_classified_{subject}",
                agent_decision=f"Subject: {subject}",
                turn_number=turn_number,
            )
            return

        if self._claim_step == "await_body":
            self._claim_body = user_text
            self._save_claim(turn_number)
            return

    def _save_claim(self, turn_number: int) -> None:
        client = self._client_info or {}
        claim_transcript = self._claim_body
        try:
            result = _mcp("create_claim", {
                "customer_id": self._customer_id,
                "subject": self._claim_subject,
                "body": self._claim_body,
                "name": str(client.get("customer_name", "")),
                "phone": str(client.get("telephone_1", "")),
                "email": str(client.get("email", "")),
            })

            if result.get("success"):
                claim_id = result.get("id_acm_claims", "")
                msg = (
                    f"Votre réclamation a bien été enregistrée "
                    f"sous le numéro {claim_id}. "
                    "Notre équipe vous contactera dans les plus brefs délais. "
                )

                # If a negotiation was active and we already have a proposed plan,
                # resume directly with a recap and request confirmation.
                if self._negotiation_active and self._proposed_amount > 0:
                    msg += (
                        "Revenons à notre discussion : "
                        f"vous proposiez {self._proposed_installments} mensualité(s) "
                        f"de {self._proposed_amount} DT, "
                        f"première échéance le {self._proposed_date}. "
                        "Confirmez-vous cet engagement ?"
                    )
                    self._negotiation_step = "await_final_confirmation"
                else:
                    msg += "Y a-t-il autre chose que je puisse faire pour vous ?"
                outcome = "claim_saved"
            else:
                msg = (
                    "Votre réclamation a été notée. "
                    "Notre équipe va traiter votre demande rapidement."
                )

                if self._negotiation_active and self._proposed_amount > 0:
                    msg += (
                        " Revenons à notre plan : "
                        f"{self._proposed_installments} mensualité(s) "
                        f"de {self._proposed_amount} DT, "
                        f"première échéance le {self._proposed_date}. "
                        "Confirmez-vous ?"
                    )
                    self._negotiation_step = "await_final_confirmation"
                outcome = "claim_save_failed"
        except Exception as exc:
            _log(f"[CLAIM] save failed: {exc}")
            msg = (
                "Votre réclamation a été notée. Notre équipe va traiter votre demande."
            )
            if self._negotiation_active and self._proposed_amount > 0:
                msg += (
                    " Revenons à notre plan : "
                    f"{self._proposed_installments} mensualité(s) "
                    f"de {self._proposed_amount} DT, "
                    f"première échéance le {self._proposed_date}. "
                    "Confirmez-vous ?"
                )
                self._negotiation_step = "await_final_confirmation"
            outcome = "claim_exception"

        self._stt.pause()
        try:
            self.speak(msg)
        finally:
            self._stt.resume()

        self._claim_active = False
        self._claim_step = ""
        self._claim_subject = ""
        self._claim_body = ""

        self._log_call_event(
            transcript=claim_transcript,
            intent="claim_save",
            outcome=outcome,
            agent_decision=msg,
            turn_number=turn_number,
        )

    def _log_call_event(
        self,
        transcript: str,
        intent: str,
        outcome: str,
        agent_decision: str,
        turn_number: int | None = None,
    ) -> None:
        try:
            _mcp("log_call", {
                "customer_id": self._customer_id,
                "transcript": transcript,
                "intent": intent,
                "outcome": outcome,
                "agent_decision": (agent_decision or "")[:500],
                "session_id": self._session_id,
                "turn_number": int(turn_number or self._turn_number or 1),
            })
        except Exception as log_exc:
            _log(f"[DB] log_call event failed: {log_exc}")

    def generate_response(self, user_text: str) -> str:
        """LLM boundary (kept as a method for easy future tool/RAG integration)."""

        # Non-streaming path is intentionally not used for final responses.
        # Keep this method for future integrations.
        return ""

    def _stream_sentences_with_decimal_fix(self, user_msg: str, system_msg: str):
        """Yield streamed sentences while stitching decimal fragments.

        Some streamers may split "444.44" into "444." then "44 ...".
        This keeps natural TTS by merging those two parts before speaking.
        """

        pending: str | None = None

        for raw_sentence in stream_raw_sentences(user_msg, system_content=system_msg):
            sentence = (raw_sentence or "").strip()
            if not sentence:
                continue

            if pending:
                # Merge cents when sentence segmentation splits decimals.
                # Allow cases like "49 DT" and "49DT" (no word boundary after digits).
                if re.search(r"\b\d+\.$", pending) and re.match(
                    r"^\d{1,4}(?=\D|$)", sentence
                ):
                    sentence = f"{pending}{sentence}"
                    pending = None
                else:
                    yield pending
                    pending = None

            if re.search(r"\b\d+\.$", sentence):
                pending = sentence
                continue

            yield sentence

        if pending:
            yield pending

    def speak(self, text: str, log_output: bool = True) -> None:
        """TTS boundary: streaming playback with no intermediate WAV files."""
        raw_text = (text or "").strip()
        if raw_text and log_output:
            _log(f"Assistant: {raw_text!r}")
        speak_streaming(clean_for_tts(text))

    def shutdown(self) -> None:
        """Stop streaming and write a session summary."""

        if self._shutdown_event.is_set():
            return

        # Persist call status based on negotiation outcome.
        # This must never block shutdown.
        try:
            from datetime import date, timedelta

            current_status = ""
            try:
                result = _mcp("get_call_status", {"customer_id": self._customer_id})
                current_status = str((result or {}).get("status") or "").strip()
            except Exception:
                current_status = ""

            # Never override a saved promise.
            if current_status == "PROMISED":
                pass
            # Callback accepted (stage reached after 3 refusals).
            elif self._negotiation_refusals == 3 and self._negotiation_step in (
                "propose_rappel",
                "done",
            ):
                _mcp("set_call_status", {
                    "customer_id": self._customer_id,
                    "status": "CALLBACK",
                    "next_call_date": (date.today() + timedelta(days=15)).isoformat(),
                    "session_id": self._session_id,
                    "notes": "Rappel demandé dans 15 jours",
                })
            elif self._negotiation_step == "done":
                _mcp("set_call_status", {
                    "customer_id": self._customer_id,
                    "status": "FREE",
                    "session_id": self._session_id,
                    "notes": "Appel terminé — retour libre",
                })
            elif self._negotiation_refusals >= 4:
                _mcp("set_call_status", {
                    "customer_id": self._customer_id,
                    "status": "REFUSED",
                    "next_call_date": (date.today() + timedelta(days=7)).isoformat(),
                    "session_id": self._session_id,
                    "notes": "Refus total après 4 tentatives",
                })
            else:
                _mcp("set_call_status", {
                    "customer_id": self._customer_id,
                    "status": "FREE",
                    "session_id": self._session_id,
                    "notes": "Appel terminé — retour libre",
                })
        except Exception as exc:
            _log(f"[CALL_STATUS] shutdown status update failed: {exc}")

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
