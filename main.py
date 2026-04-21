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


OUTBOUND_GREETING = (
    "Bonjour, je suis l’assistant bancaire automatique et je vous appelle au sujet de votre compte. "
    "Comment puis-je vous aider aujourd’hui ?"
)


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
        self._negotiation_active: bool = False
        self._negotiation_profile: dict | None = None
        self._client_info: dict | None = None
        self._negotiation_step: str = "present_debt"
        self._proposed_installments: int = 0
        self._proposed_amount: float = 0.0
        self._proposed_date: str = ""
        self._negotiation_refusals: int = 0

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

        # Pause during greeting playback to avoid STT hearing the assistant.
        self._stt.pause()
        try:
            self.speak(OUTBOUND_GREETING)
            self._session_summary["turns"].append(
                {
                    "user_text": None,
                    "assistant_text": OUTBOUND_GREETING,
                    "event": "greeting",
                }
            )
        finally:
            self._stt.resume()

        self._stt.pause()
        try:
            self.speak(
                "Pour vérifier votre identité, pouvez-vous me donner votre date de naissance ?"
            )
            self._awaiting_dob = True
        finally:
            self._stt.resume()

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
            from db.tools import get_client_info

            info = get_client_info(self._customer_id)
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
                for sentence in stream_raw_sentences(
                    user_msg, system_content=system_msg
                ):
                    sentence = (sentence or "").strip()
                    if not sentence:
                        continue

                    full_response_parts.append(sentence)

                    if first_audio_time is None:
                        first_audio_time = _time.monotonic() - t0
                        _log(f"LLM first-sentence latency: {first_audio_time:.2f} sec")

                    self.speak(sentence)
            finally:
                self._stt.resume()

            assistant_text = " ".join(full_response_parts)
            _log(f"Assistant: {assistant_text!r}")

            try:
                from db.tools import log_call

                log_call(
                    customer_id=state.get("customer_id") or self._customer_id,
                    transcript=user_text,
                    intent=route,
                    outcome="completed",
                    agent_decision=assistant_text[:500],
                    session_id=self._session_id,
                    turn_number=current_turn,
                )
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
        profile = self._negotiation_profile
        client = self._client_info

        if not client:
            try:
                from db.tools import get_client_info

                info = get_client_info(self._customer_id)
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
        for sentence in stream_raw_sentences(user_msg, system_content=system_msg):
            sentence = (sentence or "").strip()
            if sentence:
                generated_sentences.append(sentence)
                self.speak(sentence)

        self._negotiation_step = "await_confirmation"
        return " ".join(generated_sentences).strip()

    def _handle_negotiation_turn(self, user_text: str, turn_number: int) -> None:
        from llm.agent import call_llm_raw
        from llm.langgraph_agent import (
            classify_client_profile,
            extract_payment_date_from_transcript,
        )

        if not self._client_info or not self._negotiation_profile:
            self._preload_client_info()

        if self._client_info and not self._negotiation_profile:
            self._negotiation_profile = classify_client_profile(self._client_info)

        profile = self._negotiation_profile or {}
        client = self._client_info or {}

        max_inst = int(profile.get("max_installments") or 3)
        suggested = float(profile.get("suggested_amount") or 0.0)
        first_date = str(profile.get("first_payment_date") or "")
        profile_type = str(profile.get("profile") or "DIFFICILE")
        unpaid = float(client.get("unpaid_amount") or 0.0)
        min_amount = round((unpaid / max_inst) * 0.8, 2) if max_inst > 0 else 0.0

        def _extract_json(raw: str) -> dict:
            try:
                match = re.search(r"\{.*?\}", raw or "", flags=re.DOTALL)
                if not match:
                    return {}
                payload = json.loads(match.group(0))
                return payload if isinstance(payload, dict) else {}
            except Exception:
                return {}

        if self._negotiation_step != "await_confirmation":
            self._negotiation_step = "await_confirmation"

        intent_prompt = (
            "Détecte l'intention du client dans sa réponse à une proposition de plan de paiement. "
            "Choisis exactement une valeur parmi: accept, counter, refuse, question, other. "
            "accept: oui, d'accord, ok, je confirme, c'est bon, parfait, accepte. "
            "counter: le client propose un autre montant ou un autre nombre de mensualités. "
            "refuse: non, impossible, je ne peux pas, je refuse. "
            "question: le client pose une question sur le plan. "
            'Réponds uniquement en JSON: {"intent": "accept"|"counter"|"refuse"|"question"|"other"}.'
        )

        intent_raw = call_llm_raw(
            [
                {"role": "system", "content": intent_prompt},
                {"role": "user", "content": user_text},
            ],
            num_predict=64,
            temperature=0.0,
        )
        intent = str(_extract_json(intent_raw).get("intent") or "other").lower().strip()

        if intent == "accept":
            self._proposed_installments = max_inst
            self._proposed_amount = suggested
            self._proposed_date = first_date
            self._negotiation_step = "save_promise"
            self._save_payment_promise(
                transcript=user_text,
                turn_number=turn_number,
                intent="negotiation_accept",
            )
            return

        if intent == "counter":
            counter_prompt = (
                "Extrait le nombre de mensualités ou le montant proposé par le client. "
                'Réponds uniquement en JSON: {"installments": int|null, "amount": float|null}.'
            )
            counter_raw = call_llm_raw(
                [
                    {"role": "system", "content": counter_prompt},
                    {"role": "user", "content": user_text},
                ],
                num_predict=64,
                temperature=0.0,
            )
            counter_data = _extract_json(counter_raw)

            inst_value = counter_data.get("installments")
            amount_value = counter_data.get("amount")

            try:
                inst_value = int(inst_value) if inst_value is not None else max_inst
            except Exception:
                inst_value = max_inst

            try:
                amount_value = (
                    float(amount_value)
                    if amount_value is not None
                    else round(unpaid / max(inst_value, 1), 2)
                )
            except Exception:
                amount_value = suggested

            is_valid_installments = 1 <= inst_value <= max_inst
            is_valid_amount = float(amount_value) >= min_amount

            if is_valid_installments and is_valid_amount:
                self._proposed_installments = inst_value
                self._proposed_amount = round(float(amount_value), 2)
                self._proposed_date = (
                    extract_payment_date_from_transcript(user_text) or first_date
                )
                self._negotiation_step = "await_confirmation"
                self._stt.pause()
                try:
                    counter_ok_msg = (
                        f"D'accord, je note votre proposition de {self._proposed_installments} "
                        f"mensualité(s) de {self._proposed_amount} DT. "
                        "Confirmez-vous cet engagement ?"
                    )
                    self.speak(counter_ok_msg)
                finally:
                    self._stt.resume()
                self._log_call_event(
                    transcript=user_text,
                    intent="negotiation_counter",
                    outcome="counter_accepted",
                    agent_decision=counter_ok_msg,
                    turn_number=turn_number,
                )
                return

            invalid_counter_msg = (
                "Je ne peux pas valider cette proposition. "
                f"Le maximum autorisé est {max_inst} mensualités, "
                f"avec un minimum de {min_amount} DT par mensualité. "
                "Pouvez-vous accepter ce cadre ?"
            )
            self._stt.pause()
            try:
                self.speak(invalid_counter_msg)
            finally:
                self._stt.resume()
            self._log_call_event(
                transcript=user_text,
                intent="negotiation_counter",
                outcome="counter_rejected",
                agent_decision=invalid_counter_msg,
                turn_number=turn_number,
            )
            self._negotiation_step = "await_confirmation"
            return

        if intent == "refuse":
            self._negotiation_refusals += 1

            if self._negotiation_refusals >= 2:
                stop_msg = "Je prends note de votre refus. Cette communication va prendre fin. Au revoir."
                self._stt.pause()
                try:
                    self.speak(stop_msg)
                finally:
                    self._stt.resume()
                self._log_call_event(
                    transcript=user_text,
                    intent="negotiation_refuse",
                    outcome="hangup",
                    agent_decision=stop_msg,
                    turn_number=turn_number,
                )
                Thread(target=self.shutdown, daemon=True).start()
                return

            if profile_type == "CONTENTIEUX":
                message = (
                    "Je vous informe que sans régularisation, des actions légales peuvent être engagées. "
                    "Acceptez-vous notre proposition de plan ?"
                )
            elif profile_type == "DIFFICILE":
                message = (
                    "Sans engagement, votre dossier risque de passer à une étape plus contraignante. "
                    "Pouvez-vous confirmer ce plan ?"
                )
            else:
                message = (
                    "Je comprends votre situation. Nous pouvons aussi organiser un rappel pour vous aider. "
                    "Souhaitez-vous quand même accepter ce plan aujourd'hui ?"
                )

            self._stt.pause()
            try:
                self.speak(message)
            finally:
                self._stt.resume()

            self._log_call_event(
                transcript=user_text,
                intent="negotiation_refuse",
                outcome="retry",
                agent_decision=message,
                turn_number=turn_number,
            )

            self._negotiation_step = "await_confirmation"
            return

        if intent == "question":
            state = run_voice_agent_prepare(user_text, customer_id=self._customer_id)
            route = state.get("route", "general")
            rag = (state.get("rag_context") or "").strip()
            tool_res = state.get("tool_results") or []

            if tool_res and any(item.get("ok") for item in tool_res):
                tool_client = tool_res[0]
                system_msg = (
                    "Tu es un conseiller bancaire tunisien au téléphone. "
                    "Réponds uniquement à la question posée de manière claire et courte. "
                    f"Le plan proposé est: {max_inst} mensualités de {suggested} DT, "
                    f"première échéance le {first_date}. "
                    "Si le client demande s'il peut payer en X fois et que X <= max_installments, "
                    "dis oui et confirme le plan. "
                    "Réponds exclusivement en français. Maximum 2 phrases."
                )
                answer_user_msg = (
                    f"Question du client: {user_text}\n\n"
                    f"Données client:\n"
                    f"- Nom: {tool_client.get('customer_name', '')}\n"
                    f"- Montant impayé: {tool_client.get('unpaid_amount', 0)} DT\n"
                    f"- Jours de retard: {tool_client.get('late_days', 0)}\n"
                    f"- Mensualités impayées: {tool_client.get('number_of_unpaid_installment', 0)}"
                )
            elif route == "rag" and rag:
                system_msg = (
                    "Tu es un conseiller bancaire tunisien. "
                    "Avec le contexte juridique fourni, réponds simplement en français. "
                    "Maximum 2 phrases."
                )
                answer_user_msg = (
                    f"Contexte: {rag[:500]}\n\nQuestion du client: {user_text}"
                )
            else:
                system_msg = (
                    "Tu es un conseiller bancaire tunisien au téléphone. "
                    "Réponds de façon directe, naturelle et concise. "
                    "Réponds exclusivement en français. Maximum 2 phrases."
                )
                answer_user_msg = user_text

            self._stt.pause()
            question_answers: list[str] = []
            try:
                for sentence in stream_raw_sentences(
                    answer_user_msg, system_content=system_msg
                ):
                    sentence = (sentence or "").strip()
                    if sentence:
                        question_answers.append(sentence)
                        self.speak(sentence)
                followup_msg = "Revenons à notre proposition, acceptez-vous ce plan?"
                question_answers.append(followup_msg)
                self.speak(followup_msg)
            finally:
                self._stt.resume()

            self._log_call_event(
                transcript=user_text,
                intent="negotiation_question",
                outcome="answered_and_resumed",
                agent_decision=" ".join(question_answers).strip(),
                turn_number=turn_number,
            )

            self._negotiation_step = "await_confirmation"
            return

        fallback_other_msg = (
            "Je n'ai pas bien compris. "
            f"Acceptez-vous le plan de {max_inst} mensualités de {suggested} DT ?"
        )
        self._stt.pause()
        try:
            self.speak(fallback_other_msg)
        finally:
            self._stt.resume()
        self._log_call_event(
            transcript=user_text,
            intent="negotiation_other",
            outcome="clarification",
            agent_decision=fallback_other_msg,
            turn_number=turn_number,
        )
        self._negotiation_step = "await_confirmation"

    def _save_payment_promise(
        self,
        transcript: str = "",
        turn_number: int | None = None,
        intent: str = "negotiation_save",
    ) -> None:
        try:
            from db.tools import create_payment_promise

            result = create_payment_promise(
                customer_id=self._customer_id,
                amount=self._proposed_amount,
                installments=self._proposed_installments,
                promised_date=self._proposed_date,
            )
            if result.get("success") or result.get("ok"):
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

    def _log_call_event(
        self,
        transcript: str,
        intent: str,
        outcome: str,
        agent_decision: str,
        turn_number: int | None = None,
    ) -> None:
        try:
            from db.tools import log_call

            log_call(
                customer_id=self._customer_id,
                transcript=transcript,
                intent=intent,
                outcome=outcome,
                agent_decision=(agent_decision or "")[:500],
                session_id=self._session_id,
                turn_number=int(turn_number or self._turn_number or 1),
            )
        except Exception as log_exc:
            _log(f"[DB] log_call event failed: {log_exc}")

    def generate_response(self, user_text: str) -> str:
        """LLM boundary (kept as a method for easy future tool/RAG integration)."""

        # Non-streaming path is intentionally not used for final responses.
        # Keep this method for future integrations.
        return ""

    def speak(self, text: str) -> None:
        """TTS boundary: streaming playback with no intermediate WAV files."""

        speak_streaming(clean_for_tts(text))

    def shutdown(self) -> None:
        """Stop streaming and write a session summary."""

        if self._shutdown_event.is_set():
            return

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
