"""Offline negotiation flow simulator (no microphone, no TTS, no DB).

Runs end-to-end negotiation scenarios by calling internal methods of `VoiceAgent`
from `main.py`:
- `_start_negotiation_turn()` (simulates post-verification start)
- `_handle_negotiation_turn(user_text, turn_number)` for each user turn

Heavy dependencies are mocked:
- `speak()` prints + logs (no Piper TTS)
- `_stt.pause()` / `_stt.resume()` are no-ops
- DB tools (`get_client_info`, `create_payment_promise`, `log_call`) are faked
- LLM calls (`call_llm_raw`, `stream_raw_sentences`) are faked

Usage:
    C:/.../.venv/Scripts/python.exe test_negotiation.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from threading import Event
from typing import Any, Callable
from unittest.mock import MagicMock

import db.tools as db_tools
import llm.agent as llm_agent_mod
import llm.langgraph_agent as lga
import main as main_mod
from main import VoiceAgent


# ---------------------------
# Test data (as requested)
# ---------------------------

CLIENT_1 = {  # Fatma Ben Ali - DIFFICILE profile
    "customer_id": 1002,
    "customer_name": "Fatma Ben Ali",
    "dob": "22/07/1990",
    "unpaid_amount": 444.44,
    "late_days": 60,
    "number_of_unpaid_installment": 2,
    "statut_workflow": "EN_ATTENTE",
    "normal_payment": 222.22,
    # Expected profile: DIFFICILE, max_installments=3, suggested=148.15
}

CLIENT_2 = {  # Mohamed Trabelsi - CONTENTIEUX profile
    "customer_id": 1001,
    "customer_name": "Mohamed Trabelsi",
    "dob": "15/03/1985",
    "unpaid_amount": 937.5,
    "late_days": 30,
    "number_of_unpaid_installment": 3,
    "statut_workflow": "CONTENTIEUX",
    "normal_payment": 312.5,
    # Expected profile: CONTENTIEUX, max_installments=2, suggested=468.75
}

CLIENT_3 = {  # Karim Mansouri - DIFFICILE profile (late_days=90 -> CONTENTIEUX)
    "customer_id": 1003,
    "customer_name": "Karim Mansouri",
    "dob": "08/11/1978",
    "unpaid_amount": 1562.49,
    "late_days": 90,
    "number_of_unpaid_installment": 3,
    "statut_workflow": "RELANCE",
    "normal_payment": 520.83,
    # Expected profile: CONTENTIEUX (late_days >= 90), max_installments=2, suggested=781.25
}


# ---------------------------
# Helpers / fakes
# ---------------------------


def _norm(text: str) -> str:
    """Lower + remove accents + collapse spaces (for robust contains checks)."""

    value = unicodedata.normalize("NFKD", (text or "").lower())
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _norm_tokens(text: str) -> str:
    """Normalization for intent/extraction (drops punctuation)."""

    value = _norm(text)
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _contains_any(haystack: list[str], needles: list[str]) -> bool:
    merged = "\n".join(haystack)
    merged_n = _norm(merged)
    return all(_norm(n) in merged_n for n in needles)


def _extract_amount(text: str) -> float | None:
    t = _norm_tokens(text).replace(",", ".")
    m = re.search(r"(?<!\d)(\d+(?:\.\d{1,2})?)\s*(dt|dinar|dinars)\b", t)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_installments(text: str) -> int | None:
    t = _norm_tokens(text)

    # digits first
    m = re.search(r"\b(\d+)\s*(?:fois|mensualit(?:e|es))\b", t)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None

    # "en X"
    m = re.search(r"\ben\s+(une|un|deux|trois|quatre|cinq|six)\b", t)
    if m:
        words = {
            "une": 1,
            "un": 1,
            "deux": 2,
            "trois": 3,
            "quatre": 4,
            "cinq": 5,
            "six": 6,
        }
        return words.get(m.group(1))

    # "X mensualités" word-based not supported here (not needed)
    return None


def _detect_intent(user_text: str) -> str:
    raw = user_text or ""
    t = _norm_tokens(raw)

    # Question
    if "?" in raw or "c est quoi" in t or "quels sont" in t or "mise en demeure" in t:
        return "question"

    # Counter: mention installments OR money amounts
    if (
        "mensualit" in t
        or re.search(r"\ben\s+\d+\b", t)
        or re.search(r"\ben\s+(une|un|deux|trois|quatre|cinq|six)\b", t)
        or ("payer" in t and "en" in t)
        or re.search(r"\b\d+\s*(dt|dinar|dinars)\b", t)
    ):
        return "counter"

    # Refuse
    if (
        re.search(r"\bnon\b", t)
        or "je refuse" in t
        or "impossible" in t
        or "je ne peux pas" in t
        or "je peux pas" in t
    ):
        return "refuse"

    # Accept
    if (
        "j accepte" in t
        or "j accepte" in t
        or "j'accepte" in raw.lower()
        or "je confirme" in t
        or re.search(r"\bconfirme\b", t)
        or re.search(r"\boui\b", t)
        or re.search(r"\bok\b", t)
        or "d accord" in t
        or "daccord" in t
        or "parfait" in t
    ):
        return "accept"

    return "other"


def _fake_stream_raw_sentences(user_msg: str, system_content: str | None = None):
    """Fake LLM sentence streamer used by `_stream_sentences_with_decimal_fix`."""

    msg_n = _norm(user_msg)

    # Negotiation intro
    if "propose un plan" in msg_n and "mensualit" in msg_n:
        # Try to echo structured proposal for clarity
        m = re.search(
            r"plan de (\d+) mensualit[^\d]* de ([0-9]+(?:\.[0-9]+)?) dt[^\d]* premi[^\n]* le ([0-9\-]+)",
            msg_n,
        )
        if m:
            yield (
                f"Je vous propose {m.group(1)} mensualités de {m.group(2)} DT, première échéance le {m.group(3)}. "
                "Acceptez-vous ce plan ?"
            )
        else:
            yield "Je vous propose un plan de paiement. Acceptez-vous ce plan ?"
        return

    # Question answers
    if msg_n.startswith("question:"):
        yield "Je comprends votre question. Voici une réponse concise."
        return

    yield "(réponse simulée)"


def _fake_prepare(transcript: str, customer_id: int = 0) -> dict[str, Any]:
    return {
        "transcript": transcript,
        "rag_context": "",
        "tool_results": [],
        "route": "general",
        "customer_id": customer_id,
    }


class _ImmediateThread:
    """Replacement for `threading.Thread` used inside main.py to run synchronously."""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)


@dataclass
class Runtime:
    clients_by_id: dict[int, dict[str, Any]]
    spoken: list[str]
    promises: list[dict[str, Any]]
    call_logs: list[dict[str, Any]]


def _make_fake_get_client_info(rt: Runtime) -> Callable[[int], dict[str, Any]]:
    def _fake_get_client_info(customer_id: int) -> dict[str, Any]:
        data = rt.clients_by_id.get(int(customer_id))
        if not data:
            return {"found": False, "customer_id": int(customer_id)}
        payload = dict(data)
        payload["found"] = True
        return payload

    return _fake_get_client_info


def _make_fake_create_payment_promise(rt: Runtime):
    def _fake_create_payment_promise(
        customer_id: int,
        amount: float,
        installments: int,
        promised_date: str,
    ) -> dict[str, Any]:
        record = {
            "customer_id": int(customer_id),
            "amount": float(amount),
            "installments": int(installments),
            "promised_date": str(promised_date),
        }
        rt.promises.append(record)
        return {"success": True, "id": 123, **record}

    return _fake_create_payment_promise


def _make_fake_log_call(rt: Runtime):
    def _fake_log_call(
        customer_id: int,
        transcript: str,
        intent: str,
        outcome: str,
        agent_decision: str,
        session_id: str = "",
        turn_number: int = 1,
    ) -> dict[str, Any]:
        rt.call_logs.append(
            {
                "customer_id": int(customer_id),
                "transcript": transcript,
                "intent": intent,
                "outcome": outcome,
                "agent_decision": agent_decision,
                "session_id": session_id,
                "turn_number": int(turn_number),
            }
        )
        return {"success": True, "id": len(rt.call_logs)}

    return _fake_log_call


def _make_fake_call_llm_raw(rt: Runtime):
    def _fake_call_llm_raw(
        messages: list[dict[str, str]], num_predict=64, temperature=0.0
    ):
        system = (messages[0].get("content") or "") if messages else ""
        user = (messages[-1].get("content") or "") if messages else ""

        if "Détecte l'intention" in system or "Detecte l'intention" in system:
            intent = _detect_intent(user)
            return json.dumps({"intent": intent}, ensure_ascii=False)

        if (
            "Extrait le nombre de mensual" in system
            or "Extrait le nombre de mensualités" in system
        ):
            return json.dumps(
                {
                    "installments": _extract_installments(user),
                    "amount": _extract_amount(user),
                },
                ensure_ascii=False,
            )

        return "{}"

    return _fake_call_llm_raw


# ---------------------------
# Scenario runner
# ---------------------------


def _make_agent(rt: Runtime) -> VoiceAgent:
    agent = VoiceAgent.__new__(VoiceAgent)

    # Minimal required fields for negotiation
    agent._session_id = "test-session"
    agent._turn_number = 0
    agent._verified = True
    agent._negotiation_active = True
    agent._negotiation_step = "await_confirmation"
    agent._proposed_installments = 0
    agent._proposed_amount = 0.0
    agent._proposed_date = ""
    agent._negotiation_refusals = 0
    agent._verification_attempts = 0
    agent._awaiting_dob = False
    agent._client_info = None
    agent._negotiation_profile = None

    # No microphone / streaming
    agent._stt = MagicMock()
    agent._stt.pause = lambda: None
    agent._stt.resume = lambda: None

    # Shutdown flag
    agent._shutdown_event = Event()

    # Capture speak output
    def _speak(text: str, **kwargs):
        line = (text or "").strip()
        rt.spoken.append(line)
        print(f"[AGENT]: {line}")

    agent.speak = _speak  # type: ignore[method-assign]

    # Make shutdown deterministic (no file write)
    def _shutdown():
        agent._shutdown_event.set()

    agent.shutdown = _shutdown  # type: ignore[method-assign]

    return agent


def run_scenario(
    agent: VoiceAgent,
    client_data: dict[str, Any],
    scenario_name: str,
    turns: list[str],
    expected_note: str,
    *,
    expected_installments: int | None = None,
    expected_amount: float | None = None,
    should_save: bool = False,
    should_hangup: bool = False,
    must_say: list[str] | None = None,
) -> bool:
    print(f"\n{'=' * 60}")
    print(f"SCENARIO: {scenario_name}")
    print(f"Expected: {expected_note}")
    print(f"{'=' * 60}")

    # Reset negotiation state (as requested)
    agent._negotiation_active = True
    agent._negotiation_step = "await_confirmation"
    agent._proposed_installments = 0
    agent._proposed_amount = 0.0
    agent._proposed_date = ""
    agent._negotiation_refusals = 0
    agent._verified = True

    # Reset hangup flag between scenarios.
    try:
        agent._shutdown_event.clear()
    except Exception:
        agent._shutdown_event = Event()

    agent._customer_id = int(client_data["customer_id"])
    agent._client_info = None  # force reload from DB
    agent._negotiation_profile = None

    # Start negotiation turn (post-verification)
    agent._start_negotiation_turn()

    start_promises = len(RT.promises)
    start_spoken = len(RT.spoken)

    for i, user_text in enumerate(turns):
        print(f"\n[Turn {i + 1}] User: '{user_text}'")
        agent._handle_negotiation_turn(user_text, turn_number=i + 1)

    new_promises = RT.promises[start_promises:]
    spoken = RT.spoken[start_spoken:]

    saved = len(new_promises) > 0 and agent._negotiation_step == "done"

    passed = True
    reasons: list[str] = []

    if should_hangup:
        if not agent._shutdown_event.is_set():
            passed = False
            reasons.append("shutdown_event not set")
    else:
        if agent._shutdown_event.is_set():
            passed = False
            reasons.append("unexpected hangup")

    if should_save:
        if not saved:
            passed = False
            reasons.append("promise not saved")
        if len(new_promises) != 1:
            passed = False
            reasons.append(f"expected 1 promise, got {len(new_promises)}")
        if (
            expected_installments is not None
            and agent._proposed_installments != expected_installments
        ):
            passed = False
            reasons.append(
                f"installments={agent._proposed_installments} (expected {expected_installments})"
            )
        if expected_amount is not None:
            if round(float(agent._proposed_amount), 2) != round(
                float(expected_amount), 2
            ):
                passed = False
                reasons.append(
                    f"amount={round(float(agent._proposed_amount), 2)} (expected {round(float(expected_amount), 2)})"
                )
    else:
        if saved or len(new_promises) != 0:
            passed = False
            reasons.append("promise unexpectedly saved")

    if must_say:
        if not _contains_any(spoken, must_say):
            passed = False
            reasons.append(f"missing expected phrase(s): {must_say}")

    status = "PASS" if passed else "FAIL"
    reason_str = " | ".join(reasons) if reasons else "ok"
    print(f"\n[{status}] {scenario_name} -> {reason_str}")
    return passed


# Global runtime used by run_scenario signature (matches the sample structure)
RT: Runtime


def main() -> None:
    global RT

    clients_by_id = {
        1002: CLIENT_1,
        1001: CLIENT_2,
        1003: CLIENT_3,
    }

    RT = Runtime(clients_by_id=clients_by_id, spoken=[], promises=[], call_logs=[])

    # Monkeypatches (restore at end)
    orig_get_client = db_tools.get_client_info
    orig_create_promise = db_tools.create_payment_promise
    orig_log_call = db_tools.log_call
    orig_call_llm_raw = llm_agent_mod.call_llm_raw
    orig_stream = main_mod.stream_raw_sentences
    orig_prepare = main_mod.run_voice_agent_prepare
    orig_thread = main_mod.Thread
    orig_extract_date = lga.extract_payment_date_from_transcript

    db_tools.get_client_info = _make_fake_get_client_info(RT)
    db_tools.create_payment_promise = _make_fake_create_payment_promise(RT)
    db_tools.log_call = _make_fake_log_call(RT)
    llm_agent_mod.call_llm_raw = _make_fake_call_llm_raw(RT)
    main_mod.stream_raw_sentences = _fake_stream_raw_sentences
    main_mod.run_voice_agent_prepare = _fake_prepare
    main_mod.Thread = _ImmediateThread
    lga.extract_payment_date_from_transcript = lambda transcript: None  # type: ignore[assignment]

    try:
        agent = _make_agent(RT)

        print("\n=== PROFILE CHECK (computed by classify_client_profile) ===")
        for c in (CLIENT_1, CLIENT_2, CLIENT_3):
            profile = lga.classify_client_profile(c)
            print(
                f"Client {c['customer_id']} ({c['customer_name']}): "
                f"profile={profile['profile']} max={profile['max_installments']} suggested={profile['suggested_amount']} first_date={profile['first_payment_date']}"
            )

        total = 0
        passed = 0

        # CLIENT 1 scenarios
        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_1,
                "1A - Happy path accept",
                ["J'accepte le plan."],
                "promise saved with installments=3, amount=148.15",
                should_save=True,
                expected_installments=3,
                expected_amount=148.15,
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_1,
                "1B - Counter valid (en 2 fois)",
                ["Je préfère payer en 2 fois.", "Oui je confirme."],
                "turn1 counter accepted 2x222.22, turn2 promise saved",
                should_save=True,
                expected_installments=2,
                expected_amount=222.22,
                must_say=["confirmez-vous"],
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_1,
                "1C - Counter invalid (demande 6, max=3)",
                ["Je veux payer en 6 mensualités."],
                "agent explains max=3, does NOT save promise",
                should_save=False,
                must_say=["maximum", "3"],
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_1,
                "1D - Negation then valid counter",
                ["En 6 mensualités c'est possible, mais en 3 non."],
                "agent explains max=3 (6 > max), does NOT save promise",
                should_save=False,
                must_say=["maximum", "3"],
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_1,
                "1E - Negation with valid number",
                ["Je peux pas payer en 3, est-ce que je peux payer en 2?", "Oui."],
                "turn1 counter accepted installments=2, turn2 promise saved",
                should_save=True,
                expected_installments=2,
                expected_amount=222.22,
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_1,
                "1F - One shot payment",
                ["Je veux payer en une seule fois.", "Oui je confirme."],
                "installments=1, amount=444.44",
                should_save=True,
                expected_installments=1,
                expected_amount=444.44,
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_1,
                "1G - Negated one shot",
                ["Pas en une seule fois, plutôt en 3.", "Je confirme."],
                "installments=3, amount=148.15",
                should_save=True,
                expected_installments=3,
                expected_amount=148.15,
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_1,
                "1H - Refuse then accept",
                ["Non je refuse.", "D'accord j'accepte finalement."],
                "turn1 agent insists, turn2 promise saved",
                should_save=True,
                expected_installments=3,
                expected_amount=148.15,
                must_say=["sans engagement"],
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_1,
                "1I - Double refuse → hangup",
                ["Non je refuse.", "Non toujours pas."],
                "shutdown called after turn2",
                should_save=False,
                should_hangup=True,
                must_say=["communication", "prendre fin"],
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_1,
                "1J - Question then back to negotiation",
                ["C'est quoi une mise en demeure?", "Ok j'accepte le plan."],
                "turn1 answered + 'Revenons à notre proposition', turn2 promise saved",
                should_save=True,
                expected_installments=3,
                expected_amount=148.15,
                must_say=["revenons a notre proposition"],
            )
        )

        # CLIENT 2 scenarios
        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_2,
                "2A - Happy path",
                ["J'accepte."],
                "installments=2, amount=468.75, promise saved",
                should_save=True,
                expected_installments=2,
                expected_amount=468.75,
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_2,
                "2B - Counter valid (en 1 fois)",
                ["Je veux payer tout en une seule fois.", "Oui."],
                "installments=1, amount=937.5",
                should_save=True,
                expected_installments=1,
                expected_amount=937.5,
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_2,
                "2C - Counter invalid (demande 3, max=2)",
                ["Je veux payer en 3 fois."],
                "agent rejects, explains max=2",
                should_save=False,
                must_say=["maximum", "2"],
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_2,
                "2D - Refuse → legal warning",
                ["Non je ne peux pas."],
                "agent mentions legal consequences",
                should_save=False,
                must_say=["actions legales"],
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_2,
                "2E - Double refuse → hangup",
                ["Je refuse.", "Non."],
                "shutdown called after turn2",
                should_save=False,
                should_hangup=True,
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_2,
                "2F - Negation then valid",
                ["Je ne peux pas payer en 2 fois, mais en 1 fois oui.", "Je confirme."],
                "installments=1, amount=937.5, promise saved",
                should_save=True,
                expected_installments=1,
                expected_amount=937.5,
            )
        )

        # CLIENT 3 scenarios
        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_3,
                "3A - Happy path",
                ["Oui d'accord."],
                "installments=2, amount=781.25, promise saved",
                should_save=True,
                expected_installments=2,
                expected_amount=781.25,
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_3,
                "3B - Counter valid",
                ["Je peux payer en une seule fois.", "Confirme."],
                "installments=1, amount=1562.49",
                should_save=True,
                expected_installments=1,
                expected_amount=1562.49,
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_3,
                "3C - Counter invalid (en 3, max=2)",
                ["Je préfère 3 mensualités."],
                "rejected, max=2 explained",
                should_save=False,
                must_say=["maximum", "2"],
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_3,
                "3D - Complex negation",
                ["Non pas en 2 fois, ni en 3, juste en 1 seule fois.", "Oui."],
                "installments=1 (negated 2 and 3, took 1)",
                should_save=True,
                expected_installments=1,
                expected_amount=1562.49,
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_3,
                "3E - Amount counter",
                ["Je peux payer 500 DT par mois."],
                "agent rejects (amount below minimum)",
                should_save=False,
                must_say=["minimum"],
            )
        )

        total += 1
        passed += int(
            run_scenario(
                agent,
                CLIENT_3,
                "3F - Question générale",
                [
                    "Quels sont mes droits si je ne paye pas?",
                    "Ok je comprends, j'accepte le plan.",
                ],
                "turn1 answered, turn2 promise saved",
                should_save=True,
                expected_installments=2,
                expected_amount=781.25,
                must_say=["revenons a notre proposition"],
            )
        )

        print(f"\n=== SUMMARY: {passed}/{total} scenarios passed ===")

    finally:
        # Restore monkeypatches
        db_tools.get_client_info = orig_get_client
        db_tools.create_payment_promise = orig_create_promise
        db_tools.log_call = orig_log_call
        llm_agent_mod.call_llm_raw = orig_call_llm_raw
        main_mod.stream_raw_sentences = orig_stream
        main_mod.run_voice_agent_prepare = orig_prepare
        main_mod.Thread = orig_thread
        lga.extract_payment_date_from_transcript = orig_extract_date


if __name__ == "__main__":
    main()
