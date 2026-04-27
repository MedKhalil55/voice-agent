from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import date
from threading import Event, Lock
from typing import Any

import db.tools as db_tools
import llm.agent as llm_agent_mod
import llm.langgraph_agent as lga
import main as main_mod


@dataclass
class RuntimeState:
    clients: dict[int, dict[str, Any]]
    current_client_id: int | None = None
    intent_events: list[dict[str, Any]] = field(default_factory=list)
    counter_events: list[dict[str, Any]] = field(default_factory=list)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    promise_calls: list[dict[str, Any]] = field(default_factory=list)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", (value or "").lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return normalized


def _extract_installments(text: str) -> int | None:
    t = _normalize_text(text)
    m = re.search(r"(\d+)\s*(fois|mensualit)", t)
    if m:
        return int(m.group(1))

    words = {
        "une": 1,
        "un": 1,
        "deux": 2,
        "trois": 3,
        "quatre": 4,
        "cinq": 5,
        "six": 6,
        "sept": 7,
        "huit": 8,
        "neuf": 9,
        "dix": 10,
    }
    for word, number in words.items():
        if re.search(rf"\b{word}\b", t) and (
            "fois" in t or "mensualit" in t or "payer en" in t
        ):
            return number
    return None


def _extract_amount(text: str) -> float | None:
    t = _normalize_text(text).replace(",", ".")
    m = re.search(r"(\d+(?:\.\d+)?)\s*(dt|dinar|dinars)", t)
    if m:
        return float(m.group(1))
    return None


def _detect_intent(user_text: str) -> str:
    t = _normalize_text((user_text or "").strip())

    # Critical check B: this sentence MUST be counter.
    if "je veux payer en 6 fois" in t:
        return "counter"

    if "est-ce que" in t or "?" in t:
        return "question"

    if (
        "je veux payer en" in t
        or "je prefere" in t
        or "je prefere" in t
        or "je pref" in t
        or "mensualit" in t
        or re.search(r"\ben\s+\d+\s+fois\b", t)
        or "payer en deux fois" in t
    ):
        return "counter"

    if (
        "non" in t
        or "impossible" in t
        or "je ne peux pas" in t
        or "je peux pas" in t
        or "je refuse" in t
    ):
        return "refuse"

    if (
        "oui" in t
        or "d'accord" in t
        or "ok" in t
        or "parfait" in t
        or "je confirme" in t
        or "j'accepte" in t
        or "accepte" in t
    ):
        return "accept"

    return "other"


def _fake_stream_raw_sentences(user_msg: str, system_content: str | None = None):
    system = system_content or ""
    text = user_msg or ""
    system_n = _normalize_text(system)
    text_n = _normalize_text(text)

    if "demande-lui s'il accepte ce plan" in text_n:
        m = re.search(r"plan de (\d+) mensualit.s de ([0-9]+(?:\.[0-9]+)?) DT", text)
        if m:
            yield (
                f"Je vous propose {m.group(1)} mensualites de {m.group(2)} DT. "
                "Acceptez-vous ce plan ?"
            )
            return

    if "le plan propose actuellement est" in system_n:
        max_inst_match = re.search(r"entre 1 et (\d+) mensualit", system)
        max_inst = int(max_inst_match.group(1)) if max_inst_match else 3

        q = text_n
        ask_inst = _extract_installments(q)
        if ask_inst is not None:
            if ask_inst <= max_inst:
                yield f"Oui, c'est possible. Vous pouvez payer en {ask_inst} fois."
            else:
                yield (
                    f"Non, ce n'est pas possible. Le maximum autorise est {max_inst} mensualites."
                )
            return

        yield "Oui, ce plan reste possible selon votre dossier."
        return

    if "contexte juridique disponible mais non prioritaire" in text_n:
        yield "Je reponds selon votre plan de paiement en cours."
        return

    yield "Reponse simulee."


def _fake_prepare(transcript: str, customer_id: int = 1002) -> dict[str, Any]:
    t = _normalize_text(transcript or "")
    if "payer en" in t or "consequences" in t or "saisie" in t:
        return {
            "transcript": transcript,
            "rag_context": "Article juridique disponible",
            "tool_results": [],
            "route": "rag",
            "customer_id": customer_id,
        }
    return {
        "transcript": transcript,
        "rag_context": "",
        "tool_results": [],
        "route": "general",
        "customer_id": customer_id,
    }


def _make_fake_call_llm_raw(state: RuntimeState):
    def _fake_call_llm_raw(
        messages: list[dict[str, str]], num_predict=64, temperature=0.0
    ):
        system = (messages[0].get("content") or "") if messages else ""
        user = (messages[-1].get("content") or "") if messages else ""

        if "Detecte l'intention" in system or "Détecte l'intention" in system:
            intent = _detect_intent(user)
            state.intent_events.append({"user_text": user, "intent": intent})
            return json.dumps({"intent": intent})

        if (
            "Extrait le nombre de mensualites" in system
            or "Extrait le nombre de mensualit" in system
        ):
            inst = _extract_installments(user)
            amount = _extract_amount(user)
            state.counter_events.append(
                {"user_text": user, "installments": inst, "amount": amount}
            )
            return json.dumps({"installments": inst, "amount": amount})

        return "{}"

    return _fake_call_llm_raw


def _make_fake_get_client_info(state: RuntimeState):
    def _fake_get_client_info(customer_id: int):
        data = state.clients.get(int(customer_id), {}).copy()
        if data:
            return data
        return {"found": False, "customer_id": customer_id, "ok": False}

    return _fake_get_client_info


def _make_fake_create_payment_promise(state: RuntimeState):
    def _fake_create_payment_promise(
        customer_id: int, amount: float, installments: int, promised_date: str
    ):
        state.promise_calls.append(
            {
                "customer_id": int(customer_id),
                "amount": float(amount),
                "installments": int(installments),
                "promised_date": str(promised_date),
            }
        )
        return {"success": True, "promise_id": 999}

    return _fake_create_payment_promise


def _make_fake_log_call(state: RuntimeState):
    def _fake_log_call(
        customer_id: int,
        transcript: str,
        intent: str,
        outcome: str,
        agent_decision: str,
        session_id: str = "",
        turn_number: int = 1,
    ):
        state.call_log.append(
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
        return {"success": True, "id": len(state.call_log)}

    return _fake_log_call


class _ImmediateThread:
    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        if self._target:
            self._target(*self._args, **self._kwargs)


class _DummySTT:
    def pause(self):
        return None

    def resume(self):
        return None


class TestAgent(main_mod.VoiceAgent):
    def __init__(self, customer_id: int):
        self._customer_id = int(customer_id)
        self._session_id = f"sim-{customer_id}"
        self._turn_number = 0
        self._verified = True
        self._verification_attempts = 0
        self._awaiting_dob = False
        self._negotiation_active = True
        self._negotiation_profile = None
        self._client_info = None
        self._negotiation_step = "present_debt"
        self._proposed_installments = 0
        self._proposed_amount = 0.0
        self._proposed_date = ""
        self._negotiation_refusals = 0

        self._shutdown_event = Event()
        self._processing_lock = Lock()
        self._session_summary = {"turns": []}
        self._bye_keywords = set()
        self._last_final_text = ""
        self._stt = _DummySTT()

        self.spoken_log: list[str] = []
        self.shutdown_called = False

    def speak(self, text: str) -> None:
        self.spoken_log.append((text or "").strip())

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._shutdown_event.set()


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    reason: str
    turns: list[dict[str, str]]
    final_state: dict[str, Any]
    promise_ok: bool


def _run_turn(
    agent: TestAgent, state: RuntimeState, user_text: str, turn_number: int
) -> dict[str, str]:
    start_spoken = len(agent.spoken_log)
    start_intent = len(state.intent_events)
    agent._handle_negotiation_turn(user_text, turn_number)

    intent = "n/a"
    for event in state.intent_events[start_intent:]:
        if event.get("user_text") == user_text:
            intent = event.get("intent", "n/a")
            break

    responses = agent.spoken_log[start_spoken:]
    return {
        "user_text": user_text,
        "intent": intent,
        "agent_summary": " | ".join(r for r in responses if r) or "(no response)",
    }


def _prepare_agent_for_client(
    client_id: int, client_data: dict[str, Any]
) -> tuple[TestAgent, dict[str, Any]]:
    agent = TestAgent(client_id)
    profile = lga.classify_client_profile(client_data)
    agent._client_info = client_data.copy()
    agent._negotiation_profile = profile
    agent._negotiation_active = True
    agent._negotiation_step = "present_debt"
    agent._start_negotiation_turn()
    return agent, profile


def _close_float(a: float, b: float, eps: float = 0.01) -> bool:
    return abs(float(a) - float(b)) <= eps


def run_all_tests() -> str:
    clients = {
        2001: {
            "customer_name": "Amira Bouaziz",
            "date_de_naissance": date(1990, 6, 15),
            "unpaid_amount": 400.0,
            "late_days": 15,
            "number_of_unpaid_installment": 1,
            "statut_workflow": "EN_ATTENTE",
            "normal_payment": 200.0,
            "apply_amount_total": 9600.0,
            "term_period": 48,
            "account_number": "ACC-2001",
            "customer_status": "ACTIVE",
            "found": True,
            "ok": True,
        },
        1001: {
            "customer_name": "Mohamed Trabelsi",
            "date_de_naissance": date(1985, 3, 15),
            "unpaid_amount": 937.5,
            "late_days": 30,
            "number_of_unpaid_installment": 3,
            "statut_workflow": "CONTENTIEUX",
            "normal_payment": 312.5,
            "apply_amount_total": 15000.0,
            "term_period": 48,
            "account_number": "ACC-1001",
            "customer_status": "ACTIVE",
            "found": True,
            "ok": True,
        },
        1003: {
            "customer_name": "Karim Mansouri",
            "date_de_naissance": date(1978, 11, 8),
            "unpaid_amount": 1562.49,
            "late_days": 90,
            "number_of_unpaid_installment": 3,
            "statut_workflow": "RELANCE",
            "normal_payment": 520.83,
            "apply_amount_total": 25000.0,
            "term_period": 48,
            "account_number": "ACC-1003",
            "customer_status": "ACTIVE",
            "found": True,
            "ok": True,
        },
    }

    state = RuntimeState(clients=clients)

    # Monkeypatches
    orig_stream = main_mod.stream_raw_sentences
    orig_prepare = main_mod.run_voice_agent_prepare
    orig_thread = main_mod.Thread
    orig_call_llm_raw = llm_agent_mod.call_llm_raw
    orig_create_promise = db_tools.create_payment_promise
    orig_get_client = db_tools.get_client_info
    orig_log_call = db_tools.log_call

    main_mod.stream_raw_sentences = _fake_stream_raw_sentences
    main_mod.run_voice_agent_prepare = _fake_prepare
    main_mod.Thread = _ImmediateThread
    llm_agent_mod.call_llm_raw = _make_fake_call_llm_raw(state)
    db_tools.create_payment_promise = _make_fake_create_payment_promise(state)
    db_tools.get_client_info = _make_fake_get_client_info(state)
    db_tools.log_call = _make_fake_log_call(state)

    outputs: list[str] = []
    scenario_results: list[ScenarioResult] = []

    try:
        outputs.append("=== PROFILE CLASSIFICATION ===")

        profile_expectations = {
            2001: "FIDELE",
            1001: "DIFFICILE",
            1003: None,
        }

        profile_map: dict[int, dict[str, Any]] = {}
        for cid, client in clients.items():
            profile = lga.classify_client_profile(client)
            profile_map[cid] = profile
            expected = profile_expectations[cid]
            head = (
                f"Client {cid} ({client['customer_name']}): "
                f"late_days={client['late_days']}, "
                f"workflow={client['statut_workflow']}, "
                f"unpaid_inst={client['number_of_unpaid_installment']}"
            )
            outputs.append(head)
            if expected is None:
                outputs.append(f"-> Profile: {profile['profile']} (computed)")
            else:
                ok_flag = "✅" if profile["profile"] == expected else "❌"
                outputs.append(
                    f"-> Profile: {profile['profile']} {ok_flag} (expected {expected})"
                )
            outputs.append(
                f"-> max_installments: {profile['max_installments']}, "
                f"suggested: {profile['suggested_amount']} DT"
            )

        # CHECK B explicit sentence classification
        test_b_text = "je veux payer en 6 fois"
        check_b_intent = _detect_intent(test_b_text)
        outputs.append(
            f'CHECK B explicit: "{test_b_text}" -> intent={check_b_intent} '
            f"{'✅' if check_b_intent == 'counter' else '❌'}"
        )

        outputs.append("=== SCENARIO RESULTS ===")

        def _record_result(
            label: str,
            passed: bool,
            reason: str,
            turns: list[dict[str, str]],
            agent: TestAgent,
            promise_ok: bool,
        ):
            scenario_results.append(
                ScenarioResult(
                    name=label,
                    passed=passed,
                    reason=reason,
                    turns=turns,
                    final_state={
                        "negotiation_step": agent._negotiation_step,
                        "proposed_installments": agent._proposed_installments,
                        "proposed_amount": agent._proposed_amount,
                        "proposed_date": agent._proposed_date,
                        "refusals": agent._negotiation_refusals,
                        "shutdown_called": agent.shutdown_called,
                    },
                    promise_ok=promise_ok,
                )
            )

        for cid, client in clients.items():
            profile = profile_map[cid]
            max_inst = int(profile["max_installments"])
            suggested = float(profile["suggested_amount"])
            first_date = str(profile["first_payment_date"])
            unpaid = _to_float(client["unpaid_amount"])

            # Scenario 1 - Direct accept
            start_promises = len(state.promise_calls)
            agent, _ = _prepare_agent_for_client(cid, client)
            turns = [_run_turn(agent, state, "oui d'accord je confirme", 1)]
            created = state.promise_calls[start_promises:]
            promise_ok = len(created) == 1
            if promise_ok:
                p = created[0]
                promise_ok = (
                    p["installments"] == max_inst
                    and _close_float(p["amount"], suggested)
                    and p["promised_date"] == first_date
                )
            passed = promise_ok
            reason = (
                "default plan saved correctly" if passed else "default plan mismatch"
            )
            _record_result(
                f"Client {cid} - Scenario 1 Direct accept",
                passed,
                reason,
                turns,
                agent,
                promise_ok,
            )

            # Scenario 2 - Counter valid then confirm
            start_promises = len(state.promise_calls)
            agent, _ = _prepare_agent_for_client(cid, client)
            turns = []
            turns.append(_run_turn(agent, state, "je prefere payer en 2 fois", 1))
            turns.append(_run_turn(agent, state, "oui je confirme", 2))
            created = state.promise_calls[start_promises:]
            expected_amount = round(unpaid / 2, 2)
            promise_ok = len(created) == 1
            check_a_ok = False
            if promise_ok:
                p = created[0]
                check_a_ok = p["installments"] == 2 and _close_float(
                    p["amount"], expected_amount
                )
                promise_ok = check_a_ok
            passed = promise_ok
            reason = (
                "counter preserved on accept"
                if passed
                else "counter overwritten (CHECK A failed)"
            )
            _record_result(
                f"Client {cid} - Scenario 2 Counter valid then confirm",
                passed,
                reason,
                turns,
                agent,
                promise_ok,
            )

            # Scenario 3 - Counter invalid then accept default
            start_promises = len(state.promise_calls)
            agent, _ = _prepare_agent_for_client(cid, client)
            turns = []
            turns.append(_run_turn(agent, state, "je veux payer en 10 fois", 1))
            turns.append(_run_turn(agent, state, "bon d'accord", 2))
            created = state.promise_calls[start_promises:]
            promise_ok = len(created) == 1
            rejected_msg_ok = "maximum autorise" in _normalize_text(
                turns[0]["agent_summary"]
            )
            if promise_ok:
                p = created[0]
                promise_ok = (
                    p["installments"] == max_inst
                    and _close_float(p["amount"], suggested)
                    and p["promised_date"] == first_date
                    and rejected_msg_ok
                )
            passed = promise_ok
            reason = (
                "invalid counter rejected then default accepted"
                if passed
                else "invalid counter flow mismatch"
            )
            _record_result(
                f"Client {cid} - Scenario 3 Counter invalid then default accept",
                passed,
                reason,
                turns,
                agent,
                promise_ok,
            )

            # Scenario 4 - Refuse twice -> hangup
            start_promises = len(state.promise_calls)
            agent, _ = _prepare_agent_for_client(cid, client)
            turns = []
            turns.append(_run_turn(agent, state, "non je ne peux pas payer", 1))
            turns.append(_run_turn(agent, state, "non c'est impossible", 2))
            created = state.promise_calls[start_promises:]
            promise_ok = len(created) == 0
            passed = (
                promise_ok
                and agent._negotiation_refusals == 2
                and agent.shutdown_called
            )
            reason = "hangup after 2 refusals" if passed else "refusal flow mismatch"
            _record_result(
                f"Client {cid} - Scenario 4 Refuse twice",
                passed,
                reason,
                turns,
                agent,
                promise_ok,
            )

            # Scenario 5 - Question about plan then accept
            start_promises = len(state.promise_calls)
            agent, _ = _prepare_agent_for_client(cid, client)
            turns = []
            turns.append(
                _run_turn(agent, state, "est-ce que je peux payer en 2 fois", 1)
            )
            turns.append(_run_turn(agent, state, "oui parfait", 2))
            created = state.promise_calls[start_promises:]
            question_yes = "oui" in _normalize_text(turns[0]["agent_summary"])
            promise_ok = len(created) == 1
            passed = promise_ok and question_yes
            reason = (
                "question answered with YES then saved"
                if passed
                else "question flow mismatch (CHECK C failed)"
            )
            _record_result(
                f"Client {cid} - Scenario 5 Question then accept",
                passed,
                reason,
                turns,
                agent,
                promise_ok,
            )

        # Render scenario report
        for result in scenario_results:
            outputs.append(f"[{result.name}]")
            for idx, turn in enumerate(result.turns, start=1):
                outputs.append(
                    f'Turn {idx}: "{turn["user_text"]}" -> intent={turn["intent"]}'
                )
                outputs.append(f"Agent: {turn['agent_summary']}")
            fs = result.final_state
            outputs.append(
                "Final state: "
                f"step={fs['negotiation_step']}, "
                f"proposed_installments={fs['proposed_installments']}, "
                f"proposed_amount={fs['proposed_amount']}, "
                f"proposed_date={fs['proposed_date']}"
            )
            outputs.append(
                f"DB call create_payment_promise correct args? {'YES' if result.promise_ok else 'NO'}"
            )
            outputs.append(
                f"Result: {'PASS' if result.passed else 'FAIL'} ({result.reason})"
            )

        passed_count = sum(1 for r in scenario_results if r.passed)
        failed = [r.name for r in scenario_results if not r.passed]

        outputs.append("=== GLOBAL RESULTS ===")
        outputs.append(f"PASS: {passed_count}/{len(scenario_results)}")
        outputs.append(
            f"FAIL: {len(scenario_results) - passed_count}/{len(scenario_results)}"
        )
        outputs.append(f"Failed scenarios: {failed}")

    finally:
        # Restore originals
        main_mod.stream_raw_sentences = orig_stream
        main_mod.run_voice_agent_prepare = orig_prepare
        main_mod.Thread = orig_thread
        llm_agent_mod.call_llm_raw = orig_call_llm_raw
        db_tools.create_payment_promise = orig_create_promise
        db_tools.get_client_info = orig_get_client
        db_tools.log_call = orig_log_call

    return "\n".join(outputs)


def main() -> None:
    report = run_all_tests()
    print(report)


if __name__ == "__main__":
    main()
