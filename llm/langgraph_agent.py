"""LangGraph orchestration for the voice AI agent.

Flow:
retrieve -> agent -> conditional
  - "tool" -> tool_executor -> agent
  - "done" -> speak -> END
"""

from __future__ import annotations

import os
from importlib import import_module
from typing import Dict, List, TypedDict

try:
    # Package mode: python -m llm.langgraph_agent
    from llm.agent import generate_ai_response
except ModuleNotFoundError:
    # Script mode: uv run llm/langgraph_agent.py
    from agent import generate_ai_response


class AgentState(TypedDict):
    transcript: str
    rag_context: str
    tool_calls: List[Dict]
    tool_results: List[Dict]
    response_text: str


def _get_chroma_collection():
    """Get or create the default Chroma collection used for retrieval."""

    chromadb = import_module("chromadb")

    chroma_path = os.environ.get("VOICE_AGENT_CHROMA_PATH", "artifacts/chroma")
    collection_name = os.environ.get(
        "VOICE_AGENT_CHROMA_COLLECTION", "voice_agent_docs"
    )

    client = chromadb.PersistentClient(path=chroma_path)
    try:
        return client.get_collection(name=collection_name)
    except Exception:
        return client.create_collection(name=collection_name)


def _retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant context from ChromaDB using the transcript as a query."""

    transcript = (state.get("transcript") or "").strip()
    if not transcript:
        return {"rag_context": ""}

    try:
        collection = _get_chroma_collection()
        result = collection.query(
            query_texts=[transcript],
            n_results=3,
            include=["documents"],
        )
        docs = (result.get("documents") or [[]])[0]
        rag_context = "\n".join(d for d in docs if isinstance(d, str) and d.strip())
    except Exception:
        rag_context = ""

    return {"rag_context": rag_context}


def _needs_tool_call(transcript: str, tool_results: List[Dict]) -> bool:
    """Simple heuristic to decide whether a tool call is needed."""

    if tool_results:
        return False

    lowered = transcript.lower()
    keywords = (
        "solde",
        "balance",
        "compte",
        "montant",
        "echeance",
        "echeancier",
        "plan",
        "paiement",
        "payment",
    )
    return any(k in lowered for k in keywords)


def _agent_node(state: AgentState) -> AgentState:
    """Reasoning node using the existing generate_ai_response function."""

    def parse_llm_output(text: str) -> dict:
        """Parse optional tool instruction from LLM output.

        Expected format: ACTION:tool NAME:xxx ARGS:{...}
        Falls back to a direct response payload when parsing fails.
        """

        fallback = {"action": "respond", "message": (text or "").strip()}
        try:
            raw = (text or "").strip()
            if not raw:
                return fallback

            one_line = " ".join(raw.replace("\n", " ").split())
            action_pos = one_line.find("ACTION:")
            name_label = "NAME:"
            name_pos = one_line.find(name_label)
            if name_pos < 0:
                name_label = "NAME="
                name_pos = one_line.find(name_label)

            if action_pos < 0 or name_pos < 0:
                return fallback
            if not (action_pos < name_pos):
                return fallback

            action = one_line[action_pos + len("ACTION:") : name_pos].strip().lower()

            # Accept both formats:
            # 1) ACTION:tool NAME:xxx ARGS:{...}
            # 2) ACTION:tool NAME:xxx<json_object>{...}
            remainder = one_line[name_pos + len(name_label) :].strip()
            args_label = "ARGS:"
            args_pos = remainder.find(args_label)
            if args_pos < 0:
                args_label = "ARGS="
                args_pos = remainder.find(args_label)
            marker_pos = remainder.find("<json_object>")
            brace_pos = remainder.find("{")

            split_positions = [p for p in (args_pos, marker_pos, brace_pos) if p >= 0]
            if split_positions:
                cut = min(split_positions)
                name = remainder[:cut].strip()
            else:
                name = remainder.strip()

            if args_pos >= 0:
                args_text = remainder[args_pos + len(args_label) :].strip()
            elif marker_pos >= 0:
                args_text = remainder[marker_pos + len("<json_object>") :].strip()
            elif brace_pos >= 0:
                args_text = remainder[brace_pos:].strip()
            else:
                args_text = ""

            if action != "tool" or not name:
                return fallback

            import json

            args: dict = {}
            if args_text:
                left = args_text.find("{")
                right = args_text.rfind("}")
                if left >= 0 and right > left:
                    candidate = args_text[left : right + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            args = parsed
                    except Exception:
                        # Keep the tool action even when ARGS is not strict JSON.
                        args = {}

            return {"action": "tool", "name": name, "args": args}
        except Exception:
            return fallback

    transcript = (state.get("transcript") or "").strip()
    rag_context = (state.get("rag_context") or "").strip()
    tool_results = state.get("tool_results") or []

    tool_results_text = (
        "\n".join(str(item) for item in tool_results) if tool_results else "none"
    )
    prompt = (
        "You are a voice AI assistant with optional tools.\n"
        "If a tool is required, output exactly one line in this format:\n"
        "ACTION:tool NAME:<tool_name> ARGS:<json_object>\n"
        "If no tool is required, output a direct user-facing answer only.\n\n"
        "Context for this turn:\n"
        f"TRANSCRIPT:\n{transcript or 'none'}\n\n"
        f"RAG_CONTEXT:\n{rag_context or 'none'}\n\n"
        f"TOOL_RESULTS:\n{tool_results_text}"
    )

    llm_output = (generate_ai_response(prompt) or "").strip()
    parsed = parse_llm_output(llm_output)

    if parsed.get("action") == "tool":
        return {
            "tool_calls": [
                {
                    "name": parsed.get("name", ""),
                    "args": parsed.get("args") or {},
                }
            ],
            "response_text": "",
        }

    return {
        "tool_calls": [],
        "response_text": parsed.get("message", llm_output),
    }


def _mock_account_lookup(args: Dict) -> Dict:
    query = args.get("query", "")
    return {
        "tool": "mock_account_lookup",
        "ok": True,
        "query": query,
        "account_status": "in_arrears",
        "outstanding_amount": 2450.0,
        "next_due_date": "2026-04-05",
    }


def _mock_payment_plan(args: Dict) -> Dict:
    requested = args.get("requested_installments", 3)
    return {
        "tool": "mock_payment_plan",
        "ok": True,
        "approved_installments": max(1, min(int(requested), 6)),
        "minimum_monthly_amount": 400.0,
    }


def _tool_executor_node(state: AgentState) -> AgentState:
    """Execute mock tools listed in state.tool_calls."""

    tool_calls = state.get("tool_calls") or []
    if not tool_calls:
        return {"tool_results": []}

    tools = {
        "mock_account_lookup": _mock_account_lookup,
        "mock_payment_plan": _mock_payment_plan,
        # Accept common LLM aliases for the same mock tools.
        "get_account_lookup": _mock_account_lookup,
        "get_client_info": _mock_account_lookup,
        "get_payment_plan": _mock_payment_plan,
        "get_arrears": _mock_account_lookup,
    }

    results: List[Dict] = []
    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("args") or {}
        tool_fn = tools.get(name)

        if tool_fn is None:
            normalized = str(name).strip().lower().replace("_", " ").replace("-", " ")
            if (
                "arrear" in normalized
                or "outstanding" in normalized
                or "solde" in normalized
                or "client" in normalized
                or "account" in normalized
            ):
                tool_fn = _mock_account_lookup
            elif (
                "payment" in normalized
                or "plan" in normalized
                or "mensual" in normalized
                or "echeancier" in normalized
            ):
                tool_fn = _mock_payment_plan

        if tool_fn is None:
            results.append(
                {
                    "tool": name,
                    "ok": False,
                    "error": "unknown_tool",
                }
            )
            continue

        try:
            results.append(tool_fn(args))
        except Exception as exc:
            results.append(
                {
                    "tool": name,
                    "ok": False,
                    "error": str(exc),
                }
            )

    return {
        "tool_calls": [],
        "tool_results": results,
    }


def _speak_node(state: AgentState) -> AgentState:
    """Finalize output state for downstream TTS playback."""

    response_text = (state.get("response_text") or "").strip()
    if response_text:
        return {"response_text": response_text}
    return {"response_text": "Je suis desole, je n'ai pas de reponse pour le moment."}


def _route_after_agent(state: AgentState) -> str:
    """Route to tool execution when tools are requested; otherwise finish."""

    return "tool" if (state.get("tool_calls") or []) else "done"


def build_voice_agent_graph():
    """Build and compile the LangGraph StateGraph with required flow."""

    from langgraph.graph import END, StateGraph

    graph = StateGraph(AgentState)

    graph.add_node("retrieve", _retrieve_node)
    graph.add_node("agent", _agent_node)
    graph.add_node("tool_executor", _tool_executor_node)
    graph.add_node("speak", _speak_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "agent")
    graph.add_conditional_edges(
        "agent",
        _route_after_agent,
        {
            "tool": "tool_executor",
            "done": "speak",
        },
    )
    graph.add_edge("tool_executor", "agent")
    graph.add_edge("speak", END)

    return graph.compile()


def seed_chroma() -> None:
    """Populate Chroma collection once with baseline French banking documents."""

    collection = _get_chroma_collection()

    # Avoid duplicates: seed only if collection is empty.
    try:
        if int(collection.count()) > 0:
            return
    except Exception:
        return

    documents = [
        "Client en retard de paiement: proposer un echeancier adapte a sa capacite financiere.",
        "En cas de retard superieur a 30 jours, presenter un plan de regularisation en plusieurs mensualites.",
        "Si le client refuse de payer, envoyer une reclamation formelle avec les details de la dette.",
        "Apres un refus explicite de paiement, informer le client des etapes officielles de recouvrement.",
        "Client agressif: garder un ton calme, professionnel et factuel en toutes circonstances.",
        "Face a des propos hostiles, ne pas repondre a l'agressivite et recentrer sur la solution de paiement.",
        "Demande de solde: consulter le compte et communiquer uniquement les informations pertinentes.",
        "Pour une demande de solde, verifier les echeances en retard avant de proposer un plan.",
        "Negociation de delai: proposer entre 3 et 6 mensualites selon le montant impaye.",
        "Si le client demande un report, proposer une premiere echeance proche et des mensualites realistes.",
        "Procedure de relance: commencer par un rappel amiable avant la lettre de mise en demeure.",
        "En absence de paiement apres relances, envoyer une lettre de mise en demeure conforme a la procedure.",
        "Client cooperatif: confirmer le plan d'action, les dates et les montants convenus.",
        "Quand le client accepte un echeancier, remercier et rappeler les prochaines etapes officielles.",
        "Toujours conclure avec un resume clair de l'accord et des canaux de contact de la banque.",
    ]

    ids = [f"fr_bank_doc_{i:03d}" for i in range(1, len(documents) + 1)]

    try:
        collection.add(ids=ids, documents=documents)
    except Exception:
        # Seeding must not block app startup.
        return


# Compile once at module level — not inside run_voice_agent_turn()
_app = build_voice_agent_graph()
seed_chroma()


def run_voice_agent_turn(transcript: str) -> AgentState:
    initial_state: AgentState = {
        "transcript": transcript,
        "rag_context": "",
        "tool_calls": [],
        "tool_results": [],
        "response_text": "",
    }
    return _app.invoke(initial_state, config={"recursion_limit": 10})


if __name__ == "__main__":
    result = run_voice_agent_turn("bonjour")

    print("transcript:    ", result["transcript"])
    print("rag_context:   ", result["rag_context"])
    print("tool_calls:    ", result["tool_calls"])
    print("tool_results:  ", result["tool_results"])
    print("response_text: ", result["response_text"])
