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

    transcript = (state.get("transcript") or "").strip()
    rag_context = (state.get("rag_context") or "").strip()
    tool_results = state.get("tool_results") or []

    if _needs_tool_call(transcript, tool_results):
        tool_calls = [
            {
                "name": "mock_account_lookup",
                "args": {"query": transcript},
            }
        ]
        return {
            "tool_calls": tool_calls,
            "response_text": "",
        }

    tool_results_text = (
        "\n".join(str(item) for item in tool_results) if tool_results else "none"
    )
    prompt = (
        "You are a voice AI assistant. Use the available context and tool outputs to answer. "
        "Respond briefly and clearly in one short paragraph.\n\n"
        f"User transcript:\n{transcript}\n\n"
        f"RAG context:\n{rag_context or 'none'}\n\n"
        f"Tool results:\n{tool_results_text}"
    )

    response = generate_ai_response(prompt)
    return {
        "tool_calls": [],
        "response_text": (response or "").strip(),
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
    }

    results: List[Dict] = []
    for call in tool_calls:
        name = call.get("name", "")
        args = call.get("args") or {}
        tool_fn = tools.get(name)

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


# Compile once at module level — not inside run_voice_agent_turn()
_app = build_voice_agent_graph()


def run_voice_agent_turn(transcript: str) -> AgentState:
    initial_state: AgentState = {
        "transcript": transcript,
        "rag_context": "",
        "tool_calls": [],
        "tool_results": [],
        "response_text": "",
    }
    return _app.invoke(initial_state)


if __name__ == "__main__":
    result = run_voice_agent_turn("recommende moi un plan de paiement pour mon compte en arrears")

    print("transcript:    ", result["transcript"])
    print("rag_context:   ", result["rag_context"])
    print("tool_calls:    ", result["tool_calls"])
    print("tool_results:  ", result["tool_results"])
    print("response_text: ", result["response_text"])
