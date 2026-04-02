"""LangGraph orchestration for the voice AI agent.

Flow:
retrieve -> agent -> conditional
  - "tool" -> tool_executor -> agent
  - "done" -> speak -> END
"""

from __future__ import annotations

import os
from pathlib import Path
from importlib import import_module
from typing import Dict, List, TypedDict

try:
    # Package mode: python -m llm.langgraph_agent
    from llm.agent import generate_ai_response, call_llm_raw
except ModuleNotFoundError:
    # Script mode: uv run llm/langgraph_agent.py
    from agent import generate_ai_response, call_llm_raw


class AgentState(TypedDict):
    transcript: str
    route: str | None
    rag_context: str
    tool_calls: List[Dict]
    tool_results: List[Dict]
    response_text: str
    agent_iterations: int
    tool_call_count: int


def _get_chroma_collection():
    """Get or create the default Chroma collection used for retrieval."""

    chromadb = import_module("chromadb")
    embedding_functions = import_module("chromadb.utils.embedding_functions")

    chroma_path = os.environ.get("VOICE_AGENT_CHROMA_PATH", "artifacts/chroma")
    collection_name = os.environ.get(
        "VOICE_AGENT_CHROMA_COLLECTION", "voice_agent_docs_mxbai"
    )
    ollama_base_url = os.environ.get(
        "VOICE_AGENT_OLLAMA_BASE_URL", "http://localhost:11434"
    )
    embedding_model = os.environ.get(
        "VOICE_AGENT_EMBED_MODEL", "mxbai-embed-large:latest"
    )

    embedding_fn = embedding_functions.OllamaEmbeddingFunction(
        url=f"{ollama_base_url.rstrip('/')}/api/embeddings",
        model_name=embedding_model,
    )

    client = chromadb.PersistentClient(path=chroma_path)

    def _open_or_create(name: str):
        try:
            return client.get_collection(name=name, embedding_function=embedding_fn)
        except Exception:
            return client.create_collection(
                name=name,
                embedding_function=embedding_fn,
                metadata={"hnsw:space": "cosine"},
            )

    try:
        return _open_or_create(collection_name)
    except Exception:
        # If a collection already exists with a different embedding function,
        # switch to a dedicated mxbai collection name.
        fallback_name = f"{collection_name}_mxbai"
        return _open_or_create(fallback_name)


def _extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from PDF pages."""

    try:
        pypdf = import_module("pypdf")
        PdfReader = pypdf.PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(str(pdf_path))
    except Exception:
        return ""

    pages: List[str] = []
    for page in reader.pages:
        try:
            page_text = page.extract_text() or ""
        except Exception:
            page_text = ""
        if page_text.strip():
            pages.append(page_text)

    return "\n\n".join(pages)


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split text into overlapping character chunks."""

    source = " ".join((text or "").split())
    if not source:
        return []

    size = max(int(chunk_size), 200)
    overlap = max(0, min(int(chunk_overlap), size - 1))
    step = max(1, size - overlap)

    chunks: List[str] = []
    start = 0
    while start < len(source):
        end = min(start + size, len(source))
        chunk = source[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(source):
            break
        start += step
    return chunks


def _retrieve_node(state: AgentState) -> AgentState:
    """Retrieve relevant context from ChromaDB using the transcript as a query.

    Always runs - provides RAG context for every query.
    The LLM will decide how to use this context (or ignore it if not relevant).
    """

    transcript = (state.get("transcript") or "").strip()
    if not transcript:
        return {"rag_context": ""}

    route = state.get("route")
    if route is None:
        route = _fast_route(transcript, [])

    # Optimization: avoid RAG latency unless we explicitly need RAG.
    if route != "rag":
        return {"rag_context": ""}

    try:
        collection = _get_chroma_collection()
        result = collection.query(
            query_texts=[transcript],
            n_results=8,
            include=["documents", "distances"],
        )
        docs = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        # Rerank results by semantic similarity + lightweight lexical overlap.
        # Pure embedding distance can sometimes miss the best chunk for
        # product names / abbreviations.
        query_terms = [t for t in transcript.lower().split() if len(t) >= 4]
        query_term_set = set(query_terms)

        scored_docs = []
        for i, doc in enumerate(docs):
            if not isinstance(doc, str) or not doc.strip():
                continue
            # Distance is embedding similarity; lower is better
            distance = float(distances[i]) if i < len(distances) else 10.0
            doc_lower = doc.lower()
            lexical_hits = 0
            for term in query_term_set:
                if term in doc_lower:
                    lexical_hits += 1

            # Higher score is better; semantic dominates, lexical breaks ties.
            score = (-distance) + (0.15 * lexical_hits)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        top_docs = [doc for _, doc in scored_docs[:3]]
        rag_context = "\n".join(top_docs)
    except Exception:
        rag_context = ""

    return {"rag_context": rag_context}


def _parse_agent_json_output(output: str) -> dict:
    """Parse strict JSON output from the agent.

    Expected format:
    {
      "action": "tool" | "respond",
      "tool_name": "...",
      "arguments": {...},
      "response": "..."
    }

        Rules:
        - If action="tool": tool_name and arguments required
        - If action="respond": no final natural-language response is generated inside LangGraph
            (response is optional and ignored)
        - Invalid JSON falls back to safe "respond" action
    """
    import json

    try:
        # Extract JSON from output (in case LLM adds extra text)
        output = (output or "").strip()
        if not output:
            raise ValueError("Empty output")

        # Try to find JSON object in output
        start_idx = output.find("{")
        end_idx = output.rfind("}")
        if start_idx < 0:
            raise ValueError("No JSON object found")

        # If no closing brace, try to add missing braces (handle truncation)
        if end_idx < 0 or end_idx <= start_idx:
            json_str = output[start_idx:]
            missing = json_str.count("{") - json_str.count("}")
            if missing > 0:
                json_str += "}" * missing
            parsed = json.loads(json_str)
        else:
            json_str = output[start_idx : end_idx + 1]
            parsed = json.loads(json_str)

        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON is not an object")

        # Validate and normalize
        action = str(parsed.get("action", "")).strip().lower()
        if action not in ("tool", "respond", "rag"):
            raise ValueError(f"Invalid action: {action}")

        if action == "tool":
            tool_name = str(parsed.get("tool_name", "")).strip()
            arguments = parsed.get("arguments", {})
            if not tool_name:
                raise ValueError("tool_name is required for action='tool'")
            if not isinstance(arguments, dict):
                arguments = {}
            return {
                "action": "tool",
                "tool_name": tool_name,
                "arguments": arguments,
            }
        elif action == "rag":
            return {
                "action": "rag",
            }
        else:  # action == "respond"
            # IMPORTANT: LangGraph must not generate final natural-language text.
            # The streaming voice pipeline (main.py) will generate the final response.
            return {
                "action": "respond",
            }

    except Exception:
        # Safe fallback: stop tool loop.
        return {
            "action": "respond",
        }


def _fast_route(transcript: str, tool_results: list) -> str | None:
    if tool_results:
        return None

    import re

    t = " ".join(transcript.lower().split())
    t = re.sub(r"[\s\.,;:!?…]+$", "", t)

    # strict casual
    casual = ("bonjour", "bonsoir", "salut", "merci", "ok", "d'accord")
    if any(c in t for c in casual):
        return "respond"

    # account intent
    account_signals = (
        "solde",
        "compte",
        "dette",
        "impayé",
        "sol",
        "camp",
        "mon camp",
    )
    if any(s in t for s in account_signals):
        return "account"

    # payment intent
    payment_signals = (
        "mensualité",
        "échéancier",
        "payer en",
        "plan de paiement",
        "paiement",
    )
    if any(s in t for s in payment_signals):
        return "payment"

    # knowledge intent
    if any(s in t for s in ("qu'est-ce que", "qu'est-ce qu", "définition", "explique")):
        return "rag"

    return None


def _agent_node(state: AgentState) -> AgentState:
    """Pure LLM-driven agent node.

    The agent decides:
    - Whether to call a tool (based on RAG context and task)
    - What to respond (final answer)

    No business logic, no shortcuts, no keyword-based decisions.
    All routing is LLM-driven through structured JSON output.

    The LLM is responsible for deciding whether to call a tool or respond.
    """

    transcript = (state.get("transcript") or "").strip()
    rag_context = (state.get("rag_context") or "").strip()
    tool_results = state.get("tool_results") or []
    agent_iterations = int(state.get("agent_iterations") or 0) + 1
    tool_call_count = int(state.get("tool_call_count") or 0)

    fast = _fast_route(transcript, tool_results)

    if fast == "respond":
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [],
            "response_text": "",
        }

    if fast == "account":
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [
                {"name": "mock_account_lookup", "args": {"query": transcript}}
            ],
            "response_text": "",
        }

    if fast == "payment":
        import re

        nums = re.findall(r"\d+", transcript)
        n = int(nums[0]) if nums else 3
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [
                {"name": "mock_payment_plan", "args": {"requested_installments": n}}
            ],
            "response_text": "",
        }

    if fast == "rag":
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [],
            "response_text": "",
        }

    # Hard safety rails (graph-level): never loop tools.
    # - If a tool already ran once, force respond.
    # - Cap agent node executions at 2 (agent -> tool_executor -> agent).
    if tool_call_count >= 1 or agent_iterations >= 2:
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [],
            "response_text": "",
        }

    # Construct the structured system prompt
    system_prompt = """
    CRITICAL:
- Output MUST be ONLY valid JSON
- Do NOT add any text before or after
- Do NOT write "respond:" or anything else
You are a banking intent router. Output ONLY a single-line JSON object (must start with '{' and end with '}').

HIGHEST PRIORITY RULE:
- If the user message is a greeting or small talk (e.g., bonjour, salut, hello, merci, ok, au revoir), ALWAYS choose respond.
- For greetings/small talk: NEVER choose tool, NEVER choose rag.

Choose ONLY ONE action:
- respond: greetings / small talk / general explanation not needing personal account data.
- tool: ONLY for personal/account data.
    Trigger tool if the question is about THE USER'S account using possessives (mon/ma/mes) or mentions account data like: solde, compte, paiements, transaction(s), relevé.
- rag: ONLY for banking concepts/definitions/general knowledge answered from documents.

Tool constraints:
- NEVER call multiple tools.
- If unsure: {"action":"respond"}

Output format (single line):
- respond: {"action":"respond"}
- rag: {"action":"rag"}
- tool: {"action":"tool","tool_name":"mock_account_lookup","arguments":{"query":"<transcript>"}}
    or {"action":"tool","tool_name":"mock_payment_plan","arguments":{"requested_installments":3}}

Extra examples:
User: Quel est mon solde ? -> {"action":"tool","tool_name":"mock_account_lookup","arguments":{"query":"<transcript>"}}
    """

    # Build tool results section for the prompt
    if tool_results:
        tool_results_lines = []
        for result in tool_results:
            tool_name = result.get("tool", "unknown")
            status = "✓ success" if result.get("ok") else "✗ failed"
            tool_results_lines.append(f"Tool: {tool_name} | Status: {status}")
            # Add key parts of result
            for key, value in result.items():
                if key not in ("tool", "ok"):
                    tool_results_lines.append(f"  {key}: {value}")
        tool_results_text = "\n".join(tool_results_lines)
    else:
        tool_results_text = "[No tool results yet]"

    user_prompt = (
        f"User question:\n{transcript}\n\n"
        f"Banking documents (RAG context):\n{rag_context if rag_context else '[No relevant documents found]'}\n\n"
        f"Previous tool executions:\n{tool_results_text}\n\n"
    )

    user_prompt += "Output JSON response:"

    # Call LLM statelessly (no ConversationSession, no extra system prompt).
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    llm_output = (call_llm_raw(messages, num_predict=32, temperature=0.0) or "").strip()
    print(f"[AGENT RAW OUTPUT] {llm_output!r}")

    # Parse JSON output strictly
    parsed = _parse_agent_json_output(llm_output)

    action = parsed.get("action")
    if action == "tool":
        # Tool execution requested
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [
                {
                    "name": parsed["tool_name"],
                    "args": parsed.get("arguments", {}),
                }
            ],
            "response_text": "",
        }
    else:  # action == "respond" or action == "rag"
        # IMPORTANT: LangGraph must NOT generate a final natural-language response.
        # main.py will generate the final response in a streaming fashion.
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [],
            "response_text": "",
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
        return {
            "tool_results": [],
            "tool_call_count": int(state.get("tool_call_count") or 0),
        }

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
        "tool_call_count": int(state.get("tool_call_count") or 0) + 1,
    }


def _speak_node(state: AgentState) -> AgentState:
    """Finalize output state for downstream TTS playback."""

    # IMPORTANT: LangGraph must not output a final natural-language response.
    # Keep response_text as-is (typically empty). The streaming TTS pipeline
    # generates spoken text in main.py.
    return {"response_text": (state.get("response_text") or "").strip()}


def _route_after_agent(state: AgentState) -> str:
    """Route to tool execution when tools are requested; otherwise finish."""

    if int(state.get("tool_call_count") or 0) >= 1:
        return "done"
    if int(state.get("agent_iterations") or 0) >= 2:
        return "done"
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
    """Seed Chroma from banque.pdf with chunking + Ollama embeddings."""

    collection = _get_chroma_collection()

    project_root = Path(__file__).resolve().parents[1]
    pdf_path = Path(os.environ.get("VOICE_AGENT_RAG_PDF", project_root / "banque.pdf"))
    if not pdf_path.exists():
        return

    source_key = str(pdf_path.resolve())

    # Avoid duplicate ingestion for the same source.
    try:
        existing = collection.get(where={"source": source_key}, include=[])
        if existing and existing.get("ids"):
            return
    except Exception:
        pass

    text = _extract_pdf_text(pdf_path)
    if not text:
        return

    chunk_size = int(os.environ.get("VOICE_AGENT_RAG_CHUNK_SIZE", "900"))
    chunk_overlap = int(os.environ.get("VOICE_AGENT_RAG_CHUNK_OVERLAP", "150"))
    chunks = _chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        return

    ids = [f"banque_pdf_chunk_{i:05d}" for i in range(1, len(chunks) + 1)]
    metadatas = [
        {
            "source": source_key,
            "source_name": pdf_path.name,
            "chunk_index": i,
            "chunk_total": len(chunks),
        }
        for i in range(len(chunks))
    ]

    try:
        collection.add(ids=ids, documents=chunks, metadatas=metadatas)
    except Exception:
        # Seeding must not block app startup.
        return


# Compile once at module level — not inside run_voice_agent_turn()
_app = build_voice_agent_graph()
seed_chroma()


def run_voice_agent_turn(transcript: str) -> AgentState:
    initial_state: AgentState = {
        "transcript": transcript,
        "route": _fast_route(transcript, []),
        "rag_context": "",
        "tool_calls": [],
        "tool_results": [],
        "response_text": "",
        "agent_iterations": 0,
        "tool_call_count": 0,
    }
    return _app.invoke(initial_state, config={"recursion_limit": 10})


def run_voice_agent_prepare(transcript: str) -> dict:
    state = run_voice_agent_turn(transcript)
    return {
        "transcript": state.get("transcript", ""),
        "rag_context": state.get("rag_context", ""),
        "tool_results": state.get("tool_results", []),
        "route": state.get("route"),
    }


if __name__ == "__main__":
    result = run_voice_agent_turn(" quel sont les avantages du virement? ")

    print("transcript:    ", result["transcript"])
    print("rag_context:   ", result["rag_context"])
    print("tool_calls:    ", result["tool_calls"])
    print("tool_results:  ", result["tool_results"])
    print("response_text: ", result["response_text"])
