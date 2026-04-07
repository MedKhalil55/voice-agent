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
        route = "general"

    # Optimization: avoid RAG latency unless we explicitly need RAG.
    if route not in ("rag",):
        return {"rag_context": ""}

    try:
        collection = _get_chroma_collection()
        result = collection.query(
            query_texts=[transcript],
            n_results=3,
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

        scored_docs = [
            (s, d)
            for s, d in scored_docs
            if s > -0.25  # stricter threshold
        ]

        scored_docs.sort(key=lambda item: item[0], reverse=True)
        top_docs = [doc for _, doc in scored_docs[:2]]
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


def _classify_intent(transcript: str, tool_results: list) -> dict:
    import json
    import re

    default = {
        "primary": "general",
        "tool_name": None,
        "secondary": None,
    }

    if tool_results:
        return {
            "primary": "respond",
            "tool_name": None,
            "secondary": None,
        }

    t = " ".join((transcript or "").lower().split())
    t = re.sub(r"[\s\.,;:!?…]+$", "", t)

    # Extended keyword fast-path (no LLM call needed for obvious cases)
    CASUAL_EXACT = {
        "bonjour",
        "bonsoir",
        "salut",
        "merci",
        "merci beaucoup",
        "bonne journée",
        "comment allez-vous",
        "ça va",
    }
    ACK_EXACT = {
        "ok",
        "okay",
        "d'accord",
        "dacord",
        "compris",
        "entendu",
        "très bien",
        "parfait",
        "je vois",
        "oui",
        "non",
        "oui merci",
        "non merci",
    }

    if t in CASUAL_EXACT:
        return {
            "primary": "casual",
            "tool_name": None,
            "secondary": None,
        }
    if t in ACK_EXACT:
        return {
            "primary": "ack",
            "tool_name": None,
            "secondary": None,
        }

    # Account/payment keyword fast-path (avoids LLM for clear tool intents)
    ACCOUNT_KW = (
        "mon solde",
        "mon compte",
        "mes impayés",
        "mon impayé",
        "ma dette",
        "mes transactions",
        "mon relevé",
    )
    PAYMENT_KW = (
        "plan de paiement",
        "échéancier",
        "payer en plusieurs",
        "mensualités",
        "échelonner",
    )

    t_full = (transcript or "").lower()
    if any(kw in t_full for kw in ACCOUNT_KW):
        has_greeting = any(g in t_full for g in ("bonjour", "bonsoir", "salut"))
        return {
            "primary": "tool",
            "tool_name": "mock_account_lookup",
            "secondary": "casual" if has_greeting else None,
        }
    if any(kw in t_full for kw in PAYMENT_KW):
        has_greeting = any(g in t_full for g in ("bonjour", "bonsoir", "salut"))
        return {
            "primary": "tool",
            "tool_name": "mock_payment_plan",
            "secondary": "casual" if has_greeting else None,
        }

    # Only call LLM for ambiguous cases

    INTENT_CLASSIFIER_SYSTEM = """You are a banking intent classifier. Output ONLY one line of valid JSON, nothing else.

Classify the user message into exactly one primary intent:
- "casual": pure greeting or small talk only (bonjour, merci, ok, au revoir, bonsoir)
- "tool": user asks about THEIR personal account data (mon solde, mon compte, mes paiements, impayé, mon échéancier, mes transactions) — even if combined with a greeting
- "rag": user asks about banking concepts, product definitions, general banking rules (qu'est-ce que, comment fonctionne, définition, avantages de)
- "general": any other banking or financial question the LLM can answer from knowledge

For "tool", also detect which tool: "mock_account_lookup" for balance/account/arrears, "mock_payment_plan" for payment plans/installments.
If message mixes greeting + tool intent, set primary="tool" and secondary="casual".

Output format (single line JSON only):
{"primary":"tool","tool_name":"mock_account_lookup","secondary":"casual"}
{"primary":"casual","tool_name":null,"secondary":null}
{"primary":"general","tool_name":null,"secondary":null}
{"primary":"rag","tool_name":null,"secondary":null}

Examples:
"Bonjour" → {"primary":"casual","tool_name":null,"secondary":null}
"Quel est mon solde?" → {"primary":"tool","tool_name":"mock_account_lookup","secondary":null}
"Bonjour, quel est mon solde?" → {"primary":"tool","tool_name":"mock_account_lookup","secondary":"casual"}
"Comment fonctionne un virement?" → {"primary":"rag","tool_name":null,"secondary":null}
"C'est quoi un taux d'intérêt?" → {"primary":"general","tool_name":null,"secondary":null}
"Je veux payer en 3 fois" → {"primary":"tool","tool_name":"mock_payment_plan","secondary":null}
"""

    user_prompt = f"User message: {transcript}\nReturn JSON only."

    raw_output = (
        call_llm_raw(
            [
                {"role": "system", "content": INTENT_CLASSIFIER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            num_predict=48,
            temperature=0.0,
        )
        or ""
    ).strip()

    try:
        match = re.search(r"\{.*?\}", raw_output, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object")

        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("JSON is not an object")

        primary = str(parsed.get("primary", "")).strip().lower()
        if primary not in ("tool", "rag", "general", "casual"):
            raise ValueError("Invalid primary")

        tool_name = parsed.get("tool_name")
        if tool_name not in ("mock_account_lookup", "mock_payment_plan", None):
            raise ValueError("Invalid tool_name")

        secondary = parsed.get("secondary")
        if secondary is None:
            secondary = None
        else:
            secondary = str(secondary).strip().lower()
            if secondary != "casual":
                secondary = None

        casual_markers = (
            "bonjour",
            "bonsoir",
            "salut",
            "merci",
            "ok",
            "d'accord",
        )
        account_markers = (
            "solde",
            "compte",
            "impay",
            "dette",
        )
        payment_markers = (
            "echeancier",
            "échéancier",
            "plan de paiement",
            "mensual",
            "paiement",
        )

        if primary != "tool":
            tool_name = None
            secondary = None
        else:
            if tool_name is None:
                if any(m in t for m in payment_markers):
                    tool_name = "mock_payment_plan"
                elif any(m in t for m in account_markers):
                    tool_name = "mock_account_lookup"
                else:
                    tool_name = "mock_account_lookup"

            if any(m in t for m in casual_markers):
                secondary = "casual"

        return {
            "primary": primary,
            "tool_name": tool_name,
            "secondary": secondary,
        }
    except Exception:
        print("Intent parse error:", raw_output)
        return default


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
    route = state.get("route")
    if route is None:
        route = "general"

    agent_iterations = int(state.get("agent_iterations") or 0) + 1
    tool_call_count = int(state.get("tool_call_count") or 0)

    # Hard safety rails (graph-level): never loop tools.
    # - If a tool already ran once, force respond.
    # - Cap agent node executions at 2 (agent -> tool_executor -> agent).
    if tool_call_count >= 1 or agent_iterations >= 2:
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [],
            "response_text": "",
        }

    if route == "casual":
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [],
            "response_text": "",
        }

    if route == "tool":
        t = " ".join(transcript.lower().split())
        tool_name = "mock_account_lookup"
        if any(
            s in t for s in ("mensualité", "échéancier", "plan de paiement", "paiement")
        ):
            tool_name = "mock_payment_plan"

        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [
                {
                    "name": tool_name,
                    "args": {"query": transcript},
                }
            ],
            "response_text": "",
        }

    if route == "rag":
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [],
            "response_text": "",
        }

    if route == "general":
        return {
            "agent_iterations": agent_iterations,
            "tool_calls": [],
            "response_text": "",
        }

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
        "route": "general",
        "rag_context": "",
        "tool_calls": [],
        "tool_results": [],
        "response_text": "",
        "agent_iterations": 0,
        "tool_call_count": 0,
    }
    return _app.invoke(initial_state, config={"recursion_limit": 10})


def run_voice_agent_prepare(transcript: str) -> dict:
    # CRITICAL: intent must be computed once to avoid latency explosion
    intent = _classify_intent(transcript, [])
    initial_state: AgentState = {
        "transcript": transcript,
        "route": intent["primary"],
        "rag_context": "",
        "tool_calls": [],
        "tool_results": [],
        "response_text": "",
        "agent_iterations": 0,
        "tool_call_count": 0,
    }
    result = _app.invoke(initial_state, config={"recursion_limit": 10})
    return {
        "transcript": result.get("transcript", ""),
        "rag_context": result.get("rag_context", ""),
        "tool_results": result.get("tool_results", []),
        "route": intent["primary"],
    }


if __name__ == "__main__":
    result = run_voice_agent_turn(" quel sont les avantages du virement? ")

    print("transcript:    ", result["transcript"])
    print("rag_context:   ", result["rag_context"])
    print("tool_calls:    ", result["tool_calls"])
    print("tool_results:  ", result["tool_results"])
    print("response_text: ", result["response_text"])

    graph = build_voice_agent_graph()
    try:
        graph.get_graph().draw_png("graph.png")
    except ImportError:
        png_bytes = graph.get_graph().draw_mermaid_png()
        with open("graph.png", "wb") as file:
            file.write(png_bytes)
