"""LangGraph orchestration for the voice AI agent.

Flow:
retrieve -> agent -> conditional
  - "tool" -> tool_executor -> agent
  - "done" -> speak -> END
"""

from __future__ import annotations

import os
import re
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


_CHROMA_COLLECTION = None
_CHROMA_CLIENT = None
_CHROMA_EMBEDDING_FN = None


class _OllamaEmbeddingFn:
    """Custom Ollama embedding function compatible with ChromaDB query interface."""

    def __init__(self, base_url: str, model_name: str) -> None:
        self._url = f"{base_url.rstrip('/')}/api/embeddings"
        self._model = model_name

    def name(self) -> str:
        return f"ollama-{self._model}"

    def _embed_one(self, text: str) -> list[float]:
        import requests

        resp = requests.post(
            self._url,
            json={"model": self._model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        vec = resp.json().get("embedding", [])
        if not vec or not isinstance(vec, list):
            raise ValueError(f"Empty or invalid embedding from model={self._model!r}")
        # Ensure every element is a plain float
        return [float(v) for v in vec]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        return self._embed(input)

    def embed_query(self, input: str | list[str] = None, **kwargs) -> list[float]:  # noqa: A002
        # ChromaDB may call this as embed_query(input=text) or embed_query(text)
        if input is None:
            input = kwargs.get("input", "")
        if isinstance(input, list):
            return self._embed_one(input[0])
        return self._embed_one(input)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)


def _get_chroma_collection():
    global _CHROMA_COLLECTION, _CHROMA_CLIENT, _CHROMA_EMBEDDING_FN
    if _CHROMA_COLLECTION is not None:
        return _CHROMA_COLLECTION

    chromadb = import_module("chromadb")

    chroma_path = os.environ.get("VOICE_AGENT_CHROMA_PATH", "artifacts/chroma")
    collection_name = os.environ.get(
        "VOICE_AGENT_CHROMA_COLLECTION", "voice_agent_docs_nomic"
    )
    ollama_base_url = os.environ.get(
        "VOICE_AGENT_OLLAMA_BASE_URL", "http://localhost:11434"
    )
    embedding_model = os.environ.get(
        "VOICE_AGENT_EMBED_MODEL", "nomic-embed-text:latest"
    )

    if _CHROMA_EMBEDDING_FN is None:
        _CHROMA_EMBEDDING_FN = _OllamaEmbeddingFn(
            base_url=ollama_base_url,
            model_name=embedding_model,
        )

    if _CHROMA_CLIENT is None:
        _CHROMA_CLIENT = chromadb.PersistentClient(path=chroma_path)

    # get_or_create_collection is atomic and works with newer ChromaDB
    _CHROMA_COLLECTION = _CHROMA_CLIENT.get_or_create_collection(
        name=collection_name,
        embedding_function=_CHROMA_EMBEDDING_FN,
        metadata={"hnsw:space": "cosine"},
    )

    print(
        f"[CHROMA] Collection '{collection_name}' loaded, count={_CHROMA_COLLECTION.count()}"
    )
    return _CHROMA_COLLECTION


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


def _split_long_article(text, article_num):
    """Split a long legal article into overlapping subchunks."""

    source = " ".join((text or "").split())
    if not source:
        return []

    max_size = 500
    overlap = 80
    words = source.split(" ")

    chunks: list[dict] = []
    start_word = 0
    total_words = len(words)

    while start_word < total_words:
        end_word = start_word
        char_count = 0

        # Grow a chunk one whole word at a time, keeping size near max_size.
        while end_word < total_words:
            word_len = len(words[end_word])
            add_len = word_len if char_count == 0 else word_len + 1

            if end_word > start_word and char_count + add_len > max_size:
                break

            # Always include at least one word, even if that single word exceeds max_size.
            char_count += add_len
            end_word += 1

            if char_count > max_size:
                break

        if end_word <= start_word:
            end_word = start_word + 1

        chunk_text = " ".join(words[start_word:end_word]).strip()
        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "article": article_num,
                    "type": "article",
                }
            )

        if end_word >= total_words:
            break

        overlap_chars = 0
        next_start = end_word

        # Step backward from chunk end until overlap budget is met.
        while next_start > start_word:
            previous_word_len = len(words[next_start - 1])
            add_len = previous_word_len if overlap_chars == 0 else previous_word_len + 1
            overlap_chars += add_len
            next_start -= 1
            if overlap_chars >= overlap:
                break

        if next_start <= start_word:
            next_start = start_word + 1

        start_word = next_start

    return chunks


def chunk_legal_pdf(text: str) -> list[dict]:
    """Chunk legal PDFs by article, preserving article semantics."""

    source = " ".join((text or "").split())
    if not source:
        return []

    pattern = re.compile(r"(Art(?:icle)?\.?\s*\d+\s*[-–])", flags=re.IGNORECASE)
    parts = pattern.split(source)

    chunks: list[dict] = []

    # No article markers found: keep one fallback chunk.
    if len(parts) < 3:
        if len(source) > 800:
            return _split_long_article(source, "unknown")
        return [{"text": source, "article": "unknown", "type": "article"}]

    # parts format: [prefix, marker1, body1, marker2, body2, ...]
    for i in range(1, len(parts), 2):
        marker = (parts[i] or "").strip()
        body = (parts[i + 1] if i + 1 < len(parts) else "").strip()
        article_text = f"{marker} {body}".strip()
        if not article_text:
            continue

        num_match = re.search(r"\d+", marker)
        article_num = num_match.group(0) if num_match else "unknown"

        if len(article_text) > 800:
            chunks.extend(_split_long_article(article_text, article_num))
        else:
            chunks.append(
                {
                    "text": article_text,
                    "article": article_num,
                    "type": "article",
                }
            )

    return chunks


def _hybrid_score(
    query: str,
    docs: list[str],
    distances: list[float],
    metadatas: list[dict] | None = None,
) -> list[str]:
    """Rank docs with a cosine/BM25 hybrid score and return top filtered docs."""

    if not docs:
        return []

    import unicodedata

    # Keep docs, distances, and metadatas aligned while applying distance caps.
    aligned: list[tuple[str, float, dict]] = []
    for i, doc in enumerate(docs):
        distance = float(distances[i]) if i < len(distances) else 1.0
        meta = (metadatas[i] if metadatas and i < len(metadatas) else {}) or {}
        aligned.append((doc, distance, meta))

    # Prefer very close matches first; relax cap if it would otherwise return nothing.
    capped = [row for row in aligned if row[1] < 0.25]
    if not capped:
        capped = [row for row in aligned if row[1] < 0.32]
    if not capped:
        capped = sorted(aligned, key=lambda row: row[1])[:3]

    docs = [row[0] for row in capped]
    distances = [row[1] for row in capped]
    metadatas = [row[2] for row in capped]

    from rank_bm25 import BM25Okapi

    def _tokenize(text: str) -> list[str]:
        normalized = unicodedata.normalize(
            "NFKD", (text or "").lower().replace("�", " ")
        )
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return re.findall(r"[a-z0-9]+", normalized)

    query_tokens = _tokenize(query)
    tokenized_docs = [_tokenize(doc) for doc in docs]

    stopwords = {
        "le",
        "la",
        "les",
        "de",
        "des",
        "du",
        "un",
        "une",
        "et",
        "ou",
        "en",
        "au",
        "aux",
        "a",
        "est",
        "que",
        "qui",
        "quoi",
        "qu",
        "comment",
        "c",
        "ce",
        "cela",
    }
    query_terms = {tok for tok in query_tokens if tok and tok not in stopwords}
    query_stems = {tok[:6] for tok in query_terms if len(tok) >= 4}
    doc_stem_sets = [
        {tok[:6] for tok in tokens if len(tok) >= 4} for tokens in tokenized_docs
    ]
    stem_doc_freq = {
        stem: sum(1 for stems in doc_stem_sets if stem in stems) for stem in query_stems
    }
    total_stem_weight = sum(1.0 / max(1, stem_doc_freq[stem]) for stem in query_stems)

    if any(tokenized_docs):
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(query_tokens)
    else:
        bm25_scores = [0.0] * len(docs)

    max_bm25 = max(float(score) for score in bm25_scores) if len(bm25_scores) else 0.0

    scored: list[tuple[float, str, str]] = []
    for i, doc in enumerate(docs):
        distance = float(distances[i]) if i < len(distances) else 1.0
        cosine_component = 1.0 - distance

        bm25_raw = float(bm25_scores[i]) if i < len(bm25_scores) else 0.0
        bm25_norm = (bm25_raw / max_bm25) if max_bm25 > 0.0 else 0.0
        if cosine_component < 0.70:
            bm25_norm *= 0.3

        doc_terms = set(tokenized_docs[i]) if i < len(tokenized_docs) else set()
        doc_stems = doc_stem_sets[i] if i < len(doc_stem_sets) else set()

        term_overlap = (
            (len(query_terms & doc_terms) / float(len(query_terms)))
            if query_terms
            else 0.0
        )
        stem_overlap = (
            (len(query_stems & doc_stems) / float(len(query_stems)))
            if query_stems
            else term_overlap
        )

        weighted_stem_overlap = 0.0
        if query_stems and total_stem_weight > 0.0:
            weighted_stem_overlap = (
                sum(
                    (1.0 / max(1, stem_doc_freq[stem]))
                    for stem in query_stems
                    if stem in doc_stems
                )
                / total_stem_weight
            )

        lexical_overlap = max(term_overlap, stem_overlap, weighted_stem_overlap)

        hybrid = (
            (0.75 * cosine_component) + (0.25 * bm25_norm) + (0.12 * lexical_overlap)
        )
        if query_stems and stem_overlap == 0.0:
            hybrid -= 0.05

        article_id = str((metadatas[i] or {}).get("article", "unknown"))
        scored.append((hybrid, doc, article_id))

    scored.sort(key=lambda item: item[0], reverse=True)

    deduped: list[tuple[float, str]] = []

    # Deduplicate by article number only when article metadata is present.
    has_article_metadata = any(
        (meta or {}).get("article") for meta in (metadatas or [])
    )
    if has_article_metadata:
        seen_articles: set[str] = set()
        for idx, (score, doc, article_id) in enumerate(scored):
            key = (
                article_id if article_id and article_id != "unknown" else f"__idx_{idx}"
            )
            if key in seen_articles:
                continue
            seen_articles.add(key)
            deduped.append((score, doc))
    else:
        deduped = [(score, doc) for score, doc, _ in scored]

    filtered = [doc for score, doc in deduped if score > 0.45][:2]
    if filtered:
        return filtered

    if deduped:
        return [deduped[0][1]]

    return []


def _retrieve_node(state: AgentState) -> AgentState:
    transcript = (state.get("transcript") or "").strip()
    if state.get("route") in ("casual", "ack", "tool"):
        return {"rag_context": ""}
    if not transcript:
        return {"rag_context": ""}

    docs = []
    distances = []
    top_docs = []

    try:
        collection = _get_chroma_collection()

        # Embed la query manuellement — évite les problèmes d'interface ChromaDB
        query_vec = _CHROMA_EMBEDDING_FN._embed_one(transcript)

        result = collection.query(
            query_embeddings=[query_vec],
            n_results=40,
            include=["documents", "distances", "metadatas"],
        )
        docs = (result.get("documents") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        top_docs = _hybrid_score(transcript, docs, distances, metadatas=metadatas)
        rag_context = "\n\n".join(top_docs).strip()
        if len(rag_context) > 800:
            rag_context = rag_context[:800].rstrip()

    except Exception as exc:
        print(f"[RAG] ❌ Exception: {exc}")
        rag_context = ""

    print(f"[RAG] Query: {transcript!r}")
    print(f"[RAG] Raw distances: {list(zip(docs, distances))}")
    print(f"[RAG] Top docs after filter: {top_docs}")
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
- "rag": user asks about banking concepts, product definitions, general banking rules, or legal banking notions (qu'est-ce que, comment fonctionne, définition, avantages de, mise en demeure, lettre de change, chèque sans provision, traite, billet à ordre, prescription, force majeure, résiliation, clause pénale, intérêts de retard, recouvrement, dette, créance, saisie, caution, gage, hypothèque)
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
"C'est quoi une mise en demeure?" → {"primary":"rag","tool_name":null,"secondary":null}
"Quels sont les délais de prescription?" → {"primary":"rag","tool_name":null,"secondary":null}
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
    """Seed Chroma from all legal PDFs in rag_docs/ with structured metadata."""

    collection = _get_chroma_collection()

    project_root = Path(__file__).resolve().parents[1]
    docs_dir = project_root / "rag_docs"
    if not docs_dir.exists() or not docs_dir.is_dir():
        print("[SEED] ❌ rag_docs/ introuvable")
        return

    pdf_paths = sorted(docs_dir.glob("*.pdf"))
    if not pdf_paths:
        print("[SEED] ❌ Aucun PDF trouvé")
        return

    BATCH_SIZE = 8  # Small batches — Ollama embedding one-by-one is slow but safe

    for pdf_path in pdf_paths:
        source_key = str(pdf_path.resolve())
        print(f"[SEED] Traitement: {pdf_path.name}")

        try:
            existing = collection.get(where={"source": source_key}, include=[])
            if existing and existing.get("ids"):
                print(
                    f"[SEED] ⏭ Déjà ingéré: {pdf_path.name} ({len(existing['ids'])} chunks)"
                )
                continue
        except Exception as e:
            print(f"[SEED] ⚠ Check existant échoué (ok): {e}")

        text = _extract_pdf_text(pdf_path)
        if not text:
            print(f"[SEED] ❌ Pas de texte: {pdf_path.name}")
            continue
        print(f"[SEED] ✓ Texte: {len(text)} chars")

        chunks = chunk_legal_pdf(text)
        if not chunks:
            print(f"[SEED] ❌ Pas de chunks: {pdf_path.name}")
            continue
        print(f"[SEED] ✓ Chunks: {len(chunks)}")

        filename = pdf_path.stem
        filename_norm = filename.lower()
        if "obligations" in filename_norm:
            code = "COC"
        elif "commerce" in filename_norm:
            code = "COMMERCE"
        else:
            code = "UNKNOWN"

        filename_id = re.sub(r"[^a-zA-Z0-9_]+", "_", filename)

        ids, documents, metadatas = [], [], []
        for index, chunk in enumerate(chunks, start=1):
            chunk_text = chunk.get("text", "").strip()
            if not chunk_text:
                continue
            article = str(chunk.get("article", "unknown"))
            ids.append(f"{filename_id}_article_{article}_{index}")
            documents.append(chunk_text)
            metadatas.append(
                {
                    "source": source_key,
                    "source_name": pdf_path.name,
                    "article": article,
                    "type": chunk.get("type", "article"),
                    "code": code,
                }
            )

        total = len(ids)
        ingested = 0
        failed = 0

        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            b_ids = ids[batch_start:batch_end]
            b_docs = documents[batch_start:batch_end]
            b_meta = metadatas[batch_start:batch_end]

            try:
                collection.add(ids=b_ids, documents=b_docs, metadatas=b_meta)
                ingested += len(b_ids)
                print(
                    f"[SEED]   ✓ batch {batch_start}–{batch_end} ({ingested}/{total})"
                )
            except Exception as exc:
                failed += len(b_ids)
                print(f"[SEED]   ❌ batch {batch_start}–{batch_end} ERREUR: {exc}")
                print(f"[SEED]      premier doc: {b_docs[0][:80]!r}")

        print(f"[SEED] {pdf_path.name}: {ingested} ingérés, {failed} échoués")

    final = collection.count()
    print(f"\n[SEED] ✅ TOTAL collection: {final} chunks")


# Compile once at module level — not inside run_voice_agent_turn()
_app = build_voice_agent_graph()


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
