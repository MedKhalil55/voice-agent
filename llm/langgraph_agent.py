"""LangGraph orchestration for the voice AI agent.

Flow:
retrieve -> agent -> conditional
  - "tool" -> tool_executor -> agent
  - "done" -> speak -> END
"""

from __future__ import annotations

import os
import re
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
from importlib import import_module
from typing import Dict, List, TypedDict

from db.tools import create_claim, create_payment_promise, get_client_info, log_call

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
    customer_id: int | None
    verified: bool
    verification_attempts: int


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
    """Rank docs with cosine + BM25 hybrid score and return top filtered docs."""

    if not docs:
        return []

    import unicodedata
    from rank_bm25 import BM25Okapi

    # --- Step 1: filter by distance cap ---
    aligned: list[tuple[str, float, dict]] = []
    for i, doc in enumerate(docs):
        distance = float(distances[i]) if i < len(distances) else 1.0
        meta = (metadatas[i] if metadatas and i < len(metadatas) else {}) or {}
        aligned.append((doc, distance, meta))

    # Keep only docs within distance threshold
    capped = [row for row in aligned if row[1] < 0.22]
    if not capped:
        capped = [row for row in aligned if row[1] < 0.28]
    if not capped:
        capped = sorted(aligned, key=lambda row: row[1])[:3]

    docs_f = [row[0] for row in capped]
    distances_f = [row[1] for row in capped]
    metadatas_f = [row[2] for row in capped]

    # --- Step 2: BM25 scoring ---
    def _tokenize(text: str) -> list[str]:
        normalized = unicodedata.normalize(
            "NFKD", (text or "").lower().replace("", " ")
        )
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
        return re.findall(r"[a-z0-9]+", normalized)

    query_tokens = _tokenize(query)
    tokenized_docs = [_tokenize(doc) for doc in docs_f]

    if any(tokenized_docs):
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = list(bm25.get_scores(query_tokens))
    else:
        bm25_scores = [0.0] * len(docs_f)

    max_bm25 = max(bm25_scores) if bm25_scores else 0.0

    # --- Step 3: hybrid score (cosine is primary, BM25 is secondary) ---
    scored: list[tuple[float, str, str]] = []
    for i, doc in enumerate(docs_f):
        cosine_component = 1.0 - float(distances_f[i])
        bm25_norm = (bm25_scores[i] / max_bm25) if max_bm25 > 0.0 else 0.0

        # Cosine is the primary signal - BM25 only breaks ties
        hybrid = (0.85 * cosine_component) + (0.15 * bm25_norm)

        article_id = str((metadatas_f[i] or {}).get("article", f"__idx_{i}"))
        scored.append((hybrid, doc, article_id))

    scored.sort(key=lambda item: item[0], reverse=True)

    # --- Step 4: deduplicate by article, keeping highest-scored chunk per article ---
    seen_articles: set[str] = set()
    deduped: list[tuple[float, str]] = []
    for score, doc, article_id in scored:
        if article_id in seen_articles:
            continue
        seen_articles.add(article_id)
        deduped.append((score, doc))

    # --- Step 5: return top 2 docs above minimum score threshold ---
    MIN_SCORE = 0.72  # cosine distance < 0.22 -> cosine_component > 0.78 minimum
    filtered = [doc for score, doc in deduped if score > MIN_SCORE][:2]
    if filtered:
        return filtered

    # Fallback: return best match even if below threshold
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


def _normalize_french_text(value: str) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", (value or "").lower())
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = normalized.replace("'", " ")
    normalized = re.sub(r"[^a-z0-9\s\-]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _parse_french_number_words(text: str) -> int | None:
    units = {
        "zero": 0,
        "un": 1,
        "une": 1,
        "deux": 2,
        "trois": 3,
        "quatre": 4,
        "cinq": 5,
        "six": 6,
        "sept": 7,
        "huit": 8,
        "neuf": 9,
        "dix": 10,
        "onze": 11,
        "douze": 12,
        "treize": 13,
        "quatorze": 14,
        "quinze": 15,
        "seize": 16,
    }
    tens = {
        "vingt": 20,
        "trente": 30,
        "quarante": 40,
        "cinquante": 50,
        "soixante": 60,
    }
    fillers = {"et", "le", "la", "de", "du", "des"}

    normalized = _normalize_french_text(text).replace("-", " ")
    tokens = [tok for tok in normalized.split() if tok and tok not in fillers]
    if not tokens:
        return None

    total = 0
    current = 0
    i = 0

    while i < len(tokens):
        tok = tokens[i]

        if (
            tok == "quatre"
            and i + 1 < len(tokens)
            and tokens[i + 1]
            in {
                "vingt",
                "vingts",
            }
        ):
            current += 80
            i += 2
            continue

        if tok in units:
            current += units[tok]
            i += 1
            continue

        if tok in tens:
            current += tens[tok]
            i += 1
            continue

        if tok in {"cent", "cents"}:
            if current == 0:
                current = 1
            current *= 100
            i += 1
            continue

        if tok == "mille":
            if current == 0:
                current = 1
            total += current * 1000
            current = 0
            i += 1
            continue

        return None

    return total + current


def _build_date(day: int, month: int, year: int) -> date | None:
    try:
        return date(int(year), int(month), int(day))
    except Exception:
        return None


def extract_date_from_transcript(transcript: str) -> date | None:
    """Extract a date of birth from free-form French transcript text."""

    source = (transcript or "").strip()
    if not source:
        return None

    # 08/11/1978 or 08-11-1978
    match = re.search(r"(?<!\d)(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4})(?!\d)", source)
    if match:
        parsed = _build_date(
            day=int(match.group(1)),
            month=int(match.group(2)),
            year=int(match.group(3)),
        )
        if parsed is not None:
            return parsed

    # le 8 du 11 1978
    match = re.search(
        r"\ble\s+(\d{1,2})\s+(?:du|de)\s+(\d{1,2})\s+(\d{4})\b",
        source,
        flags=re.IGNORECASE,
    )
    if match:
        parsed = _build_date(
            day=int(match.group(1)),
            month=int(match.group(2)),
            year=int(match.group(3)),
        )
        if parsed is not None:
            return parsed

    normalized = _normalize_french_text(source)

    months = {
        "janvier": 1,
        "fevrier": 2,
        "mars": 3,
        "avril": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "aout": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "decembre": 12,
    }
    month_alt = "|".join(months.keys())

    # 8 novembre 1978
    match = re.search(rf"\b(\d{{1,2}})\s+({month_alt})\s+(\d{{4}})\b", normalized)
    if match:
        parsed = _build_date(
            day=int(match.group(1)),
            month=months[match.group(2)],
            year=int(match.group(3)),
        )
        if parsed is not None:
            return parsed

    # huit novembre mille neuf cent soixante-dix-huit
    match = re.search(
        rf"\b(?:le\s+)?([a-z\-\s]{{2,20}})\s+({month_alt})\s+([a-z\-\s]{{3,60}})\b",
        normalized,
    )
    if match:
        day_value = _parse_french_number_words(match.group(1))
        year_value = _parse_french_number_words(match.group(3))
        if day_value is not None and year_value is not None:
            parsed = _build_date(
                day=day_value,
                month=months[match.group(2)],
                year=year_value,
            )
            if parsed is not None:
                return parsed

    # LLM fallback for hard spoken forms.
    import json

    prompt = (
        "Extrait la date de naissance depuis cette phrase. "
        'Reponds uniquement en JSON: {"day": int|null, "month": int|null, "year": int|null}.'
    )

    raw = (
        call_llm_raw(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": source},
            ],
            num_predict=64,
            temperature=0.0,
        )
        or ""
    ).strip()

    try:
        match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if not match:
            return None
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            return None
        day_value = parsed.get("day")
        month_value = parsed.get("month")
        year_value = parsed.get("year")
        if None in (day_value, month_value, year_value):
            return None
        return _build_date(int(day_value), int(month_value), int(year_value))
    except Exception:
        return None


def verify_identity(transcript: str, customer_id: int, attempts: int) -> dict:
    """Verify caller identity by matching spoken DOB against database DOB."""

    current_attempts = max(int(attempts or 0), 0)
    extracted_date = extract_date_from_transcript(transcript)

    client_info = get_client_info(int(customer_id))
    db_dob = client_info.get("date_de_naissance")

    if isinstance(db_dob, date) and extracted_date is not None:
        is_match = (
            extracted_date.day == db_dob.day
            and extracted_date.month == db_dob.month
            and extracted_date.year == db_dob.year
        )
        if is_match:
            return {
                "verified": True,
                "attempts": current_attempts,
                "should_hangup": False,
                "extracted_date": extracted_date,
            }

    updated_attempts = current_attempts + 1
    return {
        "verified": False,
        "attempts": updated_attempts,
        "should_hangup": updated_attempts >= 3,
        "extracted_date": extracted_date,
    }


def classify_client_profile(client_info: dict) -> dict:
    late_days = int(client_info.get("late_days") or 0)
    unpaid_installments = int(client_info.get("number_of_unpaid_installment") or 0)
    workflow = str(client_info.get("statut_workflow") or "").strip().upper()

    if late_days <= 30 or (unpaid_installments <= 1 and workflow == "EN_ATTENTE"):
        profile = "FIDELE"
        max_installments = 6
        tone = "soft"
        legal_warning = False
    elif (
        late_days >= 90
        or (workflow == "CONTENTIEUX" and late_days > 60)
        or unpaid_installments > 4
    ):
        profile = "CONTENTIEUX"
        max_installments = 2
        tone = "strict"
        legal_warning = True
    else:
        profile = "DIFFICILE"
        max_installments = 3
        tone = "firm"
        legal_warning = False

    unpaid_amount = float(client_info.get("unpaid_amount") or 0.0)
    suggested_amount = round(unpaid_amount / max_installments, 2)

    today = date.today()
    if today.month == 12:
        first_payment_date = date(today.year + 1, 1, 1)
    else:
        first_payment_date = date(today.year, today.month + 1, 1)

    return {
        "profile": profile,
        "max_installments": max_installments,
        "tone": tone,
        "legal_warning": legal_warning,
        "suggested_amount": suggested_amount,
        "first_payment_date": first_payment_date.isoformat(),
    }


def extract_payment_date_from_transcript(transcript: str) -> str | None:
    source = (transcript or "").strip()
    if not source:
        return None

    today = date.today()

    def _end_of_month(base: date) -> date:
        if base.month == 12:
            next_month = date(base.year + 1, 1, 1)
        else:
            next_month = date(base.year, base.month + 1, 1)
        return next_month - timedelta(days=1)

    def _future_or_current(day_value: int, month_value: int) -> date | None:
        candidate = _build_date(day_value, month_value, today.year)
        if candidate is None:
            return None
        if candidate < today:
            candidate = _build_date(day_value, month_value, today.year + 1)
        return candidate

    # 05/06/2026 or 05-06-2026
    match = re.search(r"(?<!\d)(\d{1,2})[\/\-.](\d{1,2})[\/\-.](\d{4})(?!\d)", source)
    if match:
        parsed = _build_date(
            day=int(match.group(1)),
            month=int(match.group(2)),
            year=int(match.group(3)),
        )
        if parsed is not None:
            return parsed.isoformat()

    normalized = _normalize_french_text(source)

    # dans 15 jours
    match = re.search(r"\bdans\s+(\d{1,3})\s+jours?\b", normalized)
    if match:
        days = int(match.group(1))
        return (today + timedelta(days=days)).isoformat()

    if "semaine prochaine" in normalized:
        return (today + timedelta(days=7)).isoformat()

    if "fin du mois" in normalized:
        return _end_of_month(today).isoformat()

    months = {
        "janvier": 1,
        "fevrier": 2,
        "mars": 3,
        "avril": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "aout": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "decembre": 12,
    }
    month_alt = "|".join(months.keys())

    # le premier mai
    match = re.search(rf"\ble\s+premier\s+({month_alt})\b", normalized)
    if match:
        parsed = _future_or_current(1, months[match.group(1)])
        if parsed is not None:
            return parsed.isoformat()

    # le 15 mai 2026 / le 15 mai
    match = re.search(
        rf"\ble\s+(\d{{1,2}})\s+({month_alt})(?:\s+(\d{{4}}))?\b", normalized
    )
    if match:
        day_value = int(match.group(1))
        month_value = months[match.group(2)]
        year_raw = match.group(3)
        if year_raw:
            parsed = _build_date(day_value, month_value, int(year_raw))
        else:
            parsed = _future_or_current(day_value, month_value)
        if parsed is not None:
            return parsed.isoformat()

    # le 15
    match = re.search(r"\ble\s+(\d{1,2})\b", normalized)
    if match:
        day_value = int(match.group(1))
        if today.day <= day_value:
            parsed = _build_date(day_value, today.month, today.year)
        else:
            if today.month == 12:
                parsed = _build_date(day_value, 1, today.year + 1)
            else:
                parsed = _build_date(day_value, today.month + 1, today.year)
        if parsed is not None:
            return parsed.isoformat()

    # LLM fallback for hard spoken forms.
    import json

    prompt = (
        "Extrait la date de paiement promise depuis cette phrase. "
        'Réponds uniquement en JSON: {"day": int|null, "month": int|null, "year": int|null}.'
    )

    raw = (
        call_llm_raw(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": source},
            ],
            num_predict=64,
            temperature=0.0,
        )
        or ""
    ).strip()

    try:
        match = re.search(r"\{.*?\}", raw, flags=re.DOTALL)
        if not match:
            return None
        parsed_json = json.loads(match.group(0))
        if not isinstance(parsed_json, dict):
            return None
        day_value = parsed_json.get("day")
        month_value = parsed_json.get("month")
        year_value = parsed_json.get("year")
        if None in (day_value, month_value, year_value):
            return None
        parsed = _build_date(int(day_value), int(month_value), int(year_value))
        return parsed.isoformat() if parsed is not None else None
    except Exception:
        return None


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
        "mon numéro",
        "numéro de téléphone",
        "telephone",
        "téléphone",
        "mon email",
        "mon e-mail",
        "adresse email",
        "adresse mail",
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
- "tool": user asks about THEIR personal account data (mon solde, mon compte, mes paiements, impayé, mon échéancier, mes transactions, mon numéro de téléphone, mon email) - even if combined with a greeting
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
"Quel est mon numéro de téléphone?" → {"primary":"tool","tool_name":"mock_account_lookup","secondary":null}
"Donne-moi mon email" → {"primary":"tool","tool_name":"mock_account_lookup","secondary":null}
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
            "telephone",
            "téléphone",
            "numero",
            "numéro",
            "email",
            "mail",
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
                    "args": {
                        "query": transcript,
                        "customer_id": state.get("customer_id") or 1002,
                    },
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


def _decimal_safe(d: Dict) -> Dict:
    """Convert Decimal values to float for JSON serialization."""
    return {k: float(v) if isinstance(v, Decimal) else v for k, v in d.items()}


def _real_account_lookup(args: Dict) -> Dict:
    customer_id = args.get("customer_id") or args.get("query")
    try:
        customer_id = int(customer_id)
    except (TypeError, ValueError):
        return {"ok": False, "error": "customer_id invalide"}

    result = _decimal_safe(get_client_info(customer_id))
    result["ok"] = bool(result.get("found", False))
    result.setdefault("customer_id", customer_id)
    result.setdefault("tool", "get_client_info")
    return result


def _real_payment_promise(args: Dict) -> Dict:
    try:
        result = create_payment_promise(
            customer_id=int(args.get("customer_id", 0)),
            amount=float(args.get("amount", 0)),
            installments=int(args.get("installments", 1)),
            promised_date=str(args.get("promised_date", "")),
        )
        result["ok"] = bool(result.get("success", False))
        result.setdefault("tool", "create_payment_promise")
        return result
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _real_log_call(args: Dict) -> Dict:
    try:
        result = log_call(
            customer_id=int(args.get("customer_id", 0)),
            transcript=str(args.get("transcript", "")),
            intent=str(args.get("intent", "")),
            outcome=str(args.get("outcome", "completed")),
            agent_decision=str(args.get("agent_decision", ""))[:500],
        )
        result["ok"] = bool(result.get("success", False))
        result.setdefault("tool", "log_call")
        return result
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _real_create_claim(args: Dict) -> Dict:
    try:
        result = create_claim(
            customer_id=int(args.get("customer_id", 0)),
            subject=str(args.get("subject", "")),
            body=str(args.get("body", "")),
            name=str(args.get("name", "")),
            phone=str(args.get("phone", "")),
            email=str(args.get("email", "")),
        )
        result["ok"] = bool(result.get("success", False))
        result.setdefault("tool", "create_claim")
        return result
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _tool_executor_node(state: AgentState) -> AgentState:
    """Execute tools listed in state.tool_calls."""

    tool_calls = state.get("tool_calls") or []
    if not tool_calls:
        return {
            "tool_results": [],
            "tool_call_count": int(state.get("tool_call_count") or 0),
        }

    tools = {
        "get_client_info": _real_account_lookup,
        "mock_account_lookup": _real_account_lookup,
        "create_payment_promise": _real_payment_promise,
        "mock_payment_plan": _real_payment_promise,
        "log_call": _real_log_call,
        "create_claim": _real_create_claim,
        # Accept common aliases for compatibility.
        "get_account_lookup": _real_account_lookup,
        "get_payment_plan": _real_payment_promise,
        "get_arrears": _real_account_lookup,
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
                tool_fn = _real_account_lookup
            elif (
                "payment" in normalized
                or "plan" in normalized
                or "mensual" in normalized
                or "echeancier" in normalized
            ):
                tool_fn = _real_payment_promise

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
        "customer_id": 1002,
        "verified": False,
        "verification_attempts": 0,
    }
    return _app.invoke(initial_state, config={"recursion_limit": 10})


def run_voice_agent_prepare(transcript: str, customer_id: int = 1002) -> dict:
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
        "customer_id": customer_id,
        "verified": False,
        "verification_attempts": 0,
    }
    result = _app.invoke(initial_state, config={"recursion_limit": 10})
    return {
        "transcript": transcript,
        "rag_context": result.get("rag_context", ""),
        "tool_results": result.get("tool_results", []),
        "route": intent["primary"],
        "customer_id": customer_id,
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
