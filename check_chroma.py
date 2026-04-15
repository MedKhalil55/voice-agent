# check_chroma.py
from dotenv import load_dotenv

load_dotenv(override=True)

import os, chromadb, requests

chroma_path = os.environ.get("VOICE_AGENT_CHROMA_PATH", "artifacts/chroma")
collection_name = os.environ.get(
    "VOICE_AGENT_CHROMA_COLLECTION", "voice_agent_docs_nomic"
)
ollama_url = os.environ.get("VOICE_AGENT_OLLAMA_BASE_URL", "http://localhost:11434")
embed_model = os.environ.get("VOICE_AGENT_EMBED_MODEL", "nomic-embed-text")

print(f"Chroma path:      {chroma_path}")
print(f"Collection name:  {collection_name}")
print(f"Embed model:      {embed_model}")
print(f"Ollama URL:       {ollama_url}")
print()

# 1. Check collection count (no embedding fn needed for count)
client = chromadb.PersistentClient(path=chroma_path)
cols = client.list_collections()
print(f"Collections found: {[c.name for c in cols]}")

for col in cols:
    c = client.get_collection(col.name)
    print(f"  {col.name}: {c.count()} chunks")

print()

# 2. Try a raw query using requests directly (bypass ChromaDB embedding fn)
print("Testing direct embedding + manual query...")
resp = requests.post(
    f"{ollama_url}/api/embeddings",
    json={"model": embed_model, "prompt": "mineur contrat crédit"},
    timeout=60,
)
vec = resp.json().get("embedding", [])
print(f"Query vector length: {len(vec)}")

# 3. Try ChromaDB query with the embedding fn
from llm.langgraph_agent import _get_chroma_collection

collection = _get_chroma_collection()
print(
    f"\nCollection via _get_chroma_collection(): {collection.name}, count={collection.count()}"
)

result = collection.query(
    query_embeddings=[vec],
    n_results=3,
    include=["documents", "distances"],
)
docs = (result.get("documents") or [[]])[0]
distances = (result.get("distances") or [[]])[0]
print(f"Query results: {len(docs)} docs")
for i, (d, dist) in enumerate(zip(docs, distances)):
    print(f"  [{i}] dist={dist:.4f} | {d[:80]!r}")
