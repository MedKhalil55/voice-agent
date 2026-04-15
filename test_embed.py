# test_embed.py
from dotenv import load_dotenv
load_dotenv(override=True)

import os, requests

url = os.environ.get("VOICE_AGENT_OLLAMA_BASE_URL", "http://localhost:11434") + "/api/embeddings"
model = os.environ.get("VOICE_AGENT_EMBED_MODEL", "nomic-embed-text:latest")

print(f"Testing: {url} with model={model!r}")

resp = requests.post(url, json={"model": model, "prompt": "mineur contrat"}, timeout=60)
print(f"Status: {resp.status_code}")
data = resp.json()
vec = data.get("embedding", [])
print(f"Vector length: {len(vec)}")
print(f"First 5 values: {vec[:5]}")