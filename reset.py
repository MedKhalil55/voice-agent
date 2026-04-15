# run once: reset_chroma.py
import chromadb, os
from dotenv import load_dotenv
load_dotenv(override=True)

client = chromadb.PersistentClient(path=os.environ.get("VOICE_AGENT_CHROMA_PATH", "artifacts/chroma"))

for col in client.list_collections():
    print(f"Deleting: {col.name}")
    client.delete_collection(col.name)

print("Done. Now run seed_db.py")
