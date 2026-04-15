"""seed_db.py — version diagnostic"""
from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(override=True)

# ── 1. Vérifier les PDFs ──────────────────────────────────────────────────────
import llm.langgraph_agent as ag

project_root = Path(ag.__file__).resolve().parents[1]
docs_dir = project_root / "rag_docs"
print(f"[DIAG] rag_docs path : {docs_dir}")
print(f"[DIAG] exists        : {docs_dir.exists()}")
if docs_dir.exists():
    pdfs = list(docs_dir.glob("*.pdf"))
    print(f"[DIAG] PDFs trouvés  : {[p.name for p in pdfs]}")
else:
    print("[DIAG] ❌ Le dossier rag_docs n'existe pas !")

# ── 2. Tester l'extraction PDF ────────────────────────────────────────────────
if docs_dir.exists():
    for pdf in docs_dir.glob("*.pdf"):
        text = ag._extract_pdf_text(pdf)
        print(f"[DIAG] {pdf.name} → {len(text)} chars extraits")
        if text:
            print(f"         Aperçu: {text[:200]!r}")
        else:
            print(f"         ❌ Aucun texte extrait !")

# ── 3. Tester le chunking ─────────────────────────────────────────────────────
        if text:
            chunks = ag.chunk_legal_pdf(text)
            print(f"         Chunks: {len(chunks)}")
            if chunks:
                print(f"         Premier chunk: {chunks[0]['text'][:150]!r}")

# ── 4. Tester la connexion Chroma ─────────────────────────────────────────────
print("\n[DIAG] Test connexion ChromaDB...")
try:
    col = ag._get_chroma_collection()
    print(f"[DIAG] Collection : {col.name}")
    print(f"[DIAG] Count avant seed : {col.count()}")
except Exception as e:
    print(f"[DIAG] ❌ Erreur Chroma : {e}")

# ── 5. Seeder avec logs verbeux ───────────────────────────────────────────────
print("\n[DIAG] Lancement seed_chroma()...")
ag.seed_chroma()

try:
    col = ag._get_chroma_collection()
    print(f"[DIAG] Count après seed : {col.count()}")
    if col.count() > 0:
        # Utilise query_embeddings au lieu de query_texts
        vec = ag._CHROMA_EMBEDDING_FN._embed_one("contrat")
        sample = col.query(
            query_embeddings=[vec],
            n_results=2,
            include=["documents"],
        )
        print(f"[DIAG] Test query 'contrat': {sample['documents']}")
    else:
        print("[DIAG] ❌ Toujours vide après seed !")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"[DIAG] ❌ Erreur post-seed : {e}")