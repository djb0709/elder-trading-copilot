"""
Pre-build FAISS index and save to disk.
Run once:  python build_index.py
"""

from rag import load_documents, build_vector_store, FAISS_INDEX_PATH

print("Loading documents...")
docs = load_documents()
print(f"Loaded {len(docs)} documents")

print("Building embeddings & FAISS index (this takes 30-60s)...")
vector_store, _ = build_vector_store(docs)

print(f"Saving index to {FAISS_INDEX_PATH}/")
vector_store.save_local(FAISS_INDEX_PATH)

print("Done! Next time app.py will load instantly.")
