"""
Pre-build FAISS indexes for all embedding models.
Run once:  python build_index.py
"""

from rag import (
    load_documents, get_embedding_model,
    FAISS_INDEX_PATH, EMBEDDING_MODELS, get_index_path_for_model,
)
from langchain_community.vectorstores import FAISS

print("Loading documents...")
docs = load_documents()
print(f"Loaded {len(docs)} documents\n")

# Build default index (bge-small)
print("=== Building default index (bge-small-en-v1.5) ===")
embedding_model = get_embedding_model()
vector_store = FAISS.from_documents(docs, embedding_model)
vector_store.save_local(FAISS_INDEX_PATH)
print(f"Saved to {FAISS_INDEX_PATH}/\n")

# Build indexes for all other models
for key, model_name in EMBEDDING_MODELS.items():
    print(f"=== Building index for {key} ({model_name}) ===")
    index_path = get_index_path_for_model(key)
    embedding_model = get_embedding_model(model_name)
    vector_store = FAISS.from_documents(docs, embedding_model)
    vector_store.save_local(index_path)
    print(f"Saved to {index_path}/\n")

print("Done! All indexes pre-built. App will load instantly for any embedding model.")
