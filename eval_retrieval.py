"""
Retrieval evaluation: query the knowledge base with questions and measure
Recall@K across embedding models and Top-K values.

Method:
  - Corpus = all Answer texts from the knowledge base.
  - Use every Question in the knowledge base as a query (full set, no sampling).
  - Retrieve Top-K; a query counts as a hit if its paired Answer is returned.
  - Metric: Recall@K = hits / number of questions.
  - Querying answers with questions (rather than questions with questions)
    reflects real usage and avoids inflated scores from matching identical text.

Run:
  python eval_retrieval.py
  (Reuses the embedding definitions from rag.py; indexes are built in memory.)
"""

import csv
import os
import sys

# Windows console defaults to cp1252; force UTF-8 output.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from rag import DATA_DIR, EMBEDDING_MODELS
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

KB_PATH = os.path.join(DATA_DIR, "knowledge_base.csv")

K_VALUES = [1, 3, 5, 10]   # Top-K values to evaluate

# BGE v1.5 expects this instruction on the query side only (not on passages).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def get_eval_embedding(model_name, query_prefix=None):
    """Embedding model for evaluation; optionally prepend a query-side prefix."""
    if query_prefix:
        return HuggingFaceEmbeddings(
            model_name=model_name,
            query_encode_kwargs={"prompt": query_prefix},
        )
    return HuggingFaceEmbeddings(model_name=model_name)


def load_qa_pairs():
    """Load the knowledge base as [(question, answer), ...], skipping blanks."""
    pairs = []
    with open(KB_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = (row.get("Question") or "").strip()
            a = (row.get("Answer") or "").strip()
            if q and a:
                pairs.append((q, a))
    return pairs


def build_answer_index(answers, model_name, query_prefix=None):
    """Build a FAISS index over Answer texts; keep the row id in metadata."""
    embedding_model = get_eval_embedding(model_name, query_prefix)
    docs = [
        Document(page_content=ans, metadata={"row": i})
        for i, ans in enumerate(answers)
    ]
    return FAISS.from_documents(docs, embedding_model)


def evaluate(vector_store, questions, target_answers, k_values):
    """Compute Recall@K. A hit is decided by answer-text equality, so
    duplicate answers are handled correctly."""
    max_k = max(k_values)
    hits = {k: 0 for k in k_values}
    for q, target in zip(questions, target_answers):
        results = vector_store.similarity_search(q, k=max_k)
        retrieved = [d.page_content for d in results]
        for k in k_values:
            if target in retrieved[:k]:
                hits[k] += 1
    n = len(questions)
    return {k: hits[k] / n for k in k_values}


def main():
    pairs = load_qa_pairs()
    answers = [a for _, a in pairs]
    print(f"Total answers in knowledge base: {len(answers)}")

    questions = [q for q, _ in pairs]
    target_answers = answers
    print(f"Queries: {len(questions)} (all questions)\n")

    header = "Embedding model".ljust(28) + "".join(
        f"R@{k}".rjust(9) for k in K_VALUES
    )
    print(header)
    print("-" * len(header))

    for key, model_name in EMBEDDING_MODELS.items():
        variants = [(key, None)]
        if model_name.startswith("BAAI/bge"):
            variants.append((key + " +prefix", BGE_QUERY_PREFIX))
        for label, prefix in variants:
            vector_store = build_answer_index(answers, model_name, prefix)
            recall = evaluate(vector_store, questions, target_answers, K_VALUES)
            row = label.ljust(28) + "".join(
                f"{recall[k] * 100:8.1f}%" for k in K_VALUES
            )
            print(row)

    print("\nR@K = share of questions whose answer appears in the Top-K results.")


if __name__ == "__main__":
    main()
