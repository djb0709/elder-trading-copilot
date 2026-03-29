"""
RAG Pipeline for Elder Trading Copilot.

Loads knowledge base (CSV QA pairs + text files), builds FAISS index,
and provides retrieval + generation functions.

Usage:
  1. Run `python build_index.py` once to pre-build the FAISS index.
  2. App loads the saved index instantly on startup.
  3. If no saved index found, builds from scratch (slower).
"""

import os
import csv
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from groq import Groq
from openai import OpenAI

load_dotenv()

# Paths (all relative to this file's directory = repo root)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
KB_PATH = os.path.join(DATA_DIR, "knowledge_base.csv")
TEXT_PATH = os.path.join(DATA_DIR, "elder_new_high_new_low.txt")
FAISS_INDEX_PATH = os.path.join(ROOT_DIR, "faiss_index")

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


def get_embedding_model():
    """Get the embedding model (needed for both building and querying)."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def load_documents():
    """Load QA pairs from CSV and raw text files into Document objects."""
    docs = []

    with open(KB_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            content = f"{row['Question']} {row['Answer']}"
            metadata = {
                "source": row.get("Source", ""),
                "label": row.get("Label", ""),
            }
            docs.append(Document(page_content=content, metadata=metadata))

    if os.path.exists(TEXT_PATH):
        with open(TEXT_PATH, encoding="utf-8") as f:
            text = f.read()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400, chunk_overlap=50
        )
        text_docs = splitter.create_documents(
            [text],
            metadatas=[{"source": "Elder Video Lecture", "label": "Video"}],
        )
        docs.extend(text_docs)

    return docs


def build_vector_store(docs):
    """Create FAISS index from documents."""
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(docs, embedding_model)
    return vector_store, embedding_model


def load_or_build_vector_store():
    """Load pre-built FAISS index from disk, or build from scratch."""
    embedding_model = get_embedding_model()

    if os.path.exists(os.path.join(FAISS_INDEX_PATH, "index.faiss")):
        vector_store = FAISS.load_local(
            FAISS_INDEX_PATH,
            embedding_model,
            allow_dangerous_deserialization=True,
        )
        return vector_store
    else:
        docs = load_documents()
        vector_store = FAISS.from_documents(docs, embedding_model)
        return vector_store


def retrieve(vector_store, query, k=5):
    """Retrieve top-k most relevant chunks for a query."""
    return vector_store.similarity_search(query, k=k)


def build_prompt(query, retrieved_docs, dashboard_context=None):
    """Build the full prompt with dashboard context + RAG context."""
    rag_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        rag_parts.append(f"[{i}] {doc.page_content}")
    rag_context = "\n\n".join(rag_parts)

    prompt = ""

    if dashboard_context:
        p = dashboard_context["params"]
        m = dashboard_context["metrics"]
        prompt += (
            f"## Current Dashboard State\n"
            f"- Ticker: {dashboard_context['ticker']}\n"
            f"- Date Range: {dashboard_context['start_date']} to {dashboard_context['end_date']}\n"
            f"- Strategy: Elder Triple Screen\n"
            f"- Parameters: EMA={p['win_short']}/{p['win_long']}, "
            f"RSI={p['rsi_lower']}-{p['rsi_upper']}, "
            f"Breakout={p['breakout_window']}\n\n"
            f"## Backtest Results\n"
            f"- Total Return: {m['total_return']}%\n"
            f"- Sharpe Ratio: {m['sharpe']}\n"
            f"- Max Drawdown: -{m['max_drawdown']}%\n"
            f"- Trade Count: {m['trade_count']}\n"
            f"- Risk Level: {m['risk_level']}\n\n"
        )

    prompt += (
        f"## Retrieved Knowledge (from Elder's books and teachings)\n"
        f"{rag_context}\n\n"
        f"## User Question\n{query}\n\n"
        f"Answer based on the retrieved knowledge and current dashboard state. "
        f"If the question is about the current setup or performance, use the dashboard data. "
        f"If the question is about Elder's strategy or concepts, use the retrieved knowledge. "
        f"Cite Elder's specific insights when relevant. Keep your answer concise but informative."
    )

    return prompt


def generate_response(prompt, model_choice, groq_key=None, openai_key=None):
    """Generate a response using the selected model."""
    system_msg = (
        "You are Elder Trading Copilot, an AI assistant "
        "specialized in Alexander Elder's trading strategies. "
        "Answer questions using the provided knowledge and "
        "dashboard context. Be helpful, accurate, and concise."
    )

    if "Groq" in model_choice:
        key = groq_key or os.getenv("GROQ_API_KEY")
        if not key:
            return "Please set GROQ_API_KEY in .env file."
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    else:
        key = openai_key or os.getenv("OPENAI_API_KEY")
        if not key:
            return "Please set OPENAI_API_KEY in .env file."
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return response.choices[0].message.content
