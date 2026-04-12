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

EMBEDDING_MODELS = {
    "bge-small-en-v1.5": "BAAI/bge-small-en-v1.5",
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
}


def get_embedding_model(model_name=None):
    """Get the embedding model (needed for both building and querying)."""
    name = model_name or EMBEDDING_MODEL_NAME
    return HuggingFaceEmbeddings(model_name=name)


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


def build_vector_store(docs, model_name=None):
    """Create FAISS index from documents."""
    embedding_model = get_embedding_model(model_name)
    vector_store = FAISS.from_documents(docs, embedding_model)
    return vector_store, embedding_model


def get_index_path_for_model(model_key):
    """Get the FAISS index path for a specific embedding model."""
    return os.path.join(ROOT_DIR, f"faiss_index_{model_key}")


def build_vector_store_for_model(model_key):
    """Load pre-built FAISS index for model, or build from scratch."""
    model_name = EMBEDDING_MODELS.get(model_key, EMBEDDING_MODEL_NAME)
    index_path = get_index_path_for_model(model_key)
    embedding_model = get_embedding_model(model_name)

    if os.path.exists(os.path.join(index_path, "index.faiss")):
        return FAISS.load_local(
            index_path, embedding_model,
            allow_dangerous_deserialization=True,
        )

    docs = load_documents()
    vector_store = FAISS.from_documents(docs, embedding_model)
    vector_store.save_local(index_path)
    return vector_store


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


def retrieve_with_scores(vector_store, query, k=5):
    """Retrieve top-k chunks with their FAISS L2 distances."""
    results = vector_store.similarity_search_with_score(query, k=k)
    docs = [doc for doc, _ in results]
    scores = [score for _, score in results]
    return docs, scores


def build_prompt(query, retrieved_docs, dashboard_context=None, previous_context=None):
    """Build the full prompt with dashboard context + RAG context.

    previous_context (optional) holds the most recent prior backtest snapshot —
    injected so the LLM can explain why results changed when the user tweaks
    parameters. Same shape as dashboard_context.
    """
    rag_parts = []
    for i, doc in enumerate(retrieved_docs, 1):
        rag_parts.append(f"[{i}] {doc.page_content}")
    rag_context = "\n\n".join(rag_parts)

    prompt = ""

    # Always include system methodology so LLM can explain calculations
    cap_str = f"${dashboard_context['params']['capital']:,}" if dashboard_context else "$10,000"
    pos_str = f"{dashboard_context['params']['position_pct']}%" if dashboard_context else "100%"
    prompt += (
        "## System Methodology (how our backtest engine works)\n"
        "Trading Logic (Elder Triple Screen):\n"
        "- Screen 1 (Trend): EMA_short > EMA_long → uptrend; < → downtrend\n"
        "- Screen 2 (Momentum): MACD line > Signal line → bullish momentum\n"
        "- Screen 3 (Entry): Long when Close > N-day highest high; "
        "Short when Close < N-day lowest low\n"
        "- Exit: Long closed when RSI > upper threshold (overbought); "
        "Short closed when RSI < lower threshold (oversold)\n"
        f"- Position sizing: {pos_str} of available cash per trade, "
        "shares = int(cash × position_pct / 100 // price)\n\n"
        "Indicator Calculations:\n"
        "- EMA: Exponential Moving Average = ewm(span=N, adjust=False)\n"
        "- MACD: MACD_line = EMA(12) - EMA(26); Signal = EMA(9) of MACD_line\n"
        "- RSI: RSI = 100 - 100/(1 + avg_gain/avg_loss) over 14-day rolling window\n\n"
        "Performance Metrics:\n"
        f"- Starting capital: {cap_str}\n"
        f"- Total Return = (final_equity - starting_capital) / starting_capital × 100%\n"
        "- Sharpe Ratio = (mean(daily_returns) / std(daily_returns)) × sqrt(252). "
        "This is the annualized Sharpe with risk-free rate assumed to be 0. "
        "daily_returns = daily percentage change of the equity curve.\n"
        "- Max Drawdown = max((peak - equity) / peak) × 100%. "
        "Peak is the running maximum of the equity curve.\n"
        "- Trade Count = total number of long entries + short entries\n"
        "- Risk Level: Low if MaxDD < 15%, Medium if < 30%, High if >= 30%\n\n"
    )

    if previous_context:
        pp = previous_context["params"]
        pm = previous_context["metrics"]
        prompt += (
            f"## Previous Backtest State (before the user's latest change)\n"
            f"- Ticker: {previous_context['ticker']}\n"
            f"- Date Range: {previous_context['start_date']} to {previous_context['end_date']}\n"
            f"- Parameters: EMA={pp['win_short']}/{pp['win_long']}, "
            f"RSI={pp['rsi_lower']}-{pp['rsi_upper']}, "
            f"Breakout={pp['breakout_window']}, "
            f"Capital=${pp.get('capital', 10000):,}, "
            f"PositionSize={pp.get('position_pct', 100)}%\n"
            f"- Results: Return {pm['total_return']}%, "
            f"Sharpe {pm['sharpe']}, "
            f"MaxDD -{pm['max_drawdown']}%, "
            f"Trades {pm['trade_count']}, "
            f"Risk {pm['risk_level']}\n\n"
        )

    if dashboard_context:
        p = dashboard_context["params"]
        m = dashboard_context["metrics"]
        prompt += (
            f"## Current Dashboard State\n"
            f"- Ticker: {dashboard_context['ticker']}\n"
            f"- Date Range: {dashboard_context['start_date']} to {dashboard_context['end_date']}\n"
            f"- Strategy: Elder Triple Screen\n"
            f"- Parameters:\n"
            f"  - EMA Short Window: {p['win_short']} days (tactical trend)\n"
            f"  - EMA Long Window: {p['win_long']} days (strategic trend)\n"
            f"  - RSI Oversold: {p['rsi_lower']} / Overbought: {p['rsi_upper']}\n"
            f"  - Breakout Window: {p['breakout_window']} days "
            f"(enter long when price exceeds {p['breakout_window']}-day high, "
            f"short when below {p['breakout_window']}-day low)\n"
            f"  - Starting Capital: ${p.get('capital', 10000):,}\n"
            f"  - Position Size: {p.get('position_pct', 100)}% of cash per trade\n\n"
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
        f"Instructions:\n"
        f"- When the question is about the current setup or performance, "
        f"reference the specific dashboard data and backtest results above.\n"
        f"- When a Previous Backtest State section is present AND the user is "
        f"asking why results changed (e.g. 'why did Sharpe drop?', 'why is "
        f"drawdown worse?', 'what changed?'), explicitly compare the two "
        f"states — call out which parameters changed and tie each change to "
        f"the specific metric it most likely affected.\n"
        f"- When the question is about Elder's concepts, use the retrieved knowledge.\n"
        f"- Explain parameters in plain language "
        f"(e.g. 'Breakout Window of 5' means the strategy enters when price "
        f"breaks the 5-day high/low, not a 5% move).\n"
        f"- Cite Elder's insights naturally (e.g. 'As Elder teaches...' "
        f"or 'Elder emphasizes...'), not as numbered references like [1].\n"
        f"- Keep the answer concise but informative."
    )

    return prompt


def generate_response(prompt, model_choice, groq_key=None, openai_key=None):
    """Generate a response using the selected model.

    API keys are read from environment variables (.env).
    The optional key parameters are kept for backward compatibility.
    """
    system_msg = (
        "You are Elder Trading Copilot, an AI assistant "
        "specialized in Alexander Elder's trading strategies. "
        "Answer questions using the provided knowledge and "
        "dashboard context. Be helpful, accurate, and concise."
    )

    groq_models = {
        "Llama 3.3 70B (Groq)": "llama-3.3-70b-versatile",
        "Llama 3.1 8B (Groq)": "llama-3.1-8b-instant",
        "Mixtral 8x7B (Groq)": "mixtral-8x7b-32768",
    }
    if model_choice in groq_models:
        key = groq_key or os.getenv("GROQ_API_KEY")
        if not key:
            return "Error: GROQ_API_KEY not found. Please add it to your .env file."
        client = Groq(api_key=key)
        response = client.chat.completions.create(
            model=groq_models[model_choice],
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
            return "Error: OPENAI_API_KEY not found. Please add it to your .env file."
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
