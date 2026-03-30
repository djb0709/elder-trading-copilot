# Elder Trading Copilot

A context-aware intelligent trading assistant that combines an interactive backtesting dashboard with a RAG-powered chatbot, built around Alexander Elder's Triple Screen Trading System.

![Dashboard](data/Dashboard.jpeg)

---

## Features

- **Interactive Backtest Dashboard** -- Adjust strategy parameters (EMA windows, RSI thresholds, breakout window), starting capital, and position sizing via sidebar controls. Charts and metrics update in real time.
- **Five Linked Charts** -- Price with buy/sell signals, MACD histogram (Screen 2), RSI with overbought/oversold zones, equity curve versus buy-and-hold baseline, and drawdown curve, all sharing a synchronized x-axis. Each chart panel can be toggled on or off.
- **Context-Aware RAG Chatbot** -- An embedded copilot panel with streaming word-by-word output that reads the current dashboard state (ticker, parameters, backtest results, risk level) and injects it into every prompt for tailored answers.
- **Multi-Model Support** -- Switch between four LLMs (Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B via Groq; GPT-4o-mini via OpenAI) and three embedding models (bge-small, bge-base, all-MiniLM-L6-v2) with a single dropdown.
- **894-Entry Knowledge Base** -- QA pairs extracted from Elder's books and video lectures, embedded and indexed in FAISS for sub-millisecond retrieval.
- **RAG Pipeline Explorer** -- A dedicated page that visualizes every stage of the pipeline: query embedding, chunk retrieval with metadata, assembled prompt, and timed LLM generation.
- **Pipeline Compare** -- Side-by-side comparison page for embedding models, LLMs, and top-k values. See how different configurations affect retrieval quality and response content.
- **Trade Log** -- Expandable table showing every closed trade with entry/exit dates, prices, shares, P&L, return percentage, days held, and summary statistics (wins, losses, win rate, total P&L).
- **Celebration Animations** -- Balloons on metric improvement, snow on decline. Toggleable from the sidebar.

---

## Architecture

```
+------------------------------------------------------------------+
|                         Streamlit UI                              |
|  +----------------------------+-------------------------------+  |
|  |  Dashboard (60%)           |  Copilot Panel (40%)          |  |
|  |                            |                               |  |
|  |  Metric Cards:             |  Embedding + LLM Selectors    |  |
|  |    Return / Sharpe /       |  Context Summary Bar          |  |
|  |    MaxDD / Trades / Risk   |  Chat History (streaming)     |  |
|  |                            |  Quick Ask Buttons             |  |
|  |  Chart 1: Price + Signals  |  Chat Input                   |  |
|  |  Chart 2: MACD (Screen 2)  |                               |  |
|  |  Chart 3: RSI              |                               |  |
|  |  Chart 4: Equity vs B&H   |                               |  |
|  |  Chart 5: Drawdown         |                               |  |
|  |  Trade Log Table           |                               |  |
|  +-------------+--------------+---------------+---------------+  |
|                |                              |                  |
+----------------+------------------------------+------------------+
                 |                              |
                 v                              v
       +-------------------+          +----------------------+
       | Backtest Engine   |          |   RAG Pipeline       |
       |                   |          |                      |
       | - yfinance data   |          | 1. Select embedding  |
       | - EMA / RSI / MACD|          |    model             |
       | - Triple Screen   |          | 2. Load FAISS index  |
       |   entry logic     |          | 3. Embed user query  |
       | - Position sizing |          | 4. Top-k retrieval   |
       | - Metrics calc    |          | 5. Build prompt      |
       +---------+---------+          +----------+-----------+
                 |                               |
                 v                               v
       +-------------------+          +----------------------+
       | Context Builder   |--------->| Generation Layer     |
       |                   |          |                      |
       | - ticker / dates  |          | - Llama 3.3 70B      |
       | - strategy params |          | - Llama 3.1 8B       |
       | - backtest metrics|          | - Mixtral 8x7B       |
       | - risk level      |          | - GPT-4o-mini        |
       +-------------------+          +----------------------+
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI | Streamlit (multi-page, fragments) | Dashboard, chatbot panel, pipeline explorer, compare page |
| Market Data | yfinance | Free historical OHLCV data |
| Indicators | pandas / numpy | EMA, RSI, MACD computed from scratch |
| Charts | Plotly (subplots, per-panel legends) | Interactive, linked 5-chart layout |
| Embedding (Option A) | BAAI/bge-small-en-v1.5 | 384-dim dense vectors, fast and light |
| Embedding (Option B) | BAAI/bge-base-en-v1.5 | 768-dim dense vectors, higher quality |
| Embedding (Option C) | sentence-transformers/all-MiniLM-L6-v2 | 384-dim, classic baseline |
| Vector Store | FAISS (IndexFlatL2) | Exact L2 nearest-neighbor search, pre-built per model |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | Chunk raw text (400 chars, 50 overlap) |
| LLM | Llama 3.3 70B, Llama 3.1 8B, Mixtral 8x7B (Groq); GPT-4o-mini (OpenAI) | Four model options with different speed/quality tradeoffs |
| Orchestration | LangChain | Unified document, embedding, and vector store abstractions |
| Config | python-dotenv | API key management via `.env` |

---

## Quick Start

### Prerequisites

- Python 3.9 or later
- A Groq API key and/or an OpenAI API key

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/AlanY1an/elder-trading-copilot.git
cd elder-trading-copilot

# 2. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # macOS / Linux
# venv\Scripts\activate    # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up API keys
cp .env.example .env
# Edit .env and add your keys:
#   GROQ_API_KEY=gsk_...
#   OPENAI_API_KEY=sk-...

# 5. Build FAISS indexes for all embedding models (one-time, ~2 minutes)
python build_index.py

# 6. Launch the application
streamlit run dashboard.py
```

The app opens at `http://localhost:8501`. Use the sidebar to select a ticker and adjust strategy parameters; the copilot panel on the right answers questions with full dashboard context.

> **Note:** Pre-built FAISS indexes for all three embedding models are included in the repo. You can skip step 5 and go straight to step 6.

---

## Project Structure

```
elder-trading-copilot/
├── dashboard.py              # Main entry point: dashboard + embedded copilot panel
├── rag.py                    # RAG pipeline: load docs, build/query FAISS, generate
├── components.py             # Shared UI components (sidebar header, about dialog)
├── build_index.py            # Pre-build FAISS indexes for all embedding models
├── requirements.txt          # Python dependencies
├── .env.example              # Template for API keys
├── .streamlit/
│   └── config.toml           # Streamlit theme config (primary color)
├── pages/
│   ├── chatbot.py            # RAG Pipeline Explorer (step-by-step visualization)
│   └── compare.py            # Pipeline Compare (side-by-side embedding/LLM/top-k)
├── data/
│   ├── knowledge_base.csv    # 877 QA pairs from Elder's books and teachings
│   ├── elder_new_high_new_low.txt   # Video lecture transcript (~5,400 chars)
│   ├── elder_jpa.png         # Elder avatar for sidebar
│   └── anime_edit.png        # Elder anime avatar for chatbot
├── faiss_index/              # Default FAISS index (bge-small-en-v1.5)
├── faiss_index_bge-small-en-v1.5/
├── faiss_index_bge-base-en-v1.5/
├── faiss_index_all-MiniLM-L6-v2/
└── docs/
    └── System_Methodology.md # Detailed formulas for every indicator and metric
```

---

## Pages

### Dashboard
The main page with an interactive backtesting dashboard on the left and a RAG-powered chatbot on the right. Adjustable parameters include ticker, date range, EMA windows, RSI thresholds, breakout window, starting capital, and position size. The chatbot streams responses word-by-word and supports quick-ask buttons.

### Pipeline Explorer
Visualizes each stage of the RAG pipeline for every query: embedding, retrieval (with chunk content and metadata), prompt assembly, and LLM generation. Displays timing for each step. Supports switching between all embedding and LLM models.

### Pipeline Compare
Side-by-side comparison of RAG configurations:
- **Embedding Models** -- Compare retrieved chunks from different embedding models (retrieval only, no LLM cost).
- **LLM Models** -- Same retrieval, different LLM responses. Choose the shared embedding model.
- **Top-K Values** -- See how the number of retrieved chunks affects the response.

---

## How It Works

### Backtest Engine

The backtest implements a simplified version of Elder's **Triple Screen Trading System**:

| Screen | Role | Condition (Long) | Condition (Short) |
|---|---|---|---|
| Screen 1 -- Trend | Filter direction | EMA Short > EMA Long | EMA Short < EMA Long |
| Screen 2 -- Momentum | Confirm strength | MACD > Signal line | MACD < Signal line |
| Screen 3 -- Entry | Time the trade | Close > N-day high | Close < N-day low |

All three screens must agree before a position is opened. Exits are triggered by RSI: long positions close when RSI exceeds the overbought threshold; short positions close when RSI drops below the oversold threshold.

Position sizing allocates a configurable percentage of available cash per trade (default 100%). The engine computes five metrics: Total Return, Sharpe Ratio, Max Drawdown, Trade Count, and Risk Level.

### RAG Pipeline

```
User Question
    |
    v
Select embedding model (bge-small / bge-base / all-MiniLM-L6-v2)
    |
    v
Encode query into dense vector (384 or 768 dimensions)
    |
    v
FAISS IndexFlatL2 retrieves top-k most similar chunks
    |
    v
Context Builder assembles:
  - System methodology (indicator formulas, trading logic)
  - Dashboard state (ticker, params, backtest metrics)
  - Retrieved knowledge chunks
  - User question + instructions
    |
    v
LLM generates the answer (streaming word-by-word)
    |
    v
Response displayed in copilot panel with model metadata
```

The context-aware design means the chatbot can explain why a specific parameter setting produced a particular Sharpe ratio or drawdown, referencing both Elder's teachings and the live backtest data.

---

## Knowledge Base

The knowledge base contains **877 QA pairs** plus supplementary text, drawn from three sources:

| Source | Entries | Description |
|---|---|---|
| Base Knowledge Dataset | 635 | Elder strategy, risk management, psychology, timing, adaptability |
| *The New Trading for a Living* | 50 | Hand-extracted QA on MA, RSI, MACD, Triple Screen, risk rules |
| *Study Guide for Come Into My Trading Room* | 100 | Auto-extracted QA pairs from the study guide |
| Elder video lecture transcript | ~15 chunks | `elder_new_high_new_low.txt`, split at 400 chars with 50-char overlap |

QA pairs are not chunked further -- each question-answer pair is embedded as a single unit so that the question portion aids retrieval while the answer provides the knowledge.

---

## License

This project was developed for educational purposes as part of the CSYE 7380 course (Theory and Practice of Applied AI Generative Models) at Northeastern University, taught by Dr. Yizhen Zhao.
