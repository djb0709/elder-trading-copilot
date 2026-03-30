# Elder Trading Copilot

A context-aware intelligent trading assistant that combines an interactive backtesting dashboard with a RAG-powered chatbot, built around Alexander Elder's Triple Screen Trading System.

![Dashboard](data/Dashboard.jpeg)

---

## Features

- **Interactive Backtest Dashboard** -- Adjust strategy parameters (EMA windows, RSI thresholds, breakout window, position sizing) via sidebar controls and see charts and metrics update in real time.
- **Three Linked Charts** -- Price with buy/sell signals, RSI with overbought/oversold zones, and equity curve versus buy-and-hold baseline, all sharing a synchronized x-axis.
- **Context-Aware RAG Chatbot** -- An embedded copilot panel that reads the current dashboard state (ticker, parameters, backtest results, risk level) and injects it into every prompt for tailored answers.
- **Dual LLM Support** -- Switch between Llama 3.3 70B via Groq (free, fast) and GPT-4o-mini via OpenAI (high quality) with a single dropdown.
- **877-Entry Knowledge Base** -- QA pairs extracted from Elder's books and video lectures, embedded with BGE and indexed in FAISS for sub-millisecond retrieval.
- **RAG Pipeline Explorer** -- A dedicated debug page that visualizes every stage of the pipeline: query embedding, chunk retrieval with metadata, assembled prompt, and timed LLM generation.
- **Quick Ask Buttons** -- One-click queries such as "Explain Setup", "Explain Perf", and "Risk?" for instant context-aware analysis.

---

## Architecture

```
+------------------------------------------------------------------+
|                         Streamlit UI                              |
|  +----------------------------+-------------------------------+  |
|  |  Dashboard (60%)           |  Copilot Panel (40%)          |  |
|  |                            |                               |  |
|  |  Metric Cards:             |  Context Summary Bar          |  |
|  |    Return / Sharpe /       |  Model Selector (Groq/OpenAI) |  |
|  |    MaxDD / Trades / Risk   |  Chat History                 |  |
|  |                            |  Quick Ask Buttons             |  |
|  |  Chart 1: Price + Signals  |  Chat Input                   |  |
|  |  Chart 2: RSI              |                               |  |
|  |  Chart 3: Equity vs B&H   |                               |  |
|  +-------------+--------------+---------------+---------------+  |
|                |                              |                  |
+----------------+------------------------------+------------------+
                 |                              |
                 v                              v
       +-------------------+          +----------------------+
       | Backtest Engine   |          |   RAG Pipeline       |
       |                   |          |                      |
       | - yfinance data   |          | 1. Load FAISS index  |
       | - EMA / RSI / MACD|          | 2. Embed user query  |
       | - Triple Screen   |          |    (bge-small-en)    |
       |   entry logic     |          | 3. Top-k retrieval   |
       | - Position sizing |          | 4. Build prompt      |
       | - Metrics calc    |          |                      |
       +---------+---------+          +----------+-----------+
                 |                               |
                 v                               v
       +-------------------+          +----------------------+
       | Context Builder   |--------->| Generation Layer     |
       |                   |          |                      |
       | - ticker / dates  |          | - Groq Llama 3.3 70B |
       | - strategy params |          | - OpenAI GPT-4o-mini |
       | - backtest metrics|          |                      |
       | - risk level      |          |                      |
       +-------------------+          +----------------------+
```

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| UI | Streamlit (multi-page) | Dashboard, chatbot panel, and RAG debug page |
| Market Data | yfinance | Free historical OHLCV data |
| Indicators | pandas / numpy | EMA, RSI, MACD computed from scratch |
| Charts | Plotly (subplots) | Interactive, linked price/RSI/equity charts |
| Embedding | BAAI/bge-small-en-v1.5 | 384-dim dense vectors via HuggingFace |
| Vector Store | FAISS (IndexFlatL2) | Exact L2 nearest-neighbor search |
| Text Splitting | LangChain RecursiveCharacterTextSplitter | Chunk raw text (400 chars, 50 overlap) |
| LLM (Option A) | Llama 3.3 70B via Groq | Free tier, fast inference |
| LLM (Option B) | GPT-4o-mini via OpenAI | High quality, low cost |
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

# 5. Build the FAISS index (one-time, ~30-60 seconds)
python build_index.py

# 6. Launch the application
streamlit run dashboard.py
```

The app opens at `http://localhost:8501`. Use the sidebar to select a ticker and adjust strategy parameters; the copilot panel on the right answers questions with full dashboard context.

> **Note:** If `faiss_index/` already exists (included in the repo), you can skip step 5 and go straight to step 6.

---

## Project Structure

```
elder-trading-copilot/
├── dashboard.py              # Main entry point: dashboard + embedded copilot panel
├── rag.py                    # RAG pipeline: load docs, build/query FAISS, generate
├── build_index.py            # One-time script to pre-build and save the FAISS index
├── requirements.txt          # Python dependencies
├── .env.example              # Template for API keys
├── pages/
│   └── chatbot.py            # RAG Pipeline Explorer (debug/visualization page)
├── data/
│   ├── knowledge_base.csv    # 877 QA pairs from Elder's books and teachings
│   └── elder_new_high_new_low.txt   # Video lecture transcript (~5,400 chars)
├── faiss_index/
│   ├── index.faiss           # Serialized FAISS index
│   └── index.pkl             # Document metadata store
└── docs/
    └── System_Methodology.md # Detailed formulas for every indicator and metric
```

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
bge-small-en-v1.5 encodes the query into a 384-dim vector
    |
    v
FAISS IndexFlatL2 retrieves the top-k most similar chunks
    |
    v
Context Builder assembles:
  - System methodology (indicator formulas, trading logic)
  - Dashboard state (ticker, params, backtest metrics)
  - Retrieved knowledge chunks
  - User question + instructions
    |
    v
Llama 3.3 70B (Groq) or GPT-4o-mini (OpenAI) generates the answer
    |
    v
Response displayed in the copilot panel
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

This project was developed for educational purposes as part of the CSYE 7380 course at Northeastern University.
