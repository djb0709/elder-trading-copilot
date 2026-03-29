# Elder Trading Copilot

A context-aware intelligent trading assistant combining a **Trading Strategy Dashboard** with a **RAG-based Chatbot**, built around Alexander Elder's trading philosophy.

## Features

- **Interactive Dashboard** — Backtest Elder's Triple Screen strategy with adjustable parameters, 3 linked charts (Price + Signals / RSI / Equity Curve), and real-time updates
- **RAG Chatbot** — Ask questions about Elder's trading strategies, grounded in 877 QA pairs extracted from his books
- **Context-Aware Answers** — Chatbot reads current dashboard state (ticker, parameters, backtest results) and provides tailored explanations
- **Dual Model Support** — Switch between Llama 3.3 70B (Groq) and GPT-4o-mini (OpenAI)
- **RAG Debug Page** — Visualize the full retrieval pipeline: retrieved chunks, prompt, and generation

## Project Structure

```
elder-trading-copilot/
├── Dashboard.py              # Main app: dashboard + chatbot
├── rag.py                    # RAG pipeline (embed, index, retrieve, generate)
├── build_index.py            # One-time script to pre-build FAISS index
├── requirements.txt          # Python dependencies
├── .env.example              # API key template
├── pages/
│   └── Chatbot.py            # Standalone RAG debug page
├── data/
│   ├── knowledge_base.csv    # 877 QA pairs (Elder's books + teachings)
│   └── elder_new_high_new_low.txt  # Video transcript
├── faiss_index/              # Pre-built vector index
│   ├── index.faiss
│   └── index.pkl
└── docs/
    ├── Project.md            # Project plan (Chinese)
    ├── Dashboard_Plan.md     # Dashboard design spec
    └── RAG_Plan.md           # RAG pipeline design spec
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/elder-trading-copilot.git
cd elder-trading-copilot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API keys
cp .env.example .env
# Edit .env with your Groq and/or OpenAI API keys

# 4. Run (FAISS index is pre-built, no build step needed)
streamlit run Dashboard.py
```

### Rebuild FAISS Index (optional)

Only needed if you modify `data/knowledge_base.csv`:

```bash
python build_index.py
```

## Tech Stack

| Layer | Choice |
|---|---|
| UI | Streamlit (multi-page) |
| Data | yfinance |
| Indicators | pandas (EMA, RSI, MACD) |
| Embedding | BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS (IndexFlatL2) |
| Retrieval | Top-k similarity search |
| Generation | Groq Llama 3.3 70B / OpenAI GPT-4o-mini |
| Framework | LangChain |

## Trading Strategy

Implements a simplified version of Elder's **Triple Screen Trading System**:

| Screen | Purpose | Implementation |
|---|---|---|
| Screen 1 (Trend) | Identify market direction | EMA Short > EMA Long |
| Screen 2 (Momentum) | Confirm strength | MACD > MACD Signal |
| Screen 3 (Entry) | Time the entry | Price breaks above N-day high |
| Exit | Take profits | RSI crosses overbought/oversold |

## Knowledge Base

877 QA pairs sourced from:

| Source | Count |
|---|---|
| Base Knowledge Dataset | 635 |
| *The New Trading for a Living* (manual extraction) | 50 |
| *Study Guide for Come Into My Trading Room* (auto-extracted) | 100 |
| Video transcript (chunked) | ~15 segments |

## Screenshots

*Coming soon*

## License

This project is for educational purposes as part of CSYE 7380 coursework.
