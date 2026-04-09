"""Shared UI components across pages."""

import streamlit as st

# Ghost Text
import os
import streamlit as st
import streamlit.components.v1 as components
from typing import List, Optional


@st.dialog("About Elder Copilot")
def show_about():
    """About dialog with project info."""
    st.image("data/elder_jpa.png", width=100)
    st.markdown(
        "### Elder Trading Copilot\n"
        "A context-aware intelligent trading assistant built around "
        "Alexander Elder's Triple Screen Trading System.\n\n"
        "**Features:**\n"
        "- Interactive backtesting dashboard with 5 linked charts\n"
        "- RAG-powered chatbot with 877-entry knowledge base\n"
        "- Dual LLM support (Llama 3.3 via Groq / GPT-4o-mini)\n"
        "- Real-time context injection from dashboard to chatbot\n\n"
        "**Tech Stack:**\n"
        "Streamlit, Plotly, FAISS, LangChain, BGE Embeddings, "
        "Groq, OpenAI, yfinance\n\n"
        "**Course:** CSYE 7380 — Theory & Practice of Applied AI Generative Models\n\n"
        "**Instructor:** Dr. Yizhen Zhao\n\n"
        "Northeastern University, 2026"
    )
    if st.button("Close", use_container_width=True):
        st.rerun()


def render_sidebar_header():
    """Render the sidebar brand header with avatar, description, and navigation."""
    sb_img, sb_txt = st.columns([1, 2])
    with sb_img:
        st.image("data/elder_jpa.png", width=75)
    with sb_txt:
        st.markdown(
            "<div style='padding-top:2px;'>"
            "<strong style='font-size:0.9rem;'>Elder Copilot</strong><br>"
            "<span style='font-size:0.72rem; color:#888;'>"
            "RAG-powered trading AI"
            "</span></div>",
            unsafe_allow_html=True,
        )
    st.page_link("dashboard.py", label="Dashboard", icon=":material/candlestick_chart:")
    st.page_link("pages/chatbot.py", label="Pipeline Explorer", icon=":material/search:")
    st.page_link("pages/compare.py", label="Pipeline Compare", icon=":material/compare_arrows:")
    st.toggle("Celebrations", value=True, key="anim_on", help="Toggle balloons/snow animations")
    st.markdown(
        '<style>'
        '.st-key-about_trigger button {'
        '  background: none !important; border: none !important;'
        '  padding: 0 !important; margin: 0 !important;'
        '  min-height: 0 !important; height: auto !important;'
        '  font-size: 0.72rem !important; color: #888 !important;'
        '  cursor: pointer !important;'
        '}'
        '.st-key-about_trigger button:hover { color: #667eea !important; }'
        '.st-key-about_trigger button p {'
        '  font-size: 0.72rem !important; color: inherit !important;'
        '}'
        '</style>',
        unsafe_allow_html=True,
    )
    if st.button("About this project", key="about_trigger", type="tertiary"):
        show_about()
    st.divider()


# Path to the ghost_autocomplete folder (relative to this file)
_COMPONENT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ghost_autocomplete"
)

_ghost_component = components.declare_component(
    "ghost_autocomplete",
    path=_COMPONENT_DIR
)

# Predefined suggestions for the Elder strategy chatbot
CHAT_SUGGESTIONS = [
    # Strategy concepts
    "What is the Triple Screen Trading System?",
    "What are the three screens in Elder's strategy?",
    "What type of trader is Elder's strategy suited for?",
    "How does the Triple Screen strategy filter signals?",
    # Technical indicators
    "How does the RSI indicator work?",
    "How should I set RSI parameters?",
    "What are the RSI overbought and oversold thresholds?",
    "How should I set MACD parameters?",
    "How does MACD generate buy and sell signals?",
    "What do the MACD fast and slow lines represent?",
    "How to choose short-term and long-term EMA windows?",
    "What is the difference between EMA and SMA?",
    # Trading rules
    "When should I enter a long position?",
    "When should I enter a short position?",
    "What are the three conditions for going long?",
    "What are the three conditions for going short?",
    "When should I close a long position?",
    "When should I close a short position?",
    # Risk management
    "What is Elder's 2% Rule?",
    "What is Elder's 6% Rule?",
    "How should I set a stop loss?",
    "What is the maximum risk per trade?",
    "How to calculate position size?",
    # Backtest analysis
    "How to interpret maximum drawdown?",
    "How to understand the Sharpe Ratio?",
    "What is the Calmar Ratio?",
    "How to avoid overfitting in backtesting?",
    "Are backtest results reliable?",
    "How to optimize backtest results?",
    # Psychology & books
    "What are the key points of trading psychology?",
    "How to overcome fear in trading?",
    "How to avoid overtrading?",
    "What trading books do you recommend?",
    "What books has Alexander Elder written?",
]


def ghost_autocomplete(
    suggestions: List[str] = None,
    placeholder: str = "Type a keyword, e.g.: RSI, MACD, stop loss, trend...",
    key: str = "ghost_autocomplete",
) -> Optional[str]:
    """
    Render the ghost text autocomplete input component.

    Returns the submitted text on a new submission, or None otherwise.
    Deduplicates across Streamlit reruns so the same submission is not
    returned more than once (custom components persist their last value).
    """
    if suggestions is None:
        suggestions = CHAT_SUGGESTIONS

    raw = _ghost_component(
        suggestions=suggestions,
        placeholder=placeholder,
        key=key,
        default=None,
    )
    if not raw or not isinstance(raw, dict):
        return None

    sub_id = raw.get("id")
    text = raw.get("text", "").strip() or None
    if not text:
        return None

    # Suppress if this submission ID was already processed
    state_key = f"_ghost_sub_id_{key}"
    if sub_id and sub_id == st.session_state.get(state_key):
        return None

    st.session_state[state_key] = sub_id
    return text


# ★ ============================================================
# ★ Helper: Render chat interface with ghost autocomplete
# ★ ============================================================

def render_chat_input_section() -> Optional[str]:
    """
    Render the complete chat input section:
    - Ghost autocomplete (primary)
    - st.chat_input (fallback)

    Returns the user's question, or None.
    """
    st.divider()

    # Primary: Ghost text autocomplete
    submitted = ghost_autocomplete()

    # Fallback: Standard Streamlit chat input
    fallback = st.chat_input("Or type your question here...")

    # Return whichever has a value
    return submitted or fallback