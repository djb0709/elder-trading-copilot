"""Shared UI components across pages."""

import streamlit as st


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
    st.page_link("pages/chatbot.py", label="RAG Pipeline Explorer", icon=":material/search:")
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
