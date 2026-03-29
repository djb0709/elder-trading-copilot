"""
RAG Chatbot Debug Page — Shows the full RAG pipeline process for each query.
"""

import sys
import os
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import (
    load_or_build_vector_store,
    retrieve, build_prompt, generate_response,
)

st.set_page_config(page_title="RAG Chatbot", layout="wide")

# ============================================================
# RAG Initialization
# ============================================================

@st.cache_resource(show_spinner="Loading RAG index...")
def init_rag():
    return load_or_build_vector_store()

vector_store = init_rag()
doc_count = 894  # pre-counted

# ============================================================
# Custom CSS
# ============================================================

st.markdown("""
<style>
    /* Header */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 1.8rem;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #888;
        font-size: 0.9rem;
    }

    /* Pipeline step cards */
    .pipeline-step {
        background: #f8f9fa;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #4CAF50;
    }
    .pipeline-step.step-retrieve { border-left-color: #2196F3; }
    .pipeline-step.step-prompt { border-left-color: #FF9800; }
    .pipeline-step.step-generate { border-left-color: #9C27B0; }

    /* Chunk card */
    .chunk-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
        padding: 0.7rem;
        margin-bottom: 0.4rem;
        font-size: 0.85rem;
    }
    .chunk-meta {
        color: #999;
        font-size: 0.75rem;
        margin-top: 0.3rem;
    }

    /* Stats bar */
    .stats-bar {
        display: flex;
        gap: 1.5rem;
        justify-content: center;
        padding: 0.5rem;
        background: #f0f2f6;
        border-radius: 8px;
        margin-bottom: 1rem;
        font-size: 0.85rem;
    }
    .stats-bar span {
        color: #555;
    }
    .stats-bar strong {
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Header
# ============================================================

st.markdown("""
<div class="main-header">
    <h1>Elder Strategy RAG Chatbot</h1>
    <p>Ask questions about Alexander Elder's trading strategies. See the full RAG pipeline for every answer.</p>
</div>
""", unsafe_allow_html=True)

# Stats bar
st.markdown(f"""
<div class="stats-bar">
    <span>Documents: <strong>{doc_count}</strong></span>
    <span>Embedding: <strong>bge-small-en-v1.5</strong></span>
    <span>Index: <strong>FAISS (Flat L2)</strong></span>
</div>
""", unsafe_allow_html=True)

# ============================================================
# Top controls (inline, not sidebar)
# ============================================================

ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 3, 3])
with ctrl1:
    model_choice = st.selectbox(
        "Model",
        ["Llama 3.3 (Groq)", "GPT-4o-mini (OpenAI)"],
        label_visibility="collapsed",
    )
with ctrl2:
    top_k = st.selectbox(
        "Top-K", [3, 5, 7, 10], index=1,
        format_func=lambda x: f"Top-{x} retrieval",
        label_visibility="collapsed",
    )
with ctrl3:
    groq_key = st.text_input(
        "Groq Key", type="password",
        placeholder="Groq API Key",
        label_visibility="collapsed",
    )
with ctrl4:
    openai_key = st.text_input(
        "OpenAI Key", type="password",
        placeholder="OpenAI API Key",
        label_visibility="collapsed",
    )

# Optional dashboard context
use_dashboard = st.checkbox(
    "Inject Dashboard Context (run backtest on main page first)",
    value=False,
)
dashboard_ctx = None
if use_dashboard and "backtest_results" in st.session_state:
    dashboard_ctx = st.session_state["backtest_results"]

st.divider()

# ============================================================
# Chat + Pipeline Display
# ============================================================

if "debug_chat_history" not in st.session_state:
    st.session_state["debug_chat_history"] = []

# Display history
for entry in st.session_state["debug_chat_history"]:
    # User message
    with st.chat_message("user"):
        st.write(entry["query"])

    # Two columns: response (left) + pipeline (right)
    col_resp, col_pipe = st.columns([1, 1])

    with col_resp:
        with st.chat_message("assistant"):
            st.write(entry["response"])

    with col_pipe:
        with st.expander("Pipeline Details", expanded=False):
            # Retrieved chunks
            st.markdown("**Retrieved Chunks:**")
            for i, chunk in enumerate(entry["retrieved"]):
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<strong>#{i+1}</strong> '
                    f'{chunk["content"][:250]}{"..." if len(chunk["content"]) > 250 else ""}'
                    f'<div class="chunk-meta">'
                    f'{chunk["source"]} | {chunk["label"]}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            # Prompt
            st.markdown("**Prompt sent to LLM:**")
            st.code(
                entry["prompt"][:1500]
                + ("..." if len(entry["prompt"]) > 1500 else ""),
                language="text",
            )
            st.caption(f"Model: {entry['model']} | Top-K: {entry['top_k']}")

# ============================================================
# Input
# ============================================================

# Quick ask row
qcol1, qcol2, qcol3, qcol4 = st.columns(4)
quick_query = None
with qcol1:
    if st.button("What is Triple Screen?", use_container_width=True):
        quick_query = "What is Elder's Triple Screen Trading System and how does it work?"
with qcol2:
    if st.button("Explain RSI", use_container_width=True):
        quick_query = "How does the RSI indicator work and how should overbought/oversold levels be set?"
with qcol3:
    if st.button("2% Rule", use_container_width=True):
        quick_query = "What is Elder's 2% Rule for risk management?"
with qcol4:
    if st.button("Psychology", use_container_width=True):
        quick_query = "What does Elder say about trading psychology and emotional control?"

query = st.chat_input("Ask about Elder's trading strategy...")
query = quick_query or query

if query:
    # User message
    with st.chat_message("user"):
        st.write(query)

    # Run pipeline
    with st.spinner("Running RAG pipeline..."):
        retrieved_docs = retrieve(vector_store, query, k=top_k)
        prompt = build_prompt(query, retrieved_docs, dashboard_ctx)
        response = generate_response(
            prompt, model_choice, groq_key, openai_key
        )

    # Display: response (left) + pipeline (right)
    col_resp, col_pipe = st.columns([1, 1])

    with col_resp:
        with st.chat_message("assistant"):
            st.write(response)

    with col_pipe:
        with st.expander("Pipeline Details", expanded=True):
            # Step 1: Retrieved
            st.markdown("**Step 1 — Retrieved Chunks:**")
            for i, doc in enumerate(retrieved_docs):
                content = doc.page_content
                source = doc.metadata.get("source", "N/A")
                label = doc.metadata.get("label", "N/A")
                st.markdown(
                    f'<div class="chunk-card">'
                    f'<strong>#{i+1}</strong> '
                    f'{content[:250]}{"..." if len(content) > 250 else ""}'
                    f'<div class="chunk-meta">'
                    f'{source} | {label}'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            # Step 2: Prompt
            st.markdown("**Step 2 — Prompt sent to LLM:**")
            st.code(
                prompt[:1500]
                + ("..." if len(prompt) > 1500 else ""),
                language="text",
            )

            # Step 3: Generation info
            st.markdown(f"**Step 3 — Generated by:** `{model_choice}` | Top-K: `{top_k}`")

    # Save to history
    st.session_state["debug_chat_history"].append({
        "query": query,
        "response": response,
        "retrieved": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", ""),
                "label": doc.metadata.get("label", ""),
            }
            for doc in retrieved_docs
        ],
        "prompt": prompt,
        "model": model_choice,
        "top_k": top_k,
    })
    st.rerun()
