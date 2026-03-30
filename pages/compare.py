"""
RAG Pipeline Comparison — Side-by-side comparison of embeddings, LLMs, and top-k.
"""

import sys, os, time
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import (
    load_or_build_vector_store, retrieve, build_prompt,
    generate_response, build_vector_store_for_model, EMBEDDING_MODELS,
)

st.set_page_config(page_title="RAG Compare", page_icon="data/elder_jpa.png", layout="wide")

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebarNav"] { display: none !important; }
[data-testid="stSidebar"] hr { margin-top: 10px !important; margin-bottom: 10px !important; }
.compare-header {
    background: linear-gradient(135deg, #f8f9fb, #eef1f5);
    border-radius: 10px; padding: 0.6rem 1rem; margin-bottom: 0.5rem;
    font-weight: 600; font-size: 0.85rem; color: #333;
    border-left: 3px solid #667eea;
}
.chunk-card {
    background: #f9fafb; border: 1px solid #eaecf0; border-radius: 8px;
    padding: 0.6rem 0.8rem; margin-bottom: 0.4rem; font-size: 0.82rem;
    line-height: 1.45; color: #333;
}
.chunk-meta { margin-top: 0.3rem; font-size: 0.72rem; color: #999; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    from components import render_sidebar_header
    render_sidebar_header()

    st.header("Compare Settings")
    compare_mode = st.radio(
        "Compare",
        ["Embedding Models", "LLM Models", "Top-K Values"],
        help="Choose what to compare side-by-side.",
    )

    if compare_mode == "Embedding Models":
        options = list(EMBEDDING_MODELS.keys())
        left = st.selectbox("Left Embedding", options, index=0)
        right = st.selectbox("Right Embedding", options, index=1)
        top_k = st.selectbox("Top-K (shared)", [3, 5, 7, 10], index=1)

    elif compare_mode == "LLM Models":
        all_llms = ["Llama 3.3 70B (Groq)", "Llama 3.1 8B (Groq)", "Mixtral 8x7B (Groq)", "GPT-4o-mini (OpenAI)"]
        left = st.selectbox("Left LLM", all_llms, index=0)
        right = st.selectbox("Right LLM", all_llms, index=3)
        embedding_for_llm = st.selectbox(
            "Embedding (shared)",
            list(EMBEDDING_MODELS.keys()),
            index=0,
        )
        top_k = st.selectbox("Top-K (shared)", [3, 5, 7, 10], index=1)

    else:  # Top-K
        left = st.selectbox("Left Top-K", [3, 5, 7, 10], index=0)
        right = st.selectbox("Right Top-K", [3, 5, 7, 10], index=2)
        embedding_for_topk = st.selectbox(
            "Embedding (shared)",
            list(EMBEDDING_MODELS.keys()),
            index=0,
        )
        llm = st.selectbox(
            "LLM (shared)",
            ["Llama 3.1 8B (Groq)", "Llama 3.3 70B (Groq)", "Mixtral 8x7B (Groq)", "GPT-4o-mini (OpenAI)"],
        )

# ── Cache vector stores ─────────────────────────────────────
@st.cache_resource(show_spinner="Loading default index...")
def get_default_vs():
    return load_or_build_vector_store()

@st.cache_resource(show_spinner="Building index...")
def get_vs_for_model(key):
    return build_vector_store_for_model(key)

# ── Header ──────────────────────────────────────────────────
st.markdown("## RAG Pipeline Comparison")
st.markdown(
    f"<span style='color:#888; font-size:0.88rem;'>"
    f"Comparing: <strong>{compare_mode}</strong> &mdash; "
    f"Left: <strong>{left}</strong> vs Right: <strong>{right}</strong>"
    f"</span>",
    unsafe_allow_html=True,
)

# ── Query input ─────────────────────────────────────────────
query = st.chat_input("Enter a query to compare...")

if query:
    col_left, col_right = st.columns(2)

    def run_pipeline(vs, q, k, llm_choice, label, skip_llm=False):
        """Run retrieve + optionally generate."""
        t0 = time.perf_counter()
        docs = retrieve(vs, q, k=k)
        t_ret = time.perf_counter() - t0

        resp = None
        t_gen = 0
        if not skip_llm:
            prompt = build_prompt(q, docs)
            t1 = time.perf_counter()
            resp = generate_response(prompt, llm_choice)
            t_gen = time.perf_counter() - t1

        return {
            "docs": docs,
            "response": resp,
            "t_retrieve": f"{t_ret*1000:.0f}ms",
            "t_generate": f"{t_gen*1000:.0f}ms" if not skip_llm else None,
            "label": label,
            "llm": llm_choice if not skip_llm else None,
            "k": k,
        }

    def render_result(col, result):
        """Render one side of the comparison."""
        with col:
            st.markdown(f'<div class="compare-header">{result["label"]}</div>', unsafe_allow_html=True)

            # Timing
            timing = f'Retrieve: {result["t_retrieve"]} | Top-K: {result["k"]}'
            if result.get("t_generate"):
                timing += f' | Generate: {result["t_generate"]}'
            if result.get("llm"):
                timing += f' | LLM: {result["llm"].split(" (")[0]}'
            st.caption(timing)

            # Retrieved chunks
            with st.expander(f'Retrieved {len(result["docs"])} chunks', expanded=True):
                for i, doc in enumerate(result["docs"]):
                    content = doc.page_content[:250] + ("..." if len(doc.page_content) > 250 else "")
                    source = doc.metadata.get("source", "N/A")
                    label = doc.metadata.get("label", "N/A")
                    st.markdown(
                        f'<div class="chunk-card">'
                        f'<strong>#{i+1}</strong> {content}'
                        f'<div class="chunk-meta">Source: {source} | Label: {label}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # Response (only if LLM was used)
            if result.get("response"):
                st.markdown("**Response:**")
                st.write(result["response"])

    # ── Run both pipelines ──────────────────────────────────
    with st.status("Running both pipelines...", expanded=True) as status:
        if compare_mode == "Embedding Models":
            status.update(label=f"Building left: {left}...")
            vs_left = get_default_vs() if left == "bge-small-en-v1.5" else get_vs_for_model(left)
            status.update(label=f"Building right: {right}...")
            vs_right = get_default_vs() if right == "bge-small-en-v1.5" else get_vs_for_model(right)

            status.update(label="Retrieving left...")
            r_left = run_pipeline(vs_left, query, top_k, None, f"Embedding: {left}", skip_llm=True)
            status.update(label="Retrieving right...")
            r_right = run_pipeline(vs_right, query, top_k, None, f"Embedding: {right}", skip_llm=True)

        elif compare_mode == "LLM Models":
            status.update(label=f"Loading {embedding_for_llm} index...")
            vs = get_default_vs() if embedding_for_llm == "bge-small-en-v1.5" else get_vs_for_model(embedding_for_llm)
            status.update(label=f"Running {left}...")
            r_left = run_pipeline(vs, query, top_k, left, f"LLM: {left.split(' (')[0]}")
            status.update(label=f"Running {right}...")
            r_right = run_pipeline(vs, query, top_k, right, f"LLM: {right.split(' (')[0]}")

        else:  # Top-K
            status.update(label=f"Loading {embedding_for_topk} index...")
            vs = get_default_vs() if embedding_for_topk == "bge-small-en-v1.5" else get_vs_for_model(embedding_for_topk)
            status.update(label=f"Running Top-{left}...")
            r_left = run_pipeline(vs, query, left, llm, f"Top-K: {left}")
            status.update(label=f"Running Top-{right}...")
            r_right = run_pipeline(vs, query, right, llm, f"Top-K: {right}")

        status.update(label="Done!", state="complete", expanded=False)

    render_result(col_left, r_left)
    render_result(col_right, r_right)
