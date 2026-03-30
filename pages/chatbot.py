"""
RAG Chatbot — Visualises every stage of the retrieval-augmented generation pipeline.
"""

import sys, os, time
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag import load_or_build_vector_store, retrieve, build_prompt, generate_response

st.set_page_config(page_title="RAG Pipeline Explorer", page_icon="data/elder_jpa.png", layout="wide")

# ── RAG init ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading RAG index …")
def init_rag():
    return load_or_build_vector_store()

vector_store = init_rag()

# ── CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide default Streamlit page nav */
[data-testid="stSidebarNav"] { display: none !important; }

/* ---------- header ---------- */
.page-title{text-align:center;padding:1.2rem 0 .4rem}
.page-title h1{font-size:1.75rem;margin:0}
.page-title p{color:#888;font-size:.88rem;margin:.25rem 0 0}

/* ---------- pipeline overview diagram ---------- */
.pipeline-overview{
    display:flex;align-items:center;justify-content:center;gap:0;
    padding:.7rem 1rem;margin:0 auto 1.2rem;
    background:linear-gradient(135deg,#f8f9fb,#eef1f5);border-radius:10px;
    flex-wrap:wrap;max-width:900px;
}
.po-step{
    display:flex;align-items:center;gap:.45rem;
    padding:.35rem .7rem;border-radius:6px;font-size:.82rem;
    color:#444;white-space:nowrap;
}
.po-step .num{
    display:inline-flex;align-items:center;justify-content:center;
    width:22px;height:22px;border-radius:50%;
    font-size:.72rem;font-weight:700;color:#fff;flex-shrink:0;
}
.po-arrow{color:#bbb;font-size:1.1rem;margin:0 .15rem}
.po-step.active{font-weight:600;color:#111}

/* step colours */
.c1 .num{background:#2196F3} .c2 .num{background:#00897B}
.c3 .num{background:#FF9800} .c4 .num{background:#7B1FA2}
.c5 .num{background:#E91E63}

/* ---------- step card ---------- */
.step-card{
    border-radius:10px;padding:1rem 1.1rem;margin-bottom:.85rem;
    border:1px solid #e4e7ec;background:#fff;
}
.step-card .step-header{
    display:flex;align-items:center;gap:.55rem;margin-bottom:.6rem;
}
.step-card .step-badge{
    display:inline-flex;align-items:center;justify-content:center;
    width:28px;height:28px;border-radius:50%;
    font-size:.78rem;font-weight:700;color:#fff;flex-shrink:0;
}
.step-card .step-title{font-weight:600;font-size:.95rem;color:#222}
.step-card .step-subtitle{font-size:.78rem;color:#888;margin-left:auto}

/* chunk card */
.chunk{
    background:#f9fafb;border:1px solid #eaecf0;border-radius:8px;
    padding:.65rem .8rem;margin-bottom:.45rem;font-size:.83rem;
    line-height:1.45;color:#333;
}
.chunk .chunk-id{
    display:inline-block;background:#e3f2fd;color:#1565C0;
    font-weight:700;font-size:.72rem;padding:1px 7px;border-radius:4px;
    margin-right:.4rem;
}
.chunk .chunk-meta{
    margin-top:.35rem;font-size:.73rem;color:#999;
    display:flex;gap:.8rem;
}
.chunk .chunk-meta span{display:inline-flex;align-items:center;gap:3px}

/* timing pill */
.timing{
    display:inline-block;background:#f0fdf4;color:#16a34a;
    font-size:.72rem;font-weight:600;padding:2px 8px;border-radius:20px;
    margin-left:.5rem;
}

/* response bubble */
.response-text{
    background:#f8f9fb;border-radius:10px;padding:.9rem 1rem;
    font-size:.9rem;line-height:1.6;color:#222;
}
</style>
""", unsafe_allow_html=True)

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div class="page-title">
    <h1>RAG Pipeline Explorer</h1>
    <p>Ask a question and watch every stage of the Retrieval-Augmented Generation pipeline unfold.</p>
</div>
""", unsafe_allow_html=True)

# ── Pipeline overview (static diagram) ─────────────────────
st.markdown("""
<div class="pipeline-overview">
    <div class="po-step c1"><span class="num">1</span>Query</div>
    <span class="po-arrow">&#10132;</span>
    <div class="po-step c2"><span class="num">2</span>Embed</div>
    <span class="po-arrow">&#10132;</span>
    <div class="po-step c3"><span class="num">3</span>Retrieve Top-K</div>
    <span class="po-arrow">&#10132;</span>
    <div class="po-step c4"><span class="num">4</span>Build Prompt</div>
    <span class="po-arrow">&#10132;</span>
    <div class="po-step c5"><span class="num">5</span>LLM Generate</div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar Controls ────────────────────────────────────────
with st.sidebar:
    from components import render_sidebar_header
    render_sidebar_header()
    st.header("RAG Settings")
    model_choice = st.selectbox(
        "LLM Model",
        ["Llama 3.3 70B (Groq)", "Llama 3.1 8B (Groq)", "Mixtral 8x7B (Groq)", "GPT-4o-mini (OpenAI)"],
    )
    top_k = st.selectbox(
        "Retrieval Top-K",
        [3, 5, 7, 10],
        index=1,
        format_func=lambda x: f"Top-{x} chunks",
    )
    st.divider()
    use_dashboard = st.checkbox(
        "Inject dashboard context",
        value=False,
        help="Include backtest results from the Dashboard page as additional context.",
    )
    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["debug_chat_history"] = []
        st.rerun()


dashboard_ctx = None
if use_dashboard and "backtest_results" in st.session_state:
    dashboard_ctx = st.session_state["backtest_results"]

# ── Session state ───────────────────────────────────────────
if "debug_chat_history" not in st.session_state:
    st.session_state["debug_chat_history"] = []

# ── Helpers ─────────────────────────────────────────────────

def render_pipeline(entry, expanded=False):
    """Render a full pipeline trace for one Q&A pair."""

    # Step 1 — Query
    st.markdown(
        f'<div class="step-card">'
        f'<div class="step-header">'
        f'<span class="step-badge" style="background:#2196F3">1</span>'
        f'<span class="step-title">User Query</span>'
        f'</div>'
        f'<span style="font-size:.9rem">{entry["query"]}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Step 2 — Embedding
    st.markdown(
        f'<div class="step-card">'
        f'<div class="step-header">'
        f'<span class="step-badge" style="background:#00897B">2</span>'
        f'<span class="step-title">Embed Query</span>'
        f'<span class="step-subtitle">Model: bge-small-en-v1.5 &nbsp;|&nbsp; Dim: 384</span>'
        f'</div>'
        f'<span style="font-size:.82rem;color:#555">'
        f'The query is converted into a 384-dimensional dense vector using the BGE embedding model, '
        f'enabling semantic similarity search against the knowledge base.</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Step 3 — Retrieved Chunks
    chunks_html = ""
    for i, ch in enumerate(entry["retrieved"]):
        preview = ch["content"][:300] + ("…" if len(ch["content"]) > 300 else "")
        chunks_html += (
            f'<div class="chunk">'
            f'<span class="chunk-id">#{i+1}</span>{preview}'
            f'<div class="chunk-meta">'
            f'<span>Source: {ch["source"]}</span>'
            f'<span>Label: {ch["label"]}</span>'
            f'</div></div>'
        )
    t_retrieve_html = ""
    if entry.get("t_retrieve"):
        t_retrieve_html = f'<span class="timing">{entry["t_retrieve"]}</span>'
    st.markdown(
        f'<div class="step-card">'
        f'<div class="step-header">'
        f'<span class="step-badge" style="background:#FF9800">3</span>'
        f'<span class="step-title">Retrieve Top-{entry["top_k"]} Chunks</span>'
        f'<span class="step-subtitle">FAISS Flat L2 &nbsp;|&nbsp; {entry["top_k"]} results'
        f'{t_retrieve_html}'
        f'</span></div>'
        f'{chunks_html}'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Step 4 — Prompt Assembly
    with st.expander("Step 4 — Assembled Prompt", expanded=expanded):
        st.code(entry["prompt"], language="text")

    # Step 5 — LLM Response
    t_generate_html = ""
    if entry.get("t_generate"):
        t_generate_html = f'<span class="timing">{entry["t_generate"]}</span>'
    st.markdown(
        f'<div class="step-card">'
        f'<div class="step-header">'
        f'<span class="step-badge" style="background:#E91E63">5</span>'
        f'<span class="step-title">LLM Response</span>'
        f'<span class="step-subtitle">{entry["model"]} &nbsp;|&nbsp; temp 0.3 &nbsp;|&nbsp; max 1024 tokens'
        f'{t_generate_html}'
        f'</span></div>'
        f'<div class="response-text">{entry["response"]}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ── Render history ──────────────────────────────────────────
for entry in st.session_state["debug_chat_history"]:
    with st.container():
        render_pipeline(entry, expanded=False)
    st.divider()

# ── Quick-ask buttons ───────────────────────────────────────
qcols = st.columns(4)
quick_query = None
prompts = [
    ("What is Triple Screen?",
     "What is Elder's Triple Screen Trading System and how does it work?"),
    ("Explain RSI",
     "How does the RSI indicator work and how should overbought/oversold levels be set?"),
    ("2% Rule",
     "What is Elder's 2% Rule for risk management?"),
    ("Trading Psychology",
     "What does Elder say about trading psychology and emotional control?"),
]
for col, (label, full) in zip(qcols, prompts):
    with col:
        if st.button(label, use_container_width=True):
            quick_query = full

# ── Chat input ──────────────────────────────────────────────
query = st.chat_input("Ask about Elder's trading strategy …")
query = quick_query or query

if query:
    try:
        # ── Execute pipeline & time each stage ──────────────────
        with st.status("Running RAG pipeline...", expanded=True) as status:
            status.update(label="Embedding query...", state="running")
            t0 = time.perf_counter()
            retrieved_docs = retrieve(vector_store, query, k=top_k)
            t_retrieve = f"{(time.perf_counter()-t0)*1000:.0f} ms"
            status.update(label=f"Retrieved {len(retrieved_docs)} chunks ({t_retrieve})", state="running")

            prompt = build_prompt(query, retrieved_docs, dashboard_ctx)
            status.update(label=f"Generating response ({model_choice})...", state="running")

            t1 = time.perf_counter()
            response = generate_response(prompt, model_choice)
            t_generate = f"{(time.perf_counter()-t1)*1000:.0f} ms"
            status.update(label=f"Done ({t_retrieve} retrieve + {t_generate} generate)", state="complete", expanded=False)

        # ── Build entry ─────────────────────────────────────────
        entry = {
            "query": query,
            "response": response,
            "retrieved": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", "N/A"),
                    "label": doc.metadata.get("label", "N/A"),
                }
                for doc in retrieved_docs
            ],
            "prompt": prompt,
            "model": model_choice,
            "top_k": top_k,
            "t_retrieve": t_retrieve,
            "t_generate": t_generate,
        }

        st.session_state["debug_chat_history"].append(entry)
        st.rerun()
    except Exception:
        st.error("Failed to generate response. Please check your API keys in .env file.")
