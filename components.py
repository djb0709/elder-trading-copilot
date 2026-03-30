"""Shared UI components across pages."""

import streamlit as st


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
            "RAG-powered trading AI built on Alexander Elder's strategies."
            "</span></div>",
            unsafe_allow_html=True,
        )
    st.page_link("dashboard.py", label="Dashboard", icon=":material/candlestick_chart:")
    st.page_link("pages/chatbot.py", label="RAG Pipeline Explorer", icon=":material/search:")
    st.toggle("Celebrations", value=True, key="anim_on", help="Toggle balloons/snow animations")
    st.divider()
