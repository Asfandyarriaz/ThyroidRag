# ui/layout.py
import streamlit as st


def setup_page():
    st.set_page_config(
        page_title="Thyroid Cancer RAG Assistant",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_custom_css():
    st.markdown(
        """
<style>
/* Light, clean background */
.stApp {
  background: #f6f7fb;
  color: #111827;
}

/* Main content width + bottom padding so input never overlaps */
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 7.5rem !important;
  max-width: 920px;
}

/* Hide Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }

/* Sidebar: light */
section[data-testid="stSidebar"] {
  background: #ffffff;
  border-right: 1px solid #e5e7eb;
}

/* Chat message spacing tighter (removes huge blank gaps) */
div[data-testid="stChatMessage"] p {
  margin: 0.25rem 0 !important;
}
div[data-testid="stChatMessage"] ul {
  margin: 0.25rem 0 0.25rem 1.2rem !important;
}
div[data-testid="stChatMessage"] li {
  margin: 0.15rem 0 !important;
}

/* Chat input: fixed but not overlapping */
div[data-testid="stChatInput"] {
  position: fixed;
  bottom: 0.9rem;
  left: 50%;
  transform: translateX(-50%);
  width: min(920px, calc(100% - 2rem));
  z-index: 999;
}

/* Input look */
div[data-testid="stChatInput"] textarea {
  background: #ffffff !important;
  color: #111827 !important;
  border: 1px solid #d1d5db !important;
  border-radius: 14px !important;
  box-shadow: 0 6px 18px rgba(17,24,39,0.08);
}

div[data-testid="stChatInput"] textarea::placeholder {
  color: rgba(17,24,39,0.45) !important;
}

/* Buttons: rounded */
.stButton button {
  border-radius: 12px;
}

/* Disclaimer banner */
.disclaimer {
  background: #fff7ed;
  border: 1px solid #fed7aa;
  color: #9a3412;
  border-radius: 14px;
  padding: 0.85rem 1rem;
  margin: 0.25rem 0 0.9rem 0;
  font-size: 0.95rem;
  line-height: 1.35rem;
}

/* Subtle title */
.page-title {
  margin: 0 0 0.75rem 0;
  font-weight: 750;
  color: #0f172a;
}
</style>
""",
        unsafe_allow_html=True,
    )


def page_title():
    st.markdown(
        '<h2 class="page-title">🩺 Thyroid Cancer RAG Assistant</h2>',
        unsafe_allow_html=True,
    )
