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
/* --- Overall page (ChatGPT-ish, not pitch black) --- */
.stApp {
  background: #0b1220; /* deep navy */
  color: #e5e7eb;
}

/* Main content width + bottom padding so input never overlaps */
.block-container {
  padding-top: 1.2rem;
  padding-bottom: 7.5rem !important;
  max-width: 900px;
}

/* Hide Streamlit chrome */
#MainMenu, header, footer { visibility: hidden; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #0a1020;
  border-right: 1px solid rgba(255,255,255,0.08);
}

/* Chat message bubbles */
div[data-testid="stChatMessage"] {
  border-radius: 14px;
  padding: 0.25rem 0.25rem;
}

/* Improve markdown spacing (remove huge blank gaps) */
div[data-testid="stChatMessage"] p {
  margin: 0.25rem 0 !important;
}
div[data-testid="stChatMessage"] ul {
  margin: 0.25rem 0 0.25rem 1.2rem !important;
}
div[data-testid="stChatMessage"] li {
  margin: 0.15rem 0 !important;
}

/* Chat input: consistent dark surface, no weird white rectangle */
div[data-testid="stChatInput"] {
  position: fixed;
  bottom: 0.75rem;
  left: 50%;
  transform: translateX(-50%);
  width: min(900px, calc(100% - 2rem));
  z-index: 999;
}

div[data-testid="stChatInput"] textarea {
  background: #0f1a33 !important;
  color: #e5e7eb !important;
  border: 1px solid rgba(255,255,255,0.14) !important;
  border-radius: 14px !important;
}

div[data-testid="stChatInput"] textarea::placeholder {
  color: rgba(229,231,235,0.55) !important;
}

/* Buttons/toggles */
.stButton button {
  border-radius: 12px;
}

/* Small pill for disclaimer */
.disclaimer {
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 14px;
  padding: 0.8rem 1rem;
  margin-bottom: 0.5rem;
  font-size: 0.95rem;
  line-height: 1.35rem;
}
</style>
""",
        unsafe_allow_html=True,
    )


def page_title():
    st.markdown(
        """
<h2 style="margin: 0 0 0.75rem 0; font-weight: 700;">
🩺 Thyroid Cancer RAG Assistant
</h2>
""",
        unsafe_allow_html=True,
    )
