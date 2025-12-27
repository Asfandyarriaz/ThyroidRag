import streamlit as st

APP_NAME = "Thyroid Cancer RAG Assistant"

def setup_page():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* ---------- Global (ChatGPT-like light theme) ---------- */
        html, body, .stApp {
            background: #f7f7f8 !important;
            color: #111827 !important;
            font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
        }

        /* Hide Streamlit chrome */
        #MainMenu, header, footer { visibility: hidden; }

        /* Main content width + padding */
        .block-container {
            max-width: 980px;
            padding-top: 1.25rem !important;
            padding-bottom: 9rem !important;  /* IMPORTANT: prevents chat input overlap */
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background: #ffffff !important;
            border-right: 1px solid #e5e7eb !important;
        }

        /* Title */
        .main-title {
            text-align: center;
            color: #111827;
            font-size: 2.0rem;
            font-weight: 700;
            margin: 0.25rem 0 1.0rem 0;
            letter-spacing: -0.02em;
        }

        /* ---------- Chat bubbles ---------- */
        .user-message, .bot-message {
            display: flex;
            gap: 10px;
            margin: 12px 0;
            align-items: flex-start;
        }

        .user-message { justify-content: flex-end; }
        .bot-message { justify-content: flex-start; }

        .user-content, .bot-content {
            padding: 12px 14px;
            border-radius: 14px;
            max-width: 72%;
            line-height: 1.45;
            white-space: pre-wrap;
            word-wrap: break-word;
        }

        .user-content {
            background: #dbeafe; /* soft blue */
            color: #0f172a;
            border: 1px solid #bfdbfe;
        }

        .bot-content {
            background: #ffffff;
            color: #111827;
            border: 1px solid #e5e7eb;
        }

        .user-icon, .bot-icon {
            font-size: 1.25rem;
            color: #6b7280;
            margin-top: 2px;
        }

        /* ---------- Thinking indicator ---------- */
        .typing-indicator {
            display: inline-block;
            font-style: italic;
            color: #6b7280;
        }

        /* ---------- Chat input styling ---------- */
        /* Container behind chat input */
        div[data-testid="stChatInputContainer"] {
            background: #f7f7f8 !important;
            border-top: 1px solid #e5e7eb !important;
            padding-top: 0.75rem !important;
            padding-bottom: 0.75rem !important;
        }

        /* Actual input area */
        div[data-testid="stChatInput"] textarea {
            background: #ffffff !important;
            color: #111827 !important;
            border: 1px solid #d1d5db !important;
            border-radius: 14px !important;
            padding: 12px 14px !important;
            box-shadow: none !important;
        }

        /* Placeholder text */
        div[data-testid="stChatInput"] textarea::placeholder {
            color: #9ca3af !important;
        }

        /* Make the input nicely centered (like ChatGPT) */
        div[data-testid="stChatInput"] {
            max-width: 980px !important;
            margin-left: auto !important;
            margin-right: auto !important;
        }

        /* Buttons */
        .stButton>button {
            border-radius: 12px !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )

def page_title():
    st.markdown(f'<div class="main-title">🩺 {APP_NAME}</div>', unsafe_allow_html=True)
