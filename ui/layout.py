import streamlit as st

def setup_page():
    st.set_page_config(page_title="Thyroid Cancer RAG Assistant", page_icon="🩺", layout="wide")

def inject_custom_css():
    st.markdown("""
    <style>
        /* ---------- Global ---------- */
        .stApp {
            background: #0b0f14 !important;  /* ChatGPT-like dark */
            color: #e6edf3 !important;
        }
        #MainMenu, header, footer { visibility: hidden; }

        /* Reduce excessive top padding */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 7rem; /* space for chat input */
            max-width: 950px;
        }

        /* ---------- Title ---------- */
        .main-title {
            text-align: center;
            color: #e6edf3;
            font-size: 2.0rem;
            font-weight: 700;
            margin: 0.5rem 0 1.25rem 0;
        }
        .subtitle {
            text-align: center;
            color: #9aa4b2;
            margin-top: -0.75rem;
            margin-bottom: 1.5rem;
            font-size: 0.95rem;
        }

        /* ---------- Message bubbles ---------- */
        .user-message, .bot-message {
            display: flex;
            margin: 12px 0;
            gap: 10px;
            align-items: flex-start;
        }

        .user-message { justify-content: flex-end; }
        .bot-message  { justify-content: flex-start; }

        .user-icon, .bot-icon {
            font-size: 1.4rem;
            line-height: 1.4rem;
            opacity: 0.9;
            margin-top: 2px;
        }

        .user-content, .bot-content {
            padding: 12px 14px;
            border-radius: 14px;
            max-width: 72%;
            border: 1px solid rgba(255,255,255,0.08);
            box-shadow: none;
            white-space: pre-wrap;
        }

        /* user bubble slightly lighter */
        .user-content {
            background: #111827; /* slate-ish */
            color: #e6edf3;
            border-top-right-radius: 6px;
        }

        /* assistant bubble slightly darker */
        .bot-content {
            background: #0f172a; /* deeper slate */
            color: #e6edf3;
            border-top-left-radius: 6px;
        }

        /* ---------- Typing / thinking ---------- */
        .typing-indicator {
            display: inline-block;
            color: #9aa4b2;
            font-style: italic;
            font-size: 0.9rem;
            animation: typing 1.5s infinite;
        }
        @keyframes typing { 0%, 100% {opacity: 0.35;} 50% {opacity: 1;} }

        /* ---------- Chat input styling (best-effort, stable selectors) ---------- */
        /* Streamlit uses a textarea inside the chat input; target it directly */
        [data-testid="stChatInput"] textarea {
            background: #0f172a !important;
            color: #e6edf3 !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
            border-radius: 12px !important;
        }

        [data-testid="stChatInput"] textarea:focus {
            border: 1px solid rgba(255,255,255,0.22) !important;
            box-shadow: none !important;
            outline: none !important;
        }

        /* Buttons (if any) */
        .stButton>button {
            border-radius: 10px;
        }

        /* Links */
        a { color: #7dd3fc !important; }
    </style>
    """, unsafe_allow_html=True)

def page_title():
    st.markdown('<h1 class="main-title">🩺 Thyroid Cancer RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask questions and receive evidence-grounded answers from retrieved literature excerpts.</div>', unsafe_allow_html=True)
