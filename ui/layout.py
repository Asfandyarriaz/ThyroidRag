import streamlit as st

def setup_page():
    st.set_page_config(
        page_title="Thyroid Cancer RAG Assistant",
        page_icon="🩺",
        layout="wide"
    )

def inject_custom_css():
    st.markdown("""
    <style>
        /* =========================
           Global (ChatGPT-like dark)
           ========================= */
        html, body, .stApp {
            background: #111827 !important;   /* dark gray (not pitch black) */
            color: #E5E7EB !important;        /* light text */
        }

        #MainMenu, header, footer { visibility: hidden; }

        .block-container {
            max-width: 950px;
            padding-top: 1rem;
            padding-bottom: 6.5rem; /* space for input */
        }

        /* Title */
        .main-title {
            text-align: center;
            color: #E5E7EB;
            font-size: 2.0rem;
            font-weight: 700;
            margin: 0.5rem 0 0.25rem 0;
        }
        .subtitle {
            text-align: center;
            color: #9CA3AF;
            margin: 0 0 1.25rem 0;
            font-size: 0.95rem;
        }

        /* =========================
           Chat bubbles
           ========================= */
        .user-message, .bot-message {
            display: flex;
            margin: 12px 0;
            gap: 10px;
            align-items: flex-start;
        }
        .user-message { justify-content: flex-end; }
        .bot-message  { justify-content: flex-start; }

        .user-icon, .bot-icon {
            font-size: 1.2rem;
            opacity: 0.95;
            margin-top: 2px;
        }

        .user-content, .bot-content {
            padding: 12px 14px;
            border-radius: 14px;
            max-width: 75%;
            border: 1px solid rgba(255,255,255,0.10);
            white-space: pre-wrap;
            line-height: 1.45;
        }

        /* User bubble */
        .user-content {
            background: #1F2937;   /* slate */
            color: #E5E7EB;
            border-top-right-radius: 6px;
        }

        /* Assistant bubble */
        .bot-content {
            background: #0B1220;   /* slightly darker */
            color: #E5E7EB;
            border-top-left-radius: 6px;
        }

        /* Thinking indicator */
        .typing-indicator {
            color: #9CA3AF;
            font-style: italic;
            font-size: 0.95rem;
            animation: typing 1.5s infinite;
        }
        @keyframes typing { 0%, 100% {opacity: 0.35;} 50% {opacity: 1;} }

        /* Make markdown readable */
        .stMarkdown, .stMarkdown p, .stMarkdown li {
            color: #E5E7EB !important;
        }
        a { color: #7DD3FC !important; }

        /* =========================
           Chat input (fix white wrapper + full dark background)
           ========================= */

        /* Remove default backgrounds around chat input */
        [data-testid="stChatInput"] {
            background: transparent !important;
        }

        /* BaseWeb wrapper that often appears as a white rectangle */
        [data-testid="stChatInput"] [data-baseweb="textarea"] {
            background: #0B1220 !important; /* dark background covers whole input */
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        /* Make textarea itself transparent so wrapper background shows fully */
        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: #E5E7EB !important;
            caret-color: #E5E7EB !important;
            width: 100% !important;
        }

        /* Placeholder */
        [data-testid="stChatInput"] textarea::placeholder {
            color: #9CA3AF !important;
            opacity: 1 !important;
        }

        /* Focus styling (on wrapper) */
        [data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
            border: 1px solid rgba(255,255,255,0.28) !important;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.10) !important;
        }

        /* Extra wrappers sometimes add background */
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div {
            background: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)

def page_title():
    st.markdown(
        '<h1 class="main-title">🩺 Thyroid Cancer RAG Assistant</h1>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Ask questions and get evidence-grounded answers from retrieved literature excerpts.</div>',
        unsafe_allow_html=True
    )
