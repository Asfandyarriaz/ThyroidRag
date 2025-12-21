import streamlit as st

def setup_page():
    st.set_page_config(
        page_title="Thyroid Cancer RAG Assistant",
        page_icon="🩺",
        layout="wide",
    )

def inject_custom_css():
    st.markdown("""
    <style>
        /* =========================
           Force ChatGPT-like neutral dark everywhere
           ========================= */

        :root {
            --bg: #0f0f0f;          /* ChatGPT-ish dark */
            --panel: #141414;       /* slightly lighter */
            --panel2: #1b1b1b;      /* user bubble */
            --text: #eaeaea;        /* main text */
            --muted: #a6a6a6;       /* secondary text */
            --border: rgba(255,255,255,0.10);
        }

        html, body {
            background: var(--bg) !important;
            color: var(--text) !important;
        }

        /* Streamlit containers that sometimes keep a bluish theme */
        .stApp,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            background: var(--bg) !important;
            color: var(--text) !important;
        }

        /* Optional: sidebar */
        [data-testid="stSidebar"] {
            background: var(--bg) !important;
            color: var(--text) !important;
            border-right: 1px solid var(--border) !important;
        }

        #MainMenu, header, footer { visibility: hidden; }

        .block-container {
            max-width: 950px;
            padding-top: 1rem;
            padding-bottom: 6.5rem;
        }

        /* Title */
        .main-title {
            text-align: center;
            color: var(--text);
            font-size: 2.0rem;
            font-weight: 700;
            margin: 0.5rem 0 0.25rem 0;
        }
        .subtitle {
            text-align: center;
            color: var(--muted);
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
            border: 1px solid var(--border);
            white-space: pre-wrap;
            line-height: 1.45;
        }

        /* User bubble */
        .user-content {
            background: var(--panel2);
            color: var(--text);
            border-top-right-radius: 6px;
        }

        /* Assistant bubble */
        .bot-content {
            background: var(--panel);
            color: var(--text);
            border-top-left-radius: 6px;
        }

        /* Thinking indicator */
        .typing-indicator {
            color: var(--muted);
            font-style: italic;
            font-size: 0.95rem;
            animation: typing 1.5s infinite;
        }
        @keyframes typing { 0%, 100% {opacity: 0.35;} 50% {opacity: 1;} }

        /* Make markdown readable */
        .stMarkdown, .stMarkdown p, .stMarkdown li {
            color: var(--text) !important;
        }

        /* Links muted (ChatGPT-like) */
        a { color: #7ab7ff !important; }

        /* =========================
           Chat input (no white wrapper, full dark background)
           ========================= */
        [data-testid="stChatInput"] {
            background: transparent !important;
        }

        [data-testid="stChatInput"] [data-baseweb="textarea"] {
            background: var(--panel) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: var(--text) !important;
            caret-color: var(--text) !important;
            width: 100% !important;
        }

        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--muted) !important;
            opacity: 1 !important;
        }

        [data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
            border: 1px solid rgba(255,255,255,0.28) !important;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.10) !important;
        }

        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div {
            background: transparent !important;
        }
    </style>
    """, unsafe_allow_html=True)

def page_title():
    st.markdown('<h1 class="main-title">🩺 Thyroid Cancer RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Ask questions and get evidence-grounded answers from retrieved literature excerpts.</div>',
        unsafe_allow_html=True
    )
