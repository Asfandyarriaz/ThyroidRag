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
           ChatGPT-like neutral dark theme
           ========================= */
        :root {
            --bg: #0f0f0f;               /* main background */
            --panel: #141414;            /* assistant bubble + input wrapper */
            --panel2: #1b1b1b;           /* user bubble */
            --text: #f9fafb;             /* near-white */
            --muted: #c7ced6;            /* placeholder/secondary */
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
            padding-bottom: 6.5rem; /* space for input */
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
            color: rgba(249,250,251,0.72);
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
            color: rgba(249,250,251,0.70);
            font-style: italic;
            font-size: 0.95rem;
            animation: typing 1.5s infinite;
        }
        @keyframes typing { 0%, 100% {opacity: 0.35;} 50% {opacity: 1;} }

        /* Make markdown readable */
        .stMarkdown, .stMarkdown p, .stMarkdown li {
            color: var(--text) !important;
        }

        a { color: #7ab7ff !important; }

        /* =========================
           Chat input (no white wrapper, full dark, white typing)
           ========================= */

        /* remove any default background around chat input */
        [data-testid="stChatInput"] {
            background: transparent !important;
        }

        /* BaseWeb wrapper (this is often the “white rectangle”) */
        [data-testid="stChatInput"] [data-baseweb="textarea"] {
            background: var(--panel) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 12px !important;
            box-shadow: none !important;
        }

        /* textarea itself: transparent so wrapper background covers full area */
        [data-testid="stChatInput"] textarea {
            background: transparent !important;
            color: var(--text) !important;
            caret-color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important; /* fixes grey text */
            font-weight: 500 !important;
            width: 100% !important;
        }

        /* placeholder: slightly dimmer but readable */
        [data-testid="stChatInput"] textarea::placeholder {
            color: var(--muted) !important;
            opacity: 1 !important;
        }

        /* focus styling on wrapper */
        [data-testid="stChatInput"] [data-baseweb="textarea"]:focus-within {
            border: 1px solid rgba(255,255,255,0.28) !important;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.10) !important;
        }

        /* sometimes extra wrappers add background */
        [data-testid="stChatInput"] > div,
        [data-testid="stChatInput"] > div > div {
            background: transparent !important;
        }

        /* ensure nothing inside wrapper forces grey text */
        [data-testid="stChatInput"] [data-baseweb="textarea"] * {
            color: var(--text) !important;
            -webkit-text-fill-color: var(--text) !important;
        }
    </style>
    """, unsafe_allow_html=True)

def page_title():
    st.markdown('<h1 class="main-title">🩺 Thyroid Cancer RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Ask questions and get evidence-grounded answers from retrieved literature excerpts.</div>',
        unsafe_allow_html=True
    )
