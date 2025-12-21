import streamlit as st

def setup_page():
    st.set_page_config(page_title="Thyroid Cancer RAG Assistant", page_icon="🩺", layout="wide")

def inject_custom_css():
    st.markdown("""
    <style>
        /* =========================
           Global (ChatGPT-like dark)
           ========================= */
        html, body, .stApp {
            background: #111827 !important;   /* dark gray, NOT pitch black */
            color: #E5E7EB !important;        /* light gray text */
        }

        #MainMenu, header, footer { visibility: hidden; }

        .block-container {
            max-width: 950px;
            padding-top: 1rem;
            padding-bottom: 6.5rem; /* leave room for input */
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

        /* =========================
           Chat input (fix unreadable textbox)
           ========================= */

        /* The entire chat input block */
        [data-testid="stChatInput"] {
            background: transparent !important;
        }

        /* The textarea itself */
        [data-testid="stChatInput"] textarea {
            background: #0B1220 !important;
            color: #E5E7EB !important;
            caret-color: #E5E7EB !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
            border-radius: 12px !important;
        }

        /* Placeholder text */
        [data-testid="stChatInput"] textarea::placeholder {
            color: #9CA3AF !important;
            opacity: 1 !important;
        }

        /* Focus state */
        [data-testid="stChatInput"] textarea:focus {
            border: 1px solid rgba(255,255,255,0.28) !important;
            box-shadow: 0 0 0 1px rgba(255,255,255,0.10) !important;
            outline: none !important;
        }

        /* Make markdown readable */
        .stMarkdown, .stMarkdown p, .stMarkdown li {
            color: #E5E7EB !important;
        }

        a { color: #7DD3FC !important; }
    </style>
    """, unsafe_allow_html=True)

def page_title():
    st.markdown('<h1 class="main-title">🩺 Thyroid Cancer RAG Assistant</h1>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Ask questions and get evidence-grounded answers from retrieved literature excerpts.</div>', unsafe_allow_html=True)
